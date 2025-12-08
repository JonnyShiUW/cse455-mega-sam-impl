"""
MegaSaM Stage 1: Monodepth Pre-computation
Computes monocular depth using DepthAnything and UniDepth.
"""

import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# ============================================================================
# DepthAnything Model Components
# ============================================================================

class FeatureFusionBlock(nn.Module):
    """Feature fusion block for DPT decoder."""
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.align_corners = align_corners
        self.expand = expand
        out_features = features if not expand else features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])
        output = self.resConfUnit1(output)
        output = self.resConfUnit2(output)
        if size is not None:
            output = F.interpolate(output, size=size, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class ResidualConvUnit(nn.Module):
    """Residual convolution unit."""
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class DPTHead(nn.Module):
    """DPT head for depth prediction."""
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super().__init__()
        self.nclass = nclass
        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
            for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        self.scratch = self._make_scratch(out_channels, features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def _make_scratch(self, out_channels, features, use_bn):
        scratch = nn.Module()
        scratch.layer1_rn = nn.Conv2d(out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        scratch.layer2_rn = nn.Conv2d(out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        scratch.layer3_rn = nn.Conv2d(out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        scratch.layer4_rn = nn.Conv2d(out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)
        scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)
        scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)
        scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)
        scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(False), deconv=False, bn=use_bn, expand=False, align_corners=True)
        return scratch

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode='bilinear', align_corners=True)
        out = self.scratch.output_conv2(out)

        return out


class DPT_DINOv2(nn.Module):
    """DepthAnything model using DINOv2 backbone and DPT decoder."""
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super().__init__()
        assert encoder in ['vits', 'vitb', 'vitl']

        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', f'dinov2_{encoder}14', source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', f'dinov2_{encoder}14')

        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1)


# ============================================================================
# Preprocessing Transforms
# ============================================================================

class Resize:
    """Resize image while maintaining aspect ratio."""
    def __init__(self, width, height, resize_target=True, keep_aspect_ratio=True, ensure_multiple_of=1, resize_method='upper_bound', image_interpolation_method=cv2.INTER_CUBIC):
        self.width = width
        self.height = height
        self.resize_target = resize_target
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resize_method = resize_method
        self.image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.ensure_multiple_of) * self.ensure_multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.ensure_multiple_of) * self.ensure_multiple_of).astype(int)
        if y < min_val:
            y = (np.ceil(x / self.ensure_multiple_of) * self.ensure_multiple_of).astype(int)
        return y

    def get_size(self, width, height):
        scale_height = self.height / height
        scale_width = self.width / width

        if self.keep_aspect_ratio:
            if self.resize_method == 'upper_bound':
                scale_height = max(scale_width, scale_height)
                scale_width = scale_height
            elif self.resize_method == 'lower_bound':
                scale_height = min(scale_width, scale_height)
                scale_width = scale_height
            elif self.resize_method == 'minimal':
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height

        new_height = self.constrain_to_multiple_of(scale_height * height, min_val=1)
        new_width = self.constrain_to_multiple_of(scale_width * width, min_val=1)

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample['image'].shape[1], sample['image'].shape[0])
        sample['image'] = cv2.resize(sample['image'], (width, height), interpolation=self.image_interpolation_method)
        return sample


class NormalizeImage:
    """Normalize image using ImageNet statistics."""
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        sample['image'] = (sample['image'] - self.mean) / self.std
        return sample


class PrepareForNet:
    """Prepare image for neural network (transpose to CHW format)."""
    def __call__(self, sample):
        sample['image'] = np.transpose(sample['image'], (2, 0, 1))
        return sample


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


# ============================================================================
# DepthAnything Pipeline
# ============================================================================

def run_depth_anything(img_path, outdir, checkpoint_path, encoder='vitl', device='cuda', use_cache=True):
    """Run DepthAnything on a sequence of images."""
    filenames = sorted(glob.glob(os.path.join(img_path, '*.png')))
    filenames += sorted(glob.glob(os.path.join(img_path, '*.jpg')))

    os.makedirs(outdir, exist_ok=True)

    if use_cache:
        uncached_filenames = []
        cached_count = 0
        for filename in filenames:
            output_filename = os.path.join(outdir, os.path.basename(filename)[:-4] + '.npy')
            if os.path.exists(output_filename):
                cached_count += 1
            else:
                uncached_filenames.append(filename)

        print(f'Found {cached_count} cached results, processing {len(uncached_filenames)} new images...')
        filenames = uncached_filenames

        if len(filenames) == 0:
            print(f'All depth predictions already exist in {outdir}')
            return outdir
    else:
        print(f'Processing {len(filenames)} images...')

    # Initialize model
    if encoder == 'vits':
        depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub=False)
    elif encoder == 'vitb':
        depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], localhub=False)
    else:
        depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], localhub=False)

    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    depth_anything = depth_anything.to(device)
    depth_anything.eval()

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print(f'DepthAnything model loaded with {total_params / 1e6:.2f}M parameters')

    transform = Compose([
        Resize(width=768, height=768, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method='upper_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)[..., :3]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).float().to(device)

        with torch.no_grad():
            depth = depth_anything(image)

        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth_npy = depth.cpu().numpy().astype(np.float32)

        output_filename = os.path.join(outdir, os.path.basename(filename)[:-4] + '.npy')
        np.save(output_filename, depth_npy)

    print(f'DepthAnything predictions saved to {outdir}')
    return outdir


# ============================================================================
# UniDepth Pipeline
# ============================================================================

def run_unidepth(img_path, outdir, scene_name, long_dim=640, device='cuda', use_cache=True):
    """Run UniDepth on a sequence of images."""
    try:
        from unidepth.models import UniDepthV2
    except ImportError:
        print("Warning: UniDepth not installed. Please install it.")
        return None

    outdir_scene = os.path.join(outdir, scene_name)
    os.makedirs(outdir_scene, exist_ok=True)

    img_path_list = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
    img_path_list += sorted(glob.glob(os.path.join(img_path, "*.png")))

    if use_cache:
        uncached_img_paths = []
        cached_count = 0
        for img_path_single in img_path_list:
            output_filename = os.path.join(outdir_scene, os.path.basename(img_path_single)[:-4] + '.npz')
            if os.path.exists(output_filename):
                cached_count += 1
            else:
                uncached_img_paths.append(img_path_single)

        print(f'Found {cached_count} cached results, processing {len(uncached_img_paths)} new images...')
        img_path_list = uncached_img_paths

        if len(img_path_list) == 0:
            print(f'All depth predictions already exist in {outdir_scene}')
            return outdir_scene
    else:
        print(f'Processing {len(img_path_list)} images with UniDepth...')

    print("Loading UniDepth model...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    model = model.to(device)
    model.eval()

    fovs = []
    for img_path_single in tqdm(img_path_list):
        rgb = np.array(Image.open(img_path_single))[..., :3]

        if rgb.shape[1] > rgb.shape[0]:
            final_w, final_h = long_dim, int(round(long_dim * rgb.shape[0] / rgb.shape[1]))
        else:
            final_w, final_h = int(round(long_dim * rgb.shape[1] / rgb.shape[0])), long_dim

        rgb = cv2.resize(rgb, (final_w, final_h), cv2.INTER_AREA)
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

        with torch.no_grad():
            predictions = model.infer(rgb_torch)

        fov = np.rad2deg(
            2 * np.arctan(
                predictions["depth"].shape[-1] / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())
            )
        )
        depth = predictions["depth"][0, 0].cpu().numpy()

        fovs.append(fov)

        output_filename = os.path.join(outdir_scene, os.path.basename(img_path_single)[:-4] + '.npz')
        np.savez(output_filename, depth=np.float32(depth), fov=fov)

    print(f'UniDepth predictions saved to {outdir_scene}')
    if len(fovs) > 0:
        print(f'Median FOV: {np.median(fovs):.2f} degrees')
    return outdir_scene


# ============================================================================
# Main Stage 1 Function
# ============================================================================

def run_stage1(video_path, scene_name, depth_anything_checkpoint, 
               depth_anything_output="depth_anything_output", 
               unidepth_output="unidepth_output", device='cuda'):
    """Run Stage 1: Monodepth Pre-computation."""
    print("=" * 80)
    print("Stage 1: Monodepth Pre-computation")
    print("=" * 80)

    # DepthAnything
    print("\nStage 1.1: Running DepthAnything...")
    depth_anything_dir = run_depth_anything(
        img_path=video_path,
        outdir=os.path.join(depth_anything_output, scene_name),
        checkpoint_path=depth_anything_checkpoint,
        encoder='vitl',
        device=device
    )

    # UniDepth
    print("\nStage 1.2: Running UniDepth...")
    unidepth_dir = run_unidepth(
        img_path=video_path,
        outdir=unidepth_output,
        scene_name=scene_name,
        device=device
    )

    print("\n" + "=" * 80)
    print("Stage 1 Complete: Monodepth pre-computation finished!")
    print("=" * 80)

    return depth_anything_dir, unidepth_dir