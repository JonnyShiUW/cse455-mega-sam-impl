"""
MegaSaM Stage 3: Consistent Video Depth (CVD) Optimization
Refines depth maps using multi-frame consistency.
"""

import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# Geometry Utilities
# ============================================================================

class NormalGenerator(torch.nn.Module):
    """Estimates surface normals from depth maps."""
    def __init__(self, height, width, smoothing_kernel_size=5, smoothing_kernel_std=2.0):
        super().__init__()
        self.height = height
        self.width = width
        self.kernel_size = smoothing_kernel_size
        self.std = smoothing_kernel_std
        self.backproject = BackprojectDepth(height, width)

    def forward(self, depth_b1hw, invK_b44):
        depth_smooth_b1hw = kornia.filters.gaussian_blur2d(
            depth_b1hw, (self.kernel_size, self.kernel_size), (self.std, self.std)
        )

        cam_points_b4N = self.backproject(depth_smooth_b1hw, invK_b44)
        cam_points_b3hw = cam_points_b4N[:, :3].view(-1, 3, self.height, self.width)

        gradients_b32hw = kornia.filters.spatial_gradient(cam_points_b3hw)

        return F.normalize(
            torch.cross(gradients_b32hw[:, :, 0], gradients_b32hw[:, :, 1], dim=1),
            dim=1
        )


class BackprojectDepth(torch.nn.Module):
    """Backproject 2D pixels to 3D points using depth."""
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

        xx, yy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")
        pix_coords_2hw = torch.stack((xx, yy), axis=0) + 0.5
        pix_coords_13N = torch.cat([pix_coords_2hw, torch.ones(1, height, width)], dim=0).flatten(1).unsqueeze(0)

        self.register_buffer("pix_coords_13N", pix_coords_13N)

    def forward(self, depth_b1hw, invK_b44):
        cam_points_b3N = torch.matmul(invK_b44[:, :3, :3], self.pix_coords_13N.float().cuda())
        cam_points_b3N = depth_b1hw.flatten(start_dim=2) * cam_points_b3N
        cam_points_b4N = torch.cat([cam_points_b3N, torch.ones_like(cam_points_b3N[:, :1, :])], dim=1)
        return cam_points_b4N


def sobel_fg_alpha(disp, mode="sobel", beta=10.0):
    """Compute edge-aware weights using Sobel gradients."""
    sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
    sobel_mag = torch.sqrt(sobel_grad[:, :, 0, ...] ** 2 + sobel_grad[:, :, 1, ...] ** 2)
    alpha = torch.exp(-1.0 * beta * sobel_mag).detach()
    return alpha


def warp_flow(img, flow):
    """Warp image using optical flow."""
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return res


def resize_flow(flow, img_h, img_w):
    """Resize optical flow and scale flow vectors."""
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)
    return flow


# ============================================================================
# Loss Functions
# ============================================================================

def si_loss(gt, pred):
    """Scale-invariant loss for depth."""
    log_gt = torch.log(torch.clamp(gt, 1e-3, 1e3)).view(gt.shape[0], -1)
    log_pred = torch.log(torch.clamp(pred, 1e-3, 1e3)).view(pred.shape[0], -1)
    log_diff = log_gt - log_pred
    num_pixels = gt.shape[-2] * gt.shape[-1]
    data_loss = torch.sum(log_diff**2, dim=-1) / num_pixels - torch.sum(log_diff, dim=-1)**2 / (num_pixels**2)
    return torch.mean(data_loss)


def gradient_loss(gt, pred, u):
    """Gradient consistency loss."""
    del u
    diff = pred - gt

    v_gradient = torch.abs(diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1])
    h_gradient = torch.abs(diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:])

    pred_grad = torch.abs(pred[..., 0:-2, 1:-1] - pred[..., 2:, 1:-1]) + torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
    gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - gt[..., 2:, 1:-1]) + torch.abs(gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:])

    grad_diff = torch.abs(pred_grad - gt_grad)
    nearby_mask = (torch.exp(gt[..., 1:-1, 1:-1]) > 1.0).float().detach()
    weight = 1.0 - torch.exp(-(grad_diff * 5.0)).detach()
    weight *= nearby_mask

    g_loss = torch.mean(h_gradient * weight) + torch.mean(v_gradient * weight)
    return g_loss


def consistency_loss(
    cam_c2w, K, K_inv, disp_data, init_disp, uncertainty,
    flows, flow_masks, ii, jj, compute_normals, fg_alpha,
    w_ratio=1.0, w_flow=0.2, w_si=1.0, w_grad=2.0, w_normal=4.0
):
    """Multi-view consistency loss for video depth optimization."""
    ALPHA_MOTION = 0.25
    _, H, W = disp_data.shape

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)
    grid = torch.cat((xx, yy), 1).float().cuda().permute(0, 2, 3, 1)

    loss_flow = 0.0
    loss_d_ratio = 0.0

    flows_step = flows.permute(0, 2, 3, 1)
    flow_masks_step = flow_masks.permute(0, 2, 3, 1).squeeze(-1)

    cam_1to2 = torch.bmm(
        torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
        torch.index_select(cam_c2w, dim=0, index=ii)
    )

    pixel_locations = grid + flows_step
    resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
    normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

    disp_sampled = torch.nn.functional.grid_sample(
        torch.index_select(disp_data, dim=0, index=jj)[:, None, ...],
        normalized_pixel_locations,
        align_corners=True
    )

    uu = torch.index_select(uncertainty, dim=0, index=ii).squeeze(1)

    grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(-1)

    ref_depth = 1.0 / torch.clamp(torch.index_select(disp_data, dim=0, index=ii), 1e-3, 1e3)

    pts_3d_ref = ref_depth[..., None, None] * (K_inv[None, None, None] @ grid_h)
    rot = cam_1to2[:, None, None, :3, :3]
    trans = cam_1to2[:, None, None, :3, 3:4]
    pts_3d_tgt = (rot @ pts_3d_ref) + trans
    depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
    disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

    pts_2D_tgt = K[None, None, None] @ pts_3d_tgt
    flow_masks_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1)
    pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / torch.clamp(pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3)

    disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
    disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

    ratio = torch.maximum(disp_sampled.squeeze() / disp_tgt.squeeze(), disp_tgt.squeeze() / disp_sampled.squeeze())
    ratio_error = torch.abs(ratio - 1.0)
    loss_d_ratio += torch.sum((ratio_error * uu + ALPHA_MOTION * torch.log(1.0 / uu)) * flow_masks_step_) / (torch.sum(flow_masks_step_) + 1e-8)

    flow_error = torch.abs(pts_2D_tgt - pixel_locations)
    loss_flow += torch.sum((flow_error * uu[..., None] + ALPHA_MOTION * torch.log(1.0 / uu[..., None])) * flow_masks_step_[..., None]) / (torch.sum(flow_masks_step_) * 2.0 + 1e-8)

    loss_prior = si_loss(init_disp, disp_data)

    KK = torch.inverse(K_inv)
    disp_data_ds = disp_data[:, None, ...]
    init_disp_ds = init_disp[:, None, ...]
    K_rescale = KK.clone()
    K_inv_rescale = torch.inverse(K_rescale)

    pred_normal = compute_normals[0](1.0 / torch.clamp(disp_data_ds, 1e-3, 1e3), K_inv_rescale[None])
    init_normal = compute_normals[0](1.0 / torch.clamp(init_disp_ds, 1e-3, 1e3), K_inv_rescale[None])
    loss_normal = torch.mean(fg_alpha * (1.0 - torch.sum(pred_normal * init_normal, dim=1)))

    loss_grad = 0.0
    for scale in range(4):
        interval = 2**scale
        disp_data_scale = torch.nn.functional.interpolate(
            disp_data[:, None, ...],
            scale_factor=(1.0 / interval, 1.0 / interval),
            mode="nearest-exact"
        )
        init_disp_scale = torch.nn.functional.interpolate(
            init_disp[:, None, ...],
            scale_factor=(1.0 / interval, 1.0 / interval),
            mode="nearest-exact"
        )
        uncertainty_rs = torch.nn.functional.interpolate(
            uncertainty,
            scale_factor=(1.0 / interval, 1.0 / interval),
            mode="nearest-exact"
        )
        loss_grad += gradient_loss(torch.log(disp_data_scale), torch.log(init_disp_scale), uncertainty_rs)

    return (
        w_ratio * loss_d_ratio +
        w_si * loss_prior +
        w_flow * loss_flow +
        w_normal * loss_normal +
        loss_grad * w_grad
    )


# ============================================================================
# Optical Flow Preprocessing
# ============================================================================

def preprocess_optical_flow(image_list, scene_name, raft_model_path, cache_dir="cache_flow"):
    """Precompute optical flow for all frame pairs."""
    print("Preprocessing optical flow...")

    try:
        import sys
        sys.path.append('RAFT/core')
        from raft import RAFT
        from utils.utils import InputPadder

        class Args:
            def __init__(self):
                self.small = False
                self.num_heads = 1
                self.position_only = False
                self.position_and_content = False
                self.mixed_precision = True

            def __contains__(self, key):
                return hasattr(self, key)

        args = Args()
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(raft_model_path))
        flow_model = model.module
        flow_model.cuda()
        flow_model.eval()

        print(f"RAFT model loaded from {raft_model_path}")

        img_data = []
        for image_file in tqdm(image_list, desc="Loading images"):
            image = cv2.imread(image_file)[..., ::-1]
            h0, w0, _ = image.shape

            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            image = cv2.resize(image, (w1, h1))
            image = image[:h1 - h1 % 8, :w1 - w1 % 8].transpose(2, 0, 1)
            img_data.append(image)

        img_data = np.array(img_data)

        flows_arr_low_bwd = {}
        flows_arr_low_fwd = {}
        ii = []
        jj = []
        flows_arr_up = []
        masks_arr_up = []

        for step in [1, 2, 4, 8, 15]:
            print(f"Computing flow with step size {step}...")

            for i in tqdm(range(max(0, -step), img_data.shape[0] - max(0, step))):
                image1 = torch.as_tensor(np.ascontiguousarray(img_data[i:i+1])).float().cuda()
                image2 = torch.as_tensor(np.ascontiguousarray(img_data[i+step:i+step+1])).float().cuda()

                ii.append(i)
                jj.append(i + step)

                with torch.no_grad():
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    if np.abs(step) > 1:
                        flow_init = np.stack([flows_arr_low_fwd[i], flows_arr_low_bwd[i+step]], axis=0)
                        flow_init = torch.as_tensor(np.ascontiguousarray(flow_init)).float().cuda().permute(0, 3, 1, 2)
                    else:
                        flow_init = None

                    flow_low, flow_up, _ = flow_model(
                        torch.cat([image1, image2], dim=0),
                        torch.cat([image2, image1], dim=0),
                        iters=22,
                        test_mode=True,
                        flow_init=flow_init
                    )

                    flow_low_fwd = flow_low[0].cpu().numpy().transpose(1, 2, 0)
                    flow_low_bwd = flow_low[1].cpu().numpy().transpose(1, 2, 0)

                    flow_up_fwd = resize_flow(
                        flow_up[0].cpu().numpy().transpose(1, 2, 0),
                        flow_up.shape[-2] // 2,
                        flow_up.shape[-1] // 2
                    )
                    flow_up_bwd = resize_flow(
                        flow_up[1].cpu().numpy().transpose(1, 2, 0),
                        flow_up.shape[-2] // 2,
                        flow_up.shape[-1] // 2
                    )

                    bwd2fwd_flow = warp_flow(flow_up_bwd, flow_up_fwd)
                    fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
                    fwd_mask_up = fwd_lr_error < 1.0

                    flows_arr_low_bwd[i + step] = flow_low_bwd
                    flows_arr_low_fwd[i] = flow_low_fwd
                    flows_arr_up.append(flow_up_fwd)
                    masks_arr_up.append(fwd_mask_up)

        iijj = np.stack((ii, jj), axis=0)
        flows_high = np.array(flows_arr_up).transpose(0, 3, 1, 2)
        flow_masks_high = np.array(masks_arr_up)[:, None, ...]

        Path(f"{cache_dir}/{scene_name}").mkdir(parents=True, exist_ok=True)
        np.save(f"{cache_dir}/{scene_name}/flows.npy", np.float16(flows_high))
        np.save(f"{cache_dir}/{scene_name}/flows_masks.npy", flow_masks_high)
        np.save(f"{cache_dir}/{scene_name}/ii-jj.npy", iijj)

        print(f"Optical flow saved to {cache_dir}/{scene_name}/")
        return f"{cache_dir}/{scene_name}"

    except ImportError as e:
        print(f"Error: {e}")
        print("RAFT implementation not available. Please check RAFT installation.")
        return None


# ============================================================================
# Video Depth Optimization
# ============================================================================

def run_cvd_optimization(scene_name, cache_dir="cache_flow", reconstruction_dir="reconstructions",
                        output_dir="outputs_cvd", w_grad=2.0, w_normal=5.0, resize_factor=0.5):
    """Run consistent video depth optimization."""
    from lietorch import SE3

    print(f"Running CVD optimization for {scene_name}...")

    rootdir = os.path.join(reconstruction_dir)
    img_data = np.load(os.path.join(rootdir, scene_name, "images.npy"))[:, ::-1, ...]
    disp_data = np.load(os.path.join(rootdir, scene_name, "disps.npy")) + 1e-6
    intrinsics = np.load(os.path.join(rootdir, scene_name, "intrinsics.npy"))
    poses = np.load(os.path.join(rootdir, scene_name, "poses.npy"))
    mot_prob = np.load(os.path.join(rootdir, scene_name, "motion_prob.npy"))

    flows = np.load(f"{cache_dir}/{scene_name}/flows.npy", allow_pickle=True)
    flow_masks = np.load(f"{cache_dir}/{scene_name}/flows_masks.npy", allow_pickle=True)
    flow_masks = np.float32(flow_masks)
    iijj = np.load(f"{cache_dir}/{scene_name}/ii-jj.npy", allow_pickle=True)

    print(f"Loaded {len(img_data)} frames")

    intrinsics = intrinsics[0]
    poses_th = torch.as_tensor(poses, device="cpu").float().cuda()

    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    img_data_pt = torch.from_numpy(np.ascontiguousarray(img_data)).float().cuda() / 255.0
    flows = torch.from_numpy(np.ascontiguousarray(flows)).float().cuda()
    flow_masks = torch.from_numpy(np.ascontiguousarray(flow_masks)).float().cuda()
    iijj = torch.from_numpy(np.ascontiguousarray(iijj)).float().cuda()
    ii = iijj[0, ...].long()
    jj = iijj[1, ...].long()
    K = torch.from_numpy(K).float().cuda()

    init_disp = torch.from_numpy(disp_data).float().cuda()
    disp_data = torch.from_numpy(disp_data).float().cuda()

    init_disp = torch.nn.functional.interpolate(
        init_disp.unsqueeze(1),
        scale_factor=(resize_factor, resize_factor),
        mode="bilinear"
    ).squeeze(1)
    disp_data = torch.nn.functional.interpolate(
        disp_data.unsqueeze(1),
        scale_factor=(resize_factor, resize_factor),
        mode="bilinear"
    ).squeeze(1)

    fg_alpha = sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
    fg_alpha = fg_alpha.squeeze(1).float() + 0.2

    cvd_prob = torch.nn.functional.interpolate(
        torch.from_numpy(mot_prob).unsqueeze(1).cuda(),
        scale_factor=(4, 4),
        mode="bilinear"
    )
    cvd_prob[cvd_prob > 0.5] = 0.5
    cvd_prob = torch.clamp(cvd_prob, 1e-3, 1.0)

    K_o = K.clone()
    K[0:2, ...] *= resize_factor
    K_inv = torch.linalg.inv(K)

    disp_data.requires_grad = False
    poses_th.requires_grad = False
    uncertainty = cvd_prob

    log_scale_ = torch.log(torch.ones(init_disp.shape[0]).to(disp_data.device))
    shift_ = torch.zeros(init_disp.shape[0]).to(disp_data.device)
    log_scale_.requires_grad = True
    shift_.requires_grad = True
    uncertainty.requires_grad = True

    optim = torch.optim.Adam([
        {"params": log_scale_, "lr": 1e-2},
        {"params": shift_, "lr": 1e-2},
        {"params": uncertainty, "lr": 1e-2},
    ])

    compute_normals = [NormalGenerator(disp_data.shape[-2], disp_data.shape[-1])]
    init_disp = torch.clamp(init_disp, 1e-3, 1e3)

    print("\nStage 1: Optimizing scale and shift (100 iterations)...")
    for i in tqdm(range(100)):
        optim.zero_grad()
        cam_c2w = SE3(poses_th).inv().matrix()
        scale_ = torch.exp(log_scale_)

        loss = consistency_loss(
            cam_c2w, K, K_inv,
            torch.clamp(disp_data * scale_[..., None, None] + shift_[..., None, None], 1e-3, 1e3),
            init_disp,
            torch.clamp(uncertainty, 1e-4, 1e3),
            flows, flow_masks, ii, jj,
            compute_normals, fg_alpha
        )

        loss.backward()
        uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
        log_scale_.grad = torch.nan_to_num(log_scale_.grad, nan=0.0)
        shift_.grad = torch.nan_to_num(shift_.grad, nan=0.0)

        optim.step()

        if i % 20 == 0:
            print(f"  Step {i}, Loss: {loss.item():.6f}")

    disp_data = (disp_data * torch.exp(log_scale_)[..., None, None].detach() + shift_[..., None, None].detach())
    init_disp = (init_disp * torch.exp(log_scale_)[..., None, None].detach() + shift_[..., None, None].detach())
    init_disp = torch.clamp(init_disp, 1e-3, 1e3)

    disp_data.requires_grad = True
    uncertainty.requires_grad = True
    poses_th.requires_grad = False

    optim = torch.optim.Adam([
        {"params": disp_data, "lr": 5e-3},
        {"params": uncertainty, "lr": 5e-3},
    ])

    print("\nStage 2: Optimizing depth and uncertainty (400 iterations)...")
    for i in tqdm(range(400)):
        optim.zero_grad()
        cam_c2w = SE3(poses_th).inv().matrix()

        loss = consistency_loss(
            cam_c2w, K, K_inv,
            torch.clamp(disp_data, 1e-3, 1e3),
            init_disp,
            torch.clamp(uncertainty, 1e-4, 1e3),
            flows, flow_masks, ii, jj,
            compute_normals, fg_alpha,
            w_ratio=1.0, w_flow=0.2, w_si=1.0, w_grad=w_grad, w_normal=w_normal
        )

        loss.backward()
        disp_data.grad = torch.nan_to_num(disp_data.grad, nan=0.0)
        uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)

        optim.step()

        if i % 50 == 0:
            print(f"  Step {i}, Loss: {loss.item():.6f}")

    disp_data_opt = torch.nn.functional.interpolate(
        disp_data.unsqueeze(1),
        scale_factor=(2, 2),
        mode="bilinear"
    ).squeeze(1).detach().cpu().numpy()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        f"{output_dir}/{scene_name}_sgd_cvd_hr.npz",
        images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0),
        depths=np.clip(np.float16(1.0 / disp_data_opt), 1e-3, 1e2),
        intrinsic=K_o.detach().cpu().numpy(),
        cam_c2w=SE3(poses_th).inv().matrix().detach().cpu().numpy(),
    )

    print(f"\nOptimized depths saved to {output_dir}/{scene_name}_sgd_cvd_hr.npz")
    print(f"Final loss: {loss.item():.6f}")


# ============================================================================
# Main Stage 3 Function
# ============================================================================

def run_stage3(video_path, scene_name, raft_model_path, 
               cache_dir="cache_flow", reconstruction_dir="reconstructions",
               output_dir="outputs_cvd", w_grad=2.0, w_normal=5.0):
    """Run Stage 3: Consistent Video Depth Optimization."""
    print("=" * 80)
    print("Stage 3: Consistent Video Depth Optimization")
    print("=" * 80)

    image_list = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    image_list += sorted(glob.glob(os.path.join(video_path, "*.png")))

    print("\nStep 1: Preprocessing optical flow...")
    flow_cache_dir = preprocess_optical_flow(
        image_list=image_list,
        scene_name=scene_name,
        raft_model_path=raft_model_path,
        cache_dir=cache_dir
    )

    print("\nStep 2: Running video depth optimization...")
    run_cvd_optimization(
        scene_name=scene_name,
        cache_dir=cache_dir,
        reconstruction_dir=reconstruction_dir,
        output_dir=output_dir,
        w_grad=w_grad,
        w_normal=w_normal
    )

    print("\n" + "=" * 80)
    print("Stage 3 Complete: Video depth optimization finished!")
    print("=" * 80)
    print("\nMega-SAM pipeline complete!")
    print(f"Final output saved to {output_dir}/{scene_name}_sgd_cvd_hr.npz")