"""
MegaSaM Stage 2: Camera Tracking
Estimates camera poses using depth-guided visual odometry.
"""

import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# Depth Alignment
# ============================================================================

def align_depth_predictions(mono_disp_paths, metric_depth_paths):
    """Align DepthAnything disparity with UniDepth metric depth."""
    scales = []
    shifts = []
    mono_disp_list = []
    fovs = []
    sky_frame_count = 0

    print(f"\n{'='*60}")
    print(f"Starting depth alignment for {len(mono_disp_paths)} frame pairs")
    print(f"{'='*60}\n")

    for idx, (mono_disp_file, metric_depth_file) in enumerate(
        tqdm(zip(mono_disp_paths, metric_depth_paths),
             total=len(mono_disp_paths),
             desc="Processing frames",
             unit="frame")):

        da_disp = np.float32(np.load(mono_disp_file))
        uni_data = np.load(metric_depth_file)
        metric_depth = uni_data["depth"]
        fovs.append(uni_data["fov"])

        original_shape = da_disp.shape
        da_disp = cv2.resize(da_disp, (metric_depth.shape[1], metric_depth.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        mono_disp_list.append(da_disp)

        if idx == 0:
            print(f"\nFrame resolution: {original_shape} → {da_disp.shape}")

        gt_disp = 1.0 / (metric_depth + 1e-8)

        # Handle UniDepth bugs
        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2

        # Handle sky-dominated scenes
        sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
        if sky_ratio > 0.5:
            sky_frame_count += 1
            non_sky_mask = da_disp > 0.01
            gt_disp_ms = gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
            da_disp_ms = da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
        else:
            gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
            da_disp_ms = da_disp - np.median(da_disp) + 1e-8
            scale = np.median(gt_disp_ms / da_disp_ms)
            shift = np.median(gt_disp - scale * da_disp)

        scales.append(scale)
        shifts.append(shift)

    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"{'='*60}")
    print(f"Total frames processed: {len(mono_disp_paths)}")
    print(f"Sky-dominated frames: {sky_frame_count} ({100*sky_frame_count/len(mono_disp_paths):.1f}%)")
    print(f"Scale range: [{np.min(scales):.4f}, {np.max(scales):.4f}]")
    print(f"Shift range: [{np.min(shifts):.4f}, {np.max(shifts):.4f}]")
    print(f"FOV range: [{np.min(fovs):.2f}°, {np.max(fovs):.2f}°]")

    ss_product = np.array(scales) * np.array(shifts)
    med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))

    align_scale = scales[med_idx]
    align_shift = shifts[med_idx]

    print(f"\n{'='*60}")
    print(f"Alignment Parameters (from frame {med_idx}):")
    print(f"{'='*60}")
    print(f"Scale: {align_scale:.6f}")
    print(f"Shift: {align_shift:.6f}")

    normalize_scale = np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98) / 2.0
    print(f"Normalization scale (98th percentile / 2): {normalize_scale:.6f}")

    aligns = (align_scale, align_shift, normalize_scale)

    print(f"\nMedian FOV: {np.median(fovs):.2f}°")
    print(f"{'='*60}\n")

    return mono_disp_list, aligns, fovs


# ============================================================================
# Camera Intrinsics
# ============================================================================

def initialize_intrinsics_from_fov(fovs, image_width, image_height):
    """Initialize camera intrinsics from field-of-view estimate."""
    median_fov = np.median(fovs)
    ff = image_width / (2 * np.tan(np.radians(median_fov / 2.0)))

    K = np.eye(3)
    K[0, 0] = ff
    K[1, 1] = ff
    K[0, 2] = image_width / 2.0
    K[1, 2] = image_height / 2.0

    print(f"Initialized intrinsics from FOV {median_fov:.2f}°:")
    print(f"  Focal length: {ff:.2f}")
    print(f"  Principal point: ({K[0, 2]:.2f}, {K[1, 2]:.2f})")

    return K


# ============================================================================
# Image Stream for Tracking
# ============================================================================

def image_stream(image_list, mono_disp_list, K, aligns, use_depth=True):
    """Generator that yields preprocessed images and depth maps for tracking."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    for t, image_file in enumerate(image_list):
        image = cv2.imread(image_file)
        h0, w0, _ = image.shape

        mono_disp = mono_disp_list[t]

        depth = np.clip(
            1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
            1e-4, 1e4
        )
        depth[depth < 1e-2] = 0.0

        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None, None], (h1, w1), mode="nearest-exact").squeeze()
        depth = depth[:h1 - h1 % 8, :w1 - w1 % 8]

        mask = torch.ones_like(depth)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        if use_depth:
            yield t, image[None], depth, intrinsics, mask
        else:
            yield t, image[None], intrinsics, mask


# ============================================================================
# Camera Tracking (DROID-SLAM Interface)
# ============================================================================

def run_camera_tracking(image_list, mono_disp_list, K, aligns, scene_name, 
                       checkpoint_path, output_dir="reconstructions", use_cache=True):
    """Run camera tracking on a video sequence."""
    from lietorch import SE3
    
    cache_files = {
        'images': f"{output_dir}/{scene_name}/images.npy",
        'disps': f"{output_dir}/{scene_name}/disps.npy",
        'poses': f"{output_dir}/{scene_name}/poses.npy",
        'intrinsics': f"{output_dir}/{scene_name}/intrinsics.npy",
        'motion_prob': f"{output_dir}/{scene_name}/motion_prob.npy",
    }
    compact_output = f"outputs/{scene_name}_droid.npz"

    all_cached = all(os.path.exists(f) for f in cache_files.values())
    compact_exists = os.path.exists(compact_output)

    if use_cache and all_cached and compact_exists:
        print(f"\n✓ Found cached camera tracking results in {output_dir}/{scene_name}/")
        print(f"✓ Found compact output at {compact_output}")
        print("Loading cached results...")

        poses = np.load(cache_files['poses'])
        disps = np.load(cache_files['disps'])
        depths = 1.0 / (disps + 1e-6)
        intrinsics = np.load(cache_files['intrinsics'])
        motion_prob = np.load(cache_files['motion_prob'])

        print(f"Loaded {len(poses)} camera poses from cache")
        return poses, depths, intrinsics, motion_prob

    if use_cache:
        print("\nNo complete cache found. Running camera tracking...")
    else:
        print("\nCache disabled. Running camera tracking...")

    print("\nNote: This function shows the camera tracking interface.")
    print("The full DROID-SLAM implementation requires compiled CUDA extensions.")
    print("Please refer to the original codebase for complete implementation.\n")

    # Import DROID (requires compiled extensions)
    try:
        from droid import Droid
    except ImportError:
        print("Error: DROID-SLAM not properly installed.")
        print("Please run setup.py to install DROID-SLAM and compile extensions.")
        return None, None, None, None

    class Args:
        def __init__(self):
            self.weights = checkpoint_path
            self.buffer = 1024
            self.image_size = [240, 320]
            self.disable_vis = True
            self.beta = 0.3
            self.filter_thresh = 2.0
            self.warmup = 8
            self.keyframe_thresh = 2.0
            self.frontend_thresh = 12.0
            self.frontend_window = 25
            self.frontend_radius = 2
            self.frontend_nms = 1
            self.backend_thresh = 16.0
            self.backend_radius = 2
            self.backend_nms = 3
            self.stereo = False
            self.depth = True
            self.upsample = False

    args = Args()

    rgb_list = []
    sensor_depth_list = []
    droid = None

    print("Tracking frames...")
    for t, image, depth, intrinsics, mask in tqdm(image_stream(image_list, mono_disp_list, K, aligns, use_depth=True)):
        rgb_list.append(image[0])
        sensor_depth_list.append(depth)

        if t == 0:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

    droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

    traj_est, depth_est, motion_prob = droid.terminate(
        image_stream(image_list, mono_disp_list, K, aligns, use_depth=True),
        _opt_intr=True,
        full_ba=True,
        scene_name=scene_name
    )

    save_camera_tracking_results(droid, traj_est, rgb_list, sensor_depth_list,
                                  motion_prob, scene_name, output_dir)

    intrinsics = droid.video.intrinsics[:len(traj_est)].cpu().numpy()

    return traj_est, depth_est, intrinsics, motion_prob


def save_camera_tracking_results(droid, full_traj, rgb_list, sensor_depth_list, 
                                 motion_prob, scene_name, output_dir="reconstructions"):
    """Save camera tracking results to disk."""
    from lietorch import SE3

    t = full_traj.shape[0]
    images = np.array(rgb_list[:t])
    disps = 1.0 / (np.array(sensor_depth_list[:t]) + 1e-6)
    poses = full_traj
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path(f"{output_dir}/{scene_name}").mkdir(parents=True, exist_ok=True)
    np.save(f"{output_dir}/{scene_name}/images.npy", images)
    np.save(f"{output_dir}/{scene_name}/disps.npy", disps)
    np.save(f"{output_dir}/{scene_name}/poses.npy", poses)
    np.save(f"{output_dir}/{scene_name}/intrinsics.npy", intrinsics * 8.0)
    np.save(f"{output_dir}/{scene_name}/motion_prob.npy", motion_prob)

    intrinsics = intrinsics[0] * 8.0
    poses_th = torch.as_tensor(poses, device="cpu")
    cam_c2w = SE3(poses_th).inv().matrix().numpy()

    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    Path("outputs").mkdir(parents=True, exist_ok=True)
    max_frames = min(1000, images.shape[0])
    np.savez(
        f"outputs/{scene_name}_droid.npz",
        images=np.uint8(images[:max_frames, ::-1, ...].transpose(0, 2, 3, 1)),
        depths=np.float32(1.0 / disps[:max_frames, ...]),
        intrinsic=K,
        cam_c2w=cam_c2w[:max_frames],
    )

    print(f"Camera tracking results saved to {output_dir}/{scene_name}/")
    print(f"Compact output saved to outputs/{scene_name}_droid.npz")


# ============================================================================
# Main Stage 2 Function
# ============================================================================

def run_stage2(video_path, scene_name, depth_anything_output, unidepth_output, 
               checkpoint_path, reconstruction_dir="reconstructions"):
    """Run Stage 2: Camera Tracking."""
    print("=" * 80)
    print("Stage 2: Camera Tracking")
    print("=" * 80)

    image_list = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    image_list += sorted(glob.glob(os.path.join(video_path, "*.png")))

    mono_disp_paths = sorted(glob.glob(os.path.join(depth_anything_output, scene_name, "*.npy")))
    metric_depth_paths = sorted(glob.glob(os.path.join(unidepth_output, scene_name, "*.npz")))

    print("\nAligning depth predictions...")
    mono_disp_list, aligns, fovs = align_depth_predictions(mono_disp_paths, metric_depth_paths)

    img_0 = cv2.imread(image_list[0])
    K = initialize_intrinsics_from_fov(fovs, img_0.shape[1], img_0.shape[0])

    print(f"\nRunning camera tracking with checkpoint: {checkpoint_path}")

    poses, depths, intrinsics, motion_prob = run_camera_tracking(
        image_list=image_list,
        mono_disp_list=mono_disp_list,
        K=K,
        aligns=aligns,
        scene_name=scene_name,
        checkpoint_path=checkpoint_path,
        output_dir=reconstruction_dir
    )

    print("\n" + "=" * 80)
    print("Stage 2 Complete: Camera tracking finished!")
    print("=" * 80)

    return poses, depths, intrinsics, motion_prob