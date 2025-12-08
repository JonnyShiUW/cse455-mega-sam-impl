"""
MegaSaM: Complete End-to-End Pipeline
Main script to run the complete pipeline for camera tracking and depth estimation.

Paper: MegaSam: Accurate, Fast and Robust Casual Structure and Motion from Casual Dynamic Videos
Authors: Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, 
         Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, Noah Snavely
"""

import os
import cv2
from pathlib import Path

from stage1 import run_stage1
from stage2 import run_stage2
from stage3 import run_stage3


# ============================================================================
# CONFIGURATION - Edit these values for your setup
# ============================================================================

# Input video settings
VIDEO_FILE = "test_video.mp4"  # Path to your input video
SCENE_NAME = "marching"        # Name for this scene
VIDEO_PATH = "frames"          # Directory where frames will be stored

# Model checkpoints
DEPTH_ANYTHING_CHECKPOINT = "depth_anything_vitl14.pth"
TRACKING_CHECKPOINT = "megasam_final.pth"
RAFT_MODEL = "RAFT/models/raft-things.pth"

# Output directories
DEPTH_ANYTHING_OUTPUT = "depth_anything_output"
UNIDEPTH_OUTPUT = "unidepth_output"
RECONSTRUCTION_DIR = "reconstructions"
CACHE_DIR = "cache_flow"
OUTPUT_DIR = "outputs_cvd"

# Processing options
DEVICE = "cuda"  # "cuda" or "cpu"
RESIZE_FRAMES = True  # Resize to 640x360 during extraction
EXTRACT_FRAMES = True  # Set to False if frames already extracted

# Pipeline control (set to True to skip stages and use cached results)
SKIP_STAGE1 = False  # Skip depth pre-computation
SKIP_STAGE2 = False  # Skip camera tracking
SKIP_STAGE3 = False  # Skip depth optimization

# Optimization parameters
W_GRAD = 2.0    # Weight for gradient loss
W_NORMAL = 5.0  # Weight for normal loss


# ============================================================================
# Video Preprocessing
# ============================================================================

def extract_frames(video_path, output_folder, target_size=None):
    """Extract frames from a video file."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Original resolution: {orig_width}x{orig_height}")
    if target_size:
        print(f"Target resolution: {target_size[0]}x{target_size[1]}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        frame_filename = os.path.join(output_folder, f'frame_{saved_count:06d}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    print(f"\nExtraction complete! Saved {saved_count} frames to '{output_folder}'")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_megasam_pipeline():
    """Run the complete MegaSaM pipeline."""
    print("\n" + "="*80)
    print("MegaSaM: Complete End-to-End Pipeline")
    print("="*80)
    print(f"\nScene: {SCENE_NAME}")
    print(f"Video: {VIDEO_FILE}")
    print(f"Device: {DEVICE}")
    print("="*80 + "\n")

    # Extract frames if needed
    if EXTRACT_FRAMES and os.path.exists(VIDEO_FILE):
        print("Extracting frames from video...")
        extract_frames(
            video_path=VIDEO_FILE,
            output_folder=VIDEO_PATH,
            target_size=(640, 360) if RESIZE_FRAMES else None
        )
        print(f"Frames extracted to {VIDEO_PATH}\n")
    elif EXTRACT_FRAMES:
        print(f"Warning: Video file not found: {VIDEO_FILE}")
        print(f"Assuming frames already exist in {VIDEO_PATH}\n")

    # Stage 1: Monodepth Pre-computation
    if not SKIP_STAGE1:
        print("\n" + "="*80)
        print("Running Stage 1: Monodepth Pre-computation")
        print("="*80 + "\n")
        depth_anything_dir, unidepth_dir = run_stage1(
            video_path=VIDEO_PATH,
            scene_name=SCENE_NAME,
            depth_anything_checkpoint=DEPTH_ANYTHING_CHECKPOINT,
            depth_anything_output=DEPTH_ANYTHING_OUTPUT,
            unidepth_output=UNIDEPTH_OUTPUT,
            device=DEVICE
        )
    else:
        print("\nSkipping Stage 1 (using cached depth predictions)")

    # Stage 2: Camera Tracking
    if not SKIP_STAGE2:
        print("\n" + "="*80)
        print("Running Stage 2: Camera Tracking")
        print("="*80 + "\n")
        poses, depths, intrinsics, motion_prob = run_stage2(
            video_path=VIDEO_PATH,
            scene_name=SCENE_NAME,
            depth_anything_output=DEPTH_ANYTHING_OUTPUT,
            unidepth_output=UNIDEPTH_OUTPUT,
            checkpoint_path=TRACKING_CHECKPOINT,
            reconstruction_dir=RECONSTRUCTION_DIR
        )
    else:
        print("\nSkipping Stage 2 (using cached camera tracking)")

    # Stage 3: Consistent Video Depth Optimization
    if not SKIP_STAGE3:
        print("\n" + "="*80)
        print("Running Stage 3: Consistent Video Depth Optimization")
        print("="*80 + "\n")
        run_stage3(
            video_path=VIDEO_PATH,
            scene_name=SCENE_NAME,
            raft_model_path=RAFT_MODEL,
            cache_dir=CACHE_DIR,
            reconstruction_dir=RECONSTRUCTION_DIR,
            output_dir=OUTPUT_DIR,
            w_grad=W_GRAD,
            w_normal=W_NORMAL
        )
    else:
        print("\nSkipping Stage 3 (depth optimization)")

    print("\n" + "="*80)
    print("MegaSaM Pipeline Complete!")
    print("="*80)
    print(f"\nFinal output: {OUTPUT_DIR}/{SCENE_NAME}_sgd_cvd_hr.npz")
    print(f"Camera tracking: {RECONSTRUCTION_DIR}/{SCENE_NAME}/")
    print(f"Intermediate outputs: outputs/{SCENE_NAME}_droid.npz")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        run_megasam_pipeline()
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()