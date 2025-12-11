"""
MegaSaM: Complete End-to-End Pipeline
Main script to run the complete pipeline for camera tracking and depth estimation.

Paper: MegaSam: Accurate, Fast and Robust Casual Structure and Motion from Casual Dynamic Videos
Authors: Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, 
         Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, Noah Snavely
"""

import os

from stage1 import run_stage1
from stage2 import run_stage2
from stage3 import run_stage3


# ============================================================================
# CONFIGURATION - Edit these values for your setup
# ============================================================================

# Input settings
FRAMES_DIR = "test_video"      # Directory containing input frames (JPEG images)
SCENE_NAME = "marching"        # Name for this scene

# Model checkpoints
DEPTH_ANYTHING_CHECKPOINT = "mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth"
TRACKING_CHECKPOINT = "mega-sam/checkpoints/megasam_final.pth"
RAFT_MODEL = "mega-sam/cvd_opt/raft-things.pth"

# Output directories
DEPTH_ANYTHING_OUTPUT = "depth_anything_output"
UNIDEPTH_OUTPUT = "unidepth_output"
RECONSTRUCTION_DIR = "reconstructions"
CACHE_DIR = "cache_flow"
OUTPUT_DIR = "outputs_cvd"

# Processing options
DEVICE = "cuda"  # "cuda" or "cpu"

# Pipeline control (set to True to skip stages and use cached results)
SKIP_STAGE1 = False  # Skip depth pre-computation
SKIP_STAGE2 = False  # Skip camera tracking
SKIP_STAGE3 = False  # Skip depth optimization

# Optimization parameters
W_GRAD = 2.0    # Weight for gradient loss
W_NORMAL = 5.0  # Weight for normal loss


# ============================================================================
# Main Pipeline
# ============================================================================

def run_megasam_pipeline():
    """Run the complete MegaSaM pipeline."""
    print("\n" + "="*80)
    print("MegaSaM: Complete End-to-End Pipeline")
    print("="*80)
    print(f"\nScene: {SCENE_NAME}")
    print(f"Frames directory: {FRAMES_DIR}")
    print(f"Device: {DEVICE}")
    print("="*80 + "\n")

    # Verify frames directory exists
    if not os.path.exists(FRAMES_DIR):
        print(f"Error: Frames directory not found: {FRAMES_DIR}")
        return

    frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(frame_files)} frames in {FRAMES_DIR}\n")

    # Stage 1: Monodepth Pre-computation
    if not SKIP_STAGE1:
        print("\n" + "="*80)
        print("Running Stage 1: Monodepth Pre-computation")
        print("="*80 + "\n")
        depth_anything_dir, unidepth_dir = run_stage1(
            video_path=FRAMES_DIR,
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
            video_path=FRAMES_DIR,
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
            video_path=FRAMES_DIR,
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
    print(f"\nInput frames: {FRAMES_DIR}/")
    print(f"Final output: {OUTPUT_DIR}/{SCENE_NAME}_sgd_cvd_hr.npz")
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