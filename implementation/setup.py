"""
MegaSaM Setup Script
Handles installation of dependencies and downloading of required models/repositories.
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_megasam(base_dir):
    """Clone mega-sam repository for utilities."""
    megasam_dir = os.path.join(base_dir, "mega-sam")
    
    # Add to Python path
    cvd_opt_path = os.path.join(megasam_dir, "cvd_opt")
    if cvd_opt_path not in sys.path:
        sys.path.append(cvd_opt_path)
    
    return megasam_dir

def setup_droidslam(base_dir):
    """Clone and setup DROID-SLAM repository."""
    droid_dir = os.path.join(base_dir, "DROID-SLAM")

    
    # Install DROID dependencies
    droid_slam_path = os.path.join(droid_dir, "droid_slam")
    if droid_slam_path not in sys.path:
        sys.path.append(droid_slam_path)
    
    return droid_dir


def setup_directories(base_dir):
    """Create necessary directories."""
    dirs = [
        "frames",
        "depth_anything_output",
        "unidepth_output",
        "reconstructions",
        "outputs",
        "outputs_cvd",
        "cache_flow",
    ]
    
    for d in dirs:
        path = os.path.join(base_dir, d)
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")


def setup_megasam_pipeline(base_dir=None):
    """Main setup function."""
    if base_dir is None:
        base_dir = os.getcwd()
    
    print("="*80)
    print("MegaSaM Pipeline Setup")
    print("="*80)
    
    print(f"\nBase directory: {base_dir}")
    
    # Setup repositories
    setup_megasam(base_dir)
    setup_droidslam(base_dir)
    

    # Create directories
    setup_directories(base_dir)
    
    print("\n" + "="*80)
    print("Setup Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Place your video in the base directory")
    print("2. Run: python main.py")

if __name__ == "__main__":
    setup_megasam_pipeline()