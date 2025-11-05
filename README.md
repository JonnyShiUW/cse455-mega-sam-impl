# MegaSaM Implementation

## Team Members
- Jonathan Shi
- Amogh Dhumal
- Jonathan Wang
- Jameson Rudolph

## Project Overview

MegaSaM is a deep visual SLAM system designed to accurately recover the camera's path and the 3D structure of a scene from casual/handheld videos of dynamic scenes. It addresses common failure modes of traditional methods (moving objects and limited camera motion) by implementing innovations like filtering out dynamic elements, using smart single image depth starting points, and adjusting its approach based on uncertainty.

## Project Goal

The goal of this project is to implement the core innovations of the paper **"MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos"** (Li et al., CVPR 2025). We will build upon a foundational SLAM (Simultaneous Localization and Mapping) and/or SfM (Structure from Motion) pipeline to accurately recover the 3D scene structure and camera trajectory from a single short dynamic handheld video.

## Implementation Focus

Our implementation will focus on integrating three key innovations:

1. **Moving object filtering** - Robust handling of dynamic scene elements
2. **Smarter depth starting point** - Monocular depth priors for better initialization
3. **Uncertainty-aware optimization strategy** - Adaptive optimization based on constraint quality

We are currently pursuing **Option 1: Pure Implementation**, with possible future extensions to novel benchmarking or applications.

## Deliverables

- Estimated camera trajectory
- Dense depth maps for the video sequence
- Visualized 3D point cloud of the static scene
- Working pipeline for both static and dynamic scenes with minor motion from handheld cameras (low-parallax scenarios)

## Resources

### Paper and Project Links
- [MegaSaM Project Page](https://mega-sam.github.io/)
- [ArXiv Paper](https://arxiv.org/abs/2412.04463)
- [Reddit Discussion](https://www.reddit.com/r/ResearchML/comments/1o9xmw4/from_shaky_phone_footage_to_3d_worlds_summary_of/)

### Pre-trained Modules
- **RAFT**: [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) - Optical flow estimation
- **DepthAnything**: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) - Monocular depth estimation

## Project Milestones

### Week 1: Foundational SLAM Pipeline (Static Scene)

We'll start by implementing the base DROID-SLAM system for static scenes that forms the foundation of the paper's innovations.

**Components:**
- Maintaining state of camera poses along key frames
- Low-resolution depth maps ✝
- Naive bundle-adjustment layer ✝
- Optical flow for feature correspondence
- Integration of pre-trained modules like RAFT

_✝ Designates future iteration/improvement in a later milestone_

### Week 2: Extending to Dynamic Scene Robustness (Filtering)

Increase the robustness of our pipeline to handle dynamic scenes and integrate monocular depth priors for low-parallax use cases.

**Components:**
- Incorporate monocular depth priors
- Integrate pre-trained DepthAnything module
- Update BA layer via regularization term to stick close to local prior from DepthAnything ✝
- Integrate motion mask in BA optimization
- Implement motion probability map

### Week 3: Uncertainty Awareness + Refinement

Implementation of the paper's flagship innovation: uncertainty-aware global bundle adjustment scheme.

**Components:**
- Extend BA optimization from local priors to backend global optimization over all keyframes
- Find epistemic uncertainty using Laplacian of the Hessian matrix
- Adapt based on uncertainty values:
  - Low Hessian values → high uncertainty → unobservable depth
  - Enable depth prior regularization when appropriate
- Pose graph adjustment for skipped between-frames

### Week 4: Optimization + Evaluations/Extensions

Refine the entire pipeline and conduct quantitative evaluations.

**Components:**
- Optimize depth map to full resolution using optimized camera poses/optical flow
- Set loss functions for temporal consistency loss and prior loss
- Run ADAM optimizer (potentially use aleatoric uncertainty map during optimization)
- Quantitative evaluation using metrics like ATE or RPE
- Compare against alternative methods like COLMAP
- TBD extensions

## Technical Background

### SLAM (Simultaneous Localization and Mapping)
A computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it.

### SfM (Structure from Motion)
The process of estimating three-dimensional structures from two-dimensional image sequences, coupled with local motion signals.

### Bundle Adjustment (BA)
A refinement technique that simultaneously adjusts camera parameters and 3D point positions to minimize reprojection error.
