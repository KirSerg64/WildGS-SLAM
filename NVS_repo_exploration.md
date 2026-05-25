# Agents · KirSerg64/WildGS-SLAM

Now I have a comprehensive understanding of the repository. Let me provide the full analysis.

## Comprehensive Analysis: Novel View Synthesis for WildGS-SLAM

## 1. High-Level Understanding of the Task

This is a **competition task** for autonomous driving novel view synthesis. The problem is:

- **Input**: 12 images (6 cameras × 2 timestamps), a dense LiDAR point cloud (300K–1.5M points), camera poses (4×4 C2W), and intrinsics.
- **Output**: Predict a single image from a **target camera at an intermediate time** (approximately midpoint between t0 and t1).
- **Metric**: PSNR, normalized to [0, 100] with 10 dB → 0 and 30 dB → 100.

**Critical insight**: This is **not** a standard SLAM task. It is a **per-sample feed-forward/optimization NVS task** with:

- Known poses (no tracking needed)
- Known geometry (LiDAR point cloud provided)
- Short temporal baseline (1–2 seconds)
- Multi-camera rig (360° coverage)
- Dynamic scenes (autonomous driving, moving cars, pedestrians)
- Target is an **interpolated** viewpoint in time, not extrapolated

The WildGS-SLAM pipeline provides Gaussian Splatting rendering, depth priors, uncertainty-aware mapping, and dynamic object handling. Those are relevant, but the **tracking/frontend/backend** modules are not needed because poses are given.

## 2. Current WildGS-SLAM Architecture Analysis

### Core Pipeline

```text
Stream → MotionFilter → Frontend (Local BA) → Backend (Global BA)
                                    ↓
                              DepthVideo (shared state)
                                    ↓
                              Mapper (Gaussian Splatting)
                                    ↓
                          GaussianModel + Uncertainty MLP
```

### Key Components and Their Relevance to NVS Task

| Component | File | Relevant? | Why |
| --- | --- | --- | --- |
| `SLAM` orchestrator | `src/slam.py` | ❌ | Multi-process tracking not needed |
| `Tracker` | `src/tracker.py` | ❌ | Poses are given |
| `Frontend` | `src/frontend.py` | ❌ | No BA needed |
| `Backend` | `src/backend.py` | ❌ | No global BA needed |
| `DepthVideo` | `src/depth_video.py` | ⚠️ Partial | Depth storage structure useful, but overkill |
| `Mapper` | `src/mapper.py` | ✅ Core | GS optimization, rendering, densification |
| `GaussianModel` | `thirdparty/gaussian_splatting/scene/gaussian_model.py` | ✅ Core | Gaussian representation |
| `render()` | `thirdparty/gaussian_splatting/gaussian_renderer/__init__.py` | ✅ Core | Differentiable rendering |
| `Camera` | `src/utils/camera_utils.py` | ✅ Core | Viewpoint representation |
| `Uncertainty MLP` | `src/utils/dyn_uncertainty/uncertainty_model.py` | ✅ Useful | Dynamic masking |
| `DINOv2 features` | Used for uncertainty | ✅ Useful | Semantic dynamic detection |
| `Depth Anything V2` | `thirdparty/depth_anything_v2/` | ⚠️ Optional | LiDAR already provides geometry |
| `diff-gaussian-rasterization` | `thirdparty/diff-gaussian-rasterization-w-pose/` | ✅ Core | CUDA rasterizer |

### Data Structures Already Sufficient

- `GaussianModel`: stores `_xyz`, `_features_dc/rest`, `_scaling`, `_rotation`, `_opacity` — **perfect** for this task
- `Camera` class: handles projection matrices, W2C transforms, exposure compensation — **directly usable**
- `render()` function: handles all rasterization — **directly usable**
- `create_pcd_from_image_and_depth()`: unprojection to point cloud — relevant for initialization

## 3. Feasibility Analysis

### What Makes This Task Tractable

1. LiDAR provides dense geometry, so there is no need for monocular depth estimation or SfM.
2. Poses are given, so tracking/SLAM is not required.
3. 12 input views provide near-360° coverage, so sparse-view generalization is less critical.
4. The target is temporal interpolation, so geometric overlap with inputs is high.
5. The 1–2 second baseline means limited dynamic motion.

### What Makes This Task Challenging

1. Dynamic objects: cars and pedestrians move between t0 and t1; the target is at mid-time.
2. View-dependent effects: specular reflections on car bodies, wet roads.
3. Occlusion handling: regions visible from the target but occluded in inputs.
4. LiDAR-camera misalignment: temporal aggregation over 3× delta introduces motion artifacts.
5. Limited optimization time: competition likely expects reasonable runtime per sample.
6. Distortion: non-pinhole distortion models are present in camera params.

### Feasibility Verdict

**Immediately implementable** (days): Direct 3DGS optimization per sample using LiDAR initialization + 12-view photometric optimization + dynamic masking → expected PSNR ~22–26 dB.

**Research-grade improvements** (weeks): Flow-based dynamic Gaussian tracking, diffusion-based inpainting of occluded regions, per-Gaussian temporal interpolation → potentially 26–30+ dB.

## 4. Proposed NVS Pipeline

### Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         PER-SAMPLE PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [LiDAR .npz] ──→ Point Cloud Preprocessing ──→ Initial Gaussians   │
│                    (filter, colorize, normal est.)                  │
│                                                                     │
│  [12 Images] ──→ Undistortion ──→ Feature Extraction (DINO)         │
│                                                                     │
│  [Camera Poses] ──→ Camera Objects (C2W → W2C)                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              GAUSSIAN OPTIMIZATION (500–3000 iters)           │   │
│  │                                                               │   │
│  │  For each iter:                                               │   │
│  │    1. Sample viewpoints from 12 input cameras                 │   │
│  │    2. Render RGB + depth                                      │   │
│  │    3. Compute L1 + SSIM + depth loss                          │   │
│  │    4. Uncertainty-weighted loss (suppress dynamics)           │   │
│  │    5. Densify/prune every N iterations                        │   │
│  │    6. Update Gaussian parameters (xyz, color, scale, rot, α)  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  [Target Pose] ──→ Interpolate dynamic Gaussians to t_mid          │
│                ──→ Render from target Camera                       │
│                ──→ Post-process (exposure, denoise)                │
│                ──→ Output Image                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Rendering Flow (Tensor Dimensions)

```text
LiDAR points: (N, 3) float32 → filter → (M, 3), M ≈ 100K–500K
Colorize via 12-view projection: (M, 3) RGB
Initial Gaussians:
  _xyz: (M, 3)
  _features_dc: (M, 1, 3)  [SH degree 0]
  _scaling: (M, 3)
  _rotation: (M, 4)
  _opacity: (M, 1)

Per iteration:
  Render → (3, H, W) RGB, (1, H, W) depth, (1, H, W) alpha
  Loss computation → scalar
  Backward → gradients on all Gaussian params

Final render at target pose: (3, H_target, W_target) → save as .jpg
```

## 5. Proposed Modifications to Repository

### New Modules to Introduce

| Module | Purpose |
| --- | --- |
| `src/nvs_pipeline.py` | Main NVS entry point, replaces `run.py` for this task |
| `src/nvs_mapper.py` | Simplified mapper without tracking communication |
| `src/utils/lidar_utils.py` | LiDAR loading, filtering, colorization from multi-view |
| `src/utils/nvs_dataset.py` | Dataset loader for the competition format (`meta.json`, images, LiDAR) |
| `src/utils/dynamic_handler.py` | Dynamic object detection and temporal interpolation |
| `src/utils/pose_interpolation.py` | SE3 interpolation (SLERP for rotation, linear for translation) |
| `configs/nvs_competition.yaml` | Configuration for the NVS task |

### Existing Modules to Modify

| Module | Modification |
| --- | --- |
| `src/mapper.py` | Extract `map_opt_online()` and `final_refine()` logic into reusable functions, or subclass for NVS |
| `src/utils/camera_utils.py` | Add distortion handling (fisheye/barrel models); add factory from `meta.json` intrinsics |
| `thirdparty/gaussian_splatting/scene/gaussian_model.py` | Add `init_from_point_cloud_colored()` that accepts pre-colored points without O3D conversion |
| `src/utils/slam_utils.py` | Reuse `get_loss_mapping()` / SSIM loss as-is |

### Modules That Need No Changes

- `thirdparty/gaussian_splatting/gaussian_renderer/__init__.py` — `render()` works as-is
- `thirdparty/diff-gaussian-rasterization-w-pose/` — CUDA rasterizer is fine
- `src/utils/dyn_uncertainty/uncertainty_model.py` — MLP architecture is reusable

## 6. Candidate Rendering Strategies

### Strategy A: Direct Per-Sample 3DGS Optimization (Recommended Baseline)

**Approach**: Initialize Gaussians from LiDAR, optimize against 12 views, render target.

**Pros**: Simple, well-understood, leverages existing codebase.

**Cons**: Slow per sample (2–5 min with 2000 iters); struggles with dynamics.

**Expected PSNR**: 22–26 dB

**Implementation effort**: 2–3 days using existing mapper code

### Strategy B: Feed-Forward Point Splatting (No Optimization)

**Approach**: Project LiDAR → color from nearest input view → splat directly to target.

**Pros**: Extremely fast (~100 ms); no optimization.

**Cons**: Holes, no view-dependent effects, no anti-aliasing.

**Expected PSNR**: 15–20 dB

**Implementation effort**: 1 day

### Strategy C: Hybrid — Quick Init + Short Optimization

**Approach**: Strategy B for initialization, then 200–500 iterations of GS optimization.

**Pros**: Best quality/time tradeoff.

**Cons**: Still requires optimization per sample.

**Expected PSNR**: 24–28 dB

**Implementation effort**: 3–4 days

### Strategy D: Diffusion-Assisted NVS

**Approach**: Render coarse image via Strategy B/C, then use a pretrained diffusion model (e.g. Zero123++) for refinement.

**Pros**: Can hallucinate occluded/dynamic regions.

**Cons**: PSNR may be unstable, heavy GPU memory, slow.

**Verdict**: Research-grade, not immediately practical for competition.

### Strategy E: Latent Feature Splatting

**Approach**: Encode input images to latent features, splat features to target view, decode.

**Pros**: Handles view-dependent effects; generalizes.

**Cons**: Requires training a decoder; no pretrained model available for this rig.

**Verdict**: Research-grade.

### Strategy F: Depth-Guided Warping + Inpainting

**Approach**: Warp closest input images to target view using LiDAR depth, composite, inpaint holes.

**Pros**: Simple, fast, handles static scene well.

**Cons**: Artifacts at depth edges, struggles with dynamics.

**Expected PSNR**: 20–24 dB

**Implementation effort**: 2 days

**Recommendation**: Implement Strategy C as the primary approach, with dynamic handling from Section 7.

## 7. Dynamic-Scene Handling Strategies

### Challenge

The target view is at t_mid ≈ (t0 + t1) / 2. Dynamic objects have moved between t0 and t1. The LiDAR point cloud is aggregated over 3× delta, so dynamic objects create "ghost" trails.

### Strategy 7.1: Semantic Filtering of Dynamic Gaussians

- Use a pretrained 2D detector (YOLO/SAM) or DINO features on input images to identify dynamic objects (cars, pedestrians)
- Remove LiDAR points that project into dynamic masks
- Reconstruct only static background with 3DGS
- Fill dynamic regions via simple warping from nearest temporal view
- **Pros**: Clean static reconstruction; WildGS already has DINO-based uncertainty
- **Cons**: Misses dynamic content in target image

### Strategy 7.2: Per-Gaussian Temporal Label

- Assign each Gaussian a temporal label: which timestep(s) it belongs to
- During LiDAR initialization: determine if a point is static (consistent across views) or dynamic
- For dynamic Gaussians: interpolate position to t_mid using a linear motion model
- **Pros**: Preserves dynamic objects in rendering
- **Cons**: Requires reliable motion estimation

### Strategy 7.3: Two-Stage Optimization

1. **Stage 1**: Optimize Gaussians using all 12 views, but with uncertainty-weighted loss (high uncertainty = dynamic region). This builds a clean static map.
2. **Stage 2**: For the specific target camera, identify which input images are closest in viewpoint. Optimize **additional** Gaussians only in the dynamic regions, using temporal interpolation.

- **Pros**: Clean separation; leverages existing uncertainty MLP architecture
- **Cons**: Complex implementation

### Strategy 7.4: Scene Flow from Consecutive Frames

- Compute optical flow between t0 and t1 views of the same camera
- Use flow to estimate 3D scene flow for dynamic objects
- Interpolate dynamic point positions to t_mid
- **Tools**: RAFT/GMFlow for flow, then depth + flow -> 3D motion
- **Pros**: Physics-grounded motion interpolation
- **Cons**: Requires flow estimation; ambiguity in multi-object scenes

**Recommendation**: Start with 7.1 (mask and remove), then add 7.4 for dynamic content recovery.

## 8. Pose Generation/Interpolation Strategies

### The Target Pose

The target pose is **given** in `meta.json` -> `poses_c2w.target`. No interpolation is strictly needed for the target camera extrinsics.

### However - Dynamic Object Motion Interpolation

For moving objects, we need to interpolate their positions:

```python
def interpolate_pose_se3(pose_t0, pose_t1, alpha=0.5):
  """Spherical linear interpolation between two SE3 poses."""
  import numpy as np
  from scipy.spatial.transform import Rotation, Slerp

  r0 = Rotation.from_matrix(pose_t0[:3, :3])
  r1 = Rotation.from_matrix(pose_t1[:3, :3])
  slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
  r_mid = slerp(alpha).as_matrix()

  t_mid = (1 - alpha) * pose_t0[:3, 3] + alpha * pose_t1[:3, 3]

  pose_mid = np.eye(4)
  pose_mid[:3, :3] = r_mid
  pose_mid[:3, 3] = t_mid
  return pose_mid
```

### Temporal Alpha Estimation

From `meta.json`: `delta_s` gives the time between t0 and t1. The target is at approximately t_mid, so alpha ≈ 0.5. This could be refined if the target timestamp is embedded in pose metadata.

## 9. Optimization and Training Strategy

### Loss Functions

For Gaussian optimization against the 12 input views:

```text
L_total = λ_rgb * L_rgb + λ_ssim * L_ssim + λ_depth * L_depth + λ_reg * L_reg

# Where:
L_rgb = ||rendered - gt||₁  (masked by uncertainty/static regions)
L_ssim = 1 - SSIM(rendered, gt)  [window_size=11]
L_depth = ||rendered_depth - lidar_depth||₁  (projected LiDAR to each view)
L_reg = opacity_reg + scale_reg  (prevent degenerate Gaussians)
```

Recommended weights: `λ_rgb=0.8, λ_ssim=0.2, λ_depth=0.1, λ_reg=0.001`

### Optimization Schedule

```text
Phase 1 (iters 0-200):    Position + opacity only, larger LR (0.001)
Phase 2 (iters 200-1000): All params, standard LR
Phase 3 (iters 1000-2000): Reduced LR, SH features if enabled
Densification: every 100 iters in phase 2
Pruning: opacity < 0.01, scale > 10x median
```

### GPU Memory Implications

- 500K Gaussians x (3+48+3+4+1) floats x 4 bytes ≈ 118 MB for parameters
- Plus gradients and optimizer states: x3 ≈ 354 MB
- 12 input images at 1920x1200x3: ≈ 83 MB
- Rendering buffer: ≈ 30 MB per view
- **Total**: ~1-2 GB, well within single GPU capacity (even RTX 3090)

### Real-Time Feasibility

- Per-sample optimization: 2000 iters x ~10 ms/iter = 20 seconds (optimistic) to 2 minutes (conservative)
- Feed-forward baseline: <1 second
- For competition: per-sample optimization is feasible if the test set is manageable (~100 samples)

## 10. Risks and Failure Cases

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Dynamic objects create ghosting | High | Uncertainty masking + semantic filtering |
| LiDAR sparsity in distant regions | Medium | Depth-guided densification from monocular priors |
| LiDAR temporal aggregation artifacts | High | Filter points by temporal consistency |
| Camera distortion not handled | Medium | Undistort images before optimization |
| Overfit to input views, poor generalization | Medium | Regularization + early stopping |
| Sky/unbounded regions have no LiDAR | High | Initialize sky Gaussians at far plane, or use environment map |
| Exposure differences between cameras | Medium | Per-camera exposure compensation (already in Camera class) |
| Competition time limits | Medium | Tune iteration count; precompute as much as possible |

## 11. Experimental Protocol

### Evaluation on Train Set

1. Split train samples into train/val (80/20)
2. Report PSNR, SSIM, LPIPS on validation
3. Ablate: (a) with/without dynamic masking, (b) number of iterations, (c) with/without depth loss, (d) initialization strategy

### Ablation Studies

| Experiment | Purpose |
| --- | --- |
| LiDAR init only (no optimization) | Baseline quality |
| + 500 iters RGB loss | Minimum useful optimization |
| + SSIM loss | Perceptual quality |
| + Depth loss from LiDAR | Geometric consistency |
| + Dynamic masking (DINO features) | Dynamic handling |
| + Per-camera exposure | Color consistency |
| + SH degree 3 | View-dependent effects |
| + Scene flow for dynamics | Dynamic object quality |

### Debugging Methods

1. **Render depth maps**: Compare rendered depth vs. projected LiDAR to validate alignment
2. **Render per-camera**: Check each of 12 views reconstructs well before target
3. **Visualize Gaussian distribution**: Use the GUI module (`src/gui/`) to inspect 3D structure
4. **Loss curves**: Track per-view and total loss; divergence indicates misalignment
5. **Uncertainty maps**: Visualize which regions are masked to ensure dynamics are suppressed

## 12. Open Questions

1. **What is the distortion model?** The `meta.json` mentions `distortion_model` and `distortion_coeffs` — need to know if it's Brown-Conrady, equidistant, or something else to properly undistort.
2. **Is the target always at exactly t_mid?** Or can it be at arbitrary intermediate times? This affects interpolation alpha.
3. **Are there runtime constraints?** If the competition limits inference time per sample, feed-forward approaches become necessary.
4. **LiDAR point cloud quality**: Are dynamic objects pre-filtered, or is the raw aggregation provided? The 3x delta aggregation window implies dynamic objects will have streaking.
5. **Ground truth availability**: Is the target image always from one of the 6 camera positions? (Yes, based on `target_camera` field.) This means we always have the same camera at t0 and t1, which is useful for temporal interpolation.
6. **Image resolution of target**: Is it the same as input (1920x1200 for front)? Need to confirm.

## 13. Recommended Implementation Roadmap

### Phase 1: Minimal Working Pipeline (3–5 days)

- Create `src/utils/nvs_dataset.py` — load `meta.json`, images, LiDAR
- Create `src/utils/lidar_utils.py` — load `.npz`, filter outliers, colorize from nearest views
- Create `src/nvs_pipeline.py` — end-to-end: load sample → init Gaussians → optimize → render → save
- Reuse `GaussianModel.create_pcd_from_image_and_depth()` logic for initialization
- Reuse `render()` + loss functions from `src/utils/slam_utils.py`
- Handle camera distortion (undistort input images)
- Validate on train set — target PSNR ≥ 20 dB

### Phase 2: Quality Improvements (3–5 days)

- Add dynamic masking using DINO features + uncertainty MLP
- Add per-camera exposure optimization
- Add depth regularization loss using projected LiDAR
- Tune optimization schedule (iterations, learning rates, densification)
- Add sky/far-field handling
- Target PSNR ≥ 24 dB

### Phase 3: Dynamic Object Handling (5–7 days)

- Implement optical flow between t0 → t1 same-camera pairs
- Estimate 3D scene flow for dynamic points
- Interpolate dynamic Gaussians to t_mid
- Or warp closest temporal view's dynamic regions into target
- Target PSNR ≥ 26 dB

### Phase 4: Advanced (Research-Grade, 1–2 weeks)

- Multi-scale Gaussian representation for level-of-detail
- Anisotropic Gaussian initialization from local surface normals
- Test-time augmentation (render from multiple nearby poses, average)
- Learned post-processing network (U-Net on rendered image)
- Consider pixelSplat/MVSplat-style feed-forward baseline for speed

## Summary of Key Technical Decisions

| Decision | Recommendation | Rationale |
| --- | --- | --- |
| Use tracking? | No | Poses given |
| Use monocular depth? | Optional | LiDAR provides metric geometry |
| Initialization | LiDAR point cloud → Gaussians | Dense, metric, covers scene |
| Optimization | Per-sample, 1000–2000 iters | Quality vs. speed tradeoff |
| Dynamic handling | Uncertainty masking (Phase 2) + flow (Phase 3) | Progressive improvement |
| Renderer | Existing `render()` from 3DGS | Already integrated, CUDA-optimized |
| Camera model | Undistort to pinhole first | Simplifies rendering pipeline |
| SH degree | Start with 0, try 3 | View-dependent effects from specular surfaces |
| Loss | L1 + 0.2×SSIM + 0.1×depth | Standard 3DGS recipe + depth supervision |

The WildGS-SLAM codebase provides about 70% of the needed infrastructure. The main engineering effort is building the data loading pipeline for the competition format and stripping the SLAM orchestration to create a direct per-sample optimization loop.
