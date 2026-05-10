# WildGS-SLAM: Codebase Explanation

**WildGS-SLAM** (CVPR 2025) is a **monocular SLAM system** for dynamic environments. It tracks camera pose and builds a 3D Gaussian Splatting map of the **static** parts of the scene, while filtering out dynamic distractors using a learned uncertainty model.

---

## Repository Structure

```
src/
  slam.py              # Top-level orchestrator (SLAM class)
  tracker.py           # Tracking thread: orchestrates motion filter + frontend + online BA
  frontend.py          # Local sliding-window bundle adjustment (keyframe management)
  backend.py           # Global bundle adjustment + loop closure
  motion_filter.py     # Per-frame motion check; feature extraction; keyframe selection
  factor_graph.py      # Factor graph: edge management, correlation volumes, update calls
  depth_video.py       # Shared state: poses, disparities, features (between tracker & mapper)
  mapper.py            # Mapping thread: Gaussian Splatting map optimization
  config.py            # Config loading
  geom/
    ba.py              # Differentiable bundle adjustment (full, motion-only, scale+shift)
    projective_ops.py  # Projection / reprojection math
    chol.py            # Cholesky / Schur complement solvers
  modules/droid_net/
    droid_net.py       # DroidNet: fnet (feature), cnet (context), UpdateModule (GRU-based)
    corr.py            # Correlation volume (CorrBlock, AltCorrBlock)
    extractor.py       # BasicEncoder (ResNet-like)
    gru.py             # ConvGRU
  utils/
    dyn_uncertainty/
      uncertainty_model.py  # Small MLP: DINOv2 features → per-pixel uncertainty
      mapping_utils.py      # Uncertainty-weighted loss computation
    mono_priors/
      metric_depth_estimators.py  # DepthAnythingV2 / Metric3D wrappers
      img_feature_extractors.py   # DINOv2 feature extraction
    slam_utils.py       # Loss functions (tracking, mapping, uncertainty)
    datasets.py         # Dataset loaders
    eval_traj.py        # ATE evaluation
  gui/                  # OpenGL-based real-time visualizer

thirdparty/
  gaussian_splatting/   # 3D Gaussian Splatting renderer (GaussianModel, render)
  depth_anything_v2/    # DepthAnythingV2 monocular metric depth network

configs/                # YAML configs (wildgs_slam.yaml, Dynamic/, Static/)
run.py                  # Entry point
```

---

## 1. SLAM Algorithm: Frontend, Backend, Options

### Architecture (two parallel processes)

```
Input stream → Tracker process ──pipe──→ Mapper process
                    ↑                          ↑
               DepthVideo (shared memory: poses, disps, features)
```

### Tracker Process (`tracker.py`)

Runs three stages per frame:

#### 1. MotionFilter (`motion_filter.py`)
For every incoming frame, extracts feature maps (`fnet`) and context maps (`cnet`) using DroidNet's `BasicEncoder`. Computes a one-step optical flow estimate against the last keyframe using `CorrBlock`. If the flow magnitude exceeds a threshold (`motion_filter.thresh`, default ~2.5 px), the frame becomes a **keyframe candidate**. There is also a `force_keyframe_every_n_frames` option for safety.

#### 2. Frontend (`frontend.py`)
Local sliding-window bundle adjustment over the most recent `frontend_window` keyframes (default ~20). The factor graph is maintained with **proximity factors** (edges between frames with small camera distance). Uses `iters1=8` GRU update iterations per keyframe, then drops the keyframe if camera motion is below `keyframe_thresh`. Optionally does **loop closing** via `loop_ba` (Backend) when `enable_loop=True`. Poses are represented as Lie-group SE(3) quaternions (`lietorch`).

#### 3. Online Backend (`backend.py`)
Runs `dense_ba` every `ba_freq` keyframes globally over all keyframes using a denser proximity factor graph. Uses `update_lowmem` for memory-efficient GRU iteration in chunks of 8.

### Mapper Process (`mapper.py`)
Receives each new keyframe from the tracker pipe. Maintains a `GaussianModel` and optimizes it jointly with exposure compensation parameters using rendered RGBD losses (L1 + SSIM + depth). Also runs `_update_keyframes_from_frontend` to pull updated poses/depths from the shared `DepthVideo`.

### Final Global BA (`slam.py::backend()`)
After tracking is done, runs two passes of `dense_ba` (steps=7, then steps=12) over all keyframes without metric depth regularization, then final GS refinement (`final_refine`).

### Key Config Options

| Config key | Description |
|---|---|
| `tracking.frontend.window` | Local sliding window size |
| `tracking.backend.final_ba` | Enables final global BA after tracking |
| `tracking.backend.metric_depth_reg` | Regularizes BA with monocular metric depth |
| `tracking.backend.ba_freq` | Online BA frequency (every N keyframes) |
| `mapping.uncertainty_params.activate` | Enables uncertainty-aware mapping |
| `fast_mode` | Maps only every 4 keyframes (faster) |
| `tracking.motion_filter.thresh` | Motion threshold for keyframe selection |
| `tracking.frontend.enable_loop` | Enables loop closure in frontend |

---

## 2. How Optical Flow is Integrated

Optical flow is **not an external pre-computed input**; it is **implicitly estimated inside the DroidNet** through a differentiable correlation volume mechanism:

1. `fnet` (BasicEncoder) extracts feature maps at **1/8 resolution** for each frame.
2. `CorrBlock` builds a **4-level correlation pyramid** between feature maps of frame `i` and frame `j` by computing all-pairs inner products.
3. Given current estimated 2D pixel correspondences `coords1` (reprojected using current pose + disparity estimates), the correlation block is **sampled at `coords1`** using a radius-3 neighborhood → this gives local correlation features.
4. **Motion features** `motn` are computed as `[coords1 - coords0, target - coords1]` (2+2 = 4 channels), encoding both the current flow and the residual to the target.
5. The `UpdateModule` fuses correlation features + motion features + context features through a **ConvGRU** and outputs:
   - `delta`: predicted flow correction (2D per pixel) → updates `target = coords1 + delta`
   - `weight`: per-pixel confidence weights for bundle adjustment
   - `upmask`: 8×8 softmax mask for convex upsampling of disparities

Flow estimation and pose/depth optimization are **tightly coupled** and mutually refined through iterative GRU updates (DROID-SLAM style).

---

## 3. How Motion Estimation is Integrated

Camera motion is estimated via **differentiable bundle adjustment** (`geom/ba.py`), not with a separate VO step:

- The factor graph holds edges `(i, j)` between keyframes. For each edge, the reprojection targets `target` and weights `weight` come from the DroidNet update.
- `BA()` linearizes the reprojection error w.r.t. SE(3) poses and inverse disparities, builds the normal equations (Hessian H and gradient v), and solves the Schur complement system `(H - E C⁻¹ Eᵀ) dx = v - E C⁻¹ w` using Cholesky factorization.
- **Special variant `BA_with_scale_shift()`**: WildGS-SLAM's key contribution for monocular depth integration. Jointly optimizes disparities `z`, per-keyframe **scale** `s` and **shift** `q` that align monocular metric depth to the geometric disparity (i.e., `z ≈ s·z_mono + q`). This is the metric depth regularization (`metric_depth_reg`).
- `MoBA()` (Motion-only BA) fixes depths and optimizes only poses — used for fast motion-only steps.
- Pose representation: Lie-group SE3 quaternions via `lietorch`, retraction via `pose_retr()`.

The `droid_backends` CUDA extension computes a geometry-aware frame distance metric mixing optical flow magnitude and depth difference (controlled by `beta`), used for proximity-based edge construction.

### Uncertainty-Aware Tracking

When `uncertainty_params.activate=True`:
- DINOv2 features are extracted per keyframe and stored in `DepthVideo.dino_feats`.
- A small **MLP** (`uncertainty_model.py`) maps DINOv2 features → per-pixel uncertainty scalar.
- The MLP is trained online during mapping by minimizing a heteroscedastic loss against rendered vs. real image residuals.
- The resulting uncertainty mask is fed back to tracking to **down-weight** regions with dynamic objects in both the tracking loss and the valid depth mask used in BA.

---

## Key Technologies

| Technology | Role |
|---|---|
| **DroidNet** (DROID-SLAM) | Feature/context extraction + GRU-based optical flow update |
| **Differentiable BA** | Joint pose + depth + scale/shift optimization |
| **lietorch** | Lie-group SE3 pose arithmetic |
| **droid_backends** | CUDA kernels: correlation sampling, frame distance, BA backend |
| **3D Gaussian Splatting** | Dense photorealistic map representation |
| **DepthAnythingV2 / Metric3D** | Monocular metric depth prior |
| **DINOv2** | Image features for uncertainty prediction |
| **torch_scatter** | Scatter operations in BA linear system assembly |
