# Comprehensive Analysis: Novel View Synthesis for WildGS-SLAM

## Task Overview

This is an autonomous-driving Novel View Synthesis (NVS) competition. For each sample:

| Input | Description |
|-------|-------------|
| 12 images | 6 cameras × 2 time points (t0, t1), 1–2 seconds apart |
| LiDAR point cloud | Aggregated scans around [t0, t1] in world coordinates |
| Camera poses (c2w) | 4×4 matrices for all 12 input views **and** the target view |
| Camera intrinsics | fx, fy, cx, cy, width, height, distortion coefficients |

**Goal**: predict the image from the specified target camera at an intermediate time (≈ midpoint between t0 and t1).  
**Metric**: PSNR (normalized to [0, 100] over the 10–30 dB range).

---

## Why WildGS-SLAM Is a Good Fit

WildGS-SLAM builds a **3D Gaussian Splatting** map from monocular video while handling dynamic distractors. For this competition:

- Poses are **given** — no tracking needed; the mapper can run in pose-supervised mode.
- The GS map learned from the 12 input views can be **rendered from any novel pose** via `render()`.
- The uncertainty model can be reused to suppress dynamic objects (other vehicles, pedestrians) that appear in some input frames but not the target view.
- `eval_map/nvs_wild_slam.py` already demonstrates the full render-from-pose pipeline.

---

## High-Level Implementation Plan

### Step 1 — Data Loader for the Competition Format

Create `src/utils/datasets_nvs.py` with a `NVSCompetitionDataset` class that:

1. Reads `meta.json` to get `target_camera`, `poses_c2w`, `intrinsics`, `delta_s`.
2. Loads the 12 input images from `input/t0/` and `input/t1/`.
3. Applies distortion correction (`cv2.undistort`) using each camera's `distortion_coeffs`.
4. Converts `poses_c2w` (camera-to-world) to `w2c` (world-to-camera) for the GS renderer.
5. Returns a list of `Camera` objects (one per input view) ready for the mapper.

Key conversion:
```python
# meta.json uses c2w; GS renderer needs w2c (world-to-camera)
w2c = np.linalg.inv(pose_c2w)
R = torch.tensor(w2c[:3, :3], dtype=torch.float32, device='cuda')
T = torch.tensor(w2c[:3, 3],  dtype=torch.float32, device='cuda')
viewpoint.update_RT(R, T)
```

---

### Step 2 — LiDAR-Based Gaussian Initialization

Instead of growing Gaussians from scratch, **seed them from the LiDAR point cloud** in `lidar.npz`:

```python
import numpy as np
points = np.load('lidar.npz')['xyz']        # (N, 3) world coords
colors = np.ones((len(points), 3)) * 0.5    # gray initial color
gaussians.create_from_pcd(points, colors, spatial_lr_scale=1.0)
```

This gives the optimizer a strong geometric prior, dramatically reducing the number of iterations needed and avoiding floaters in textureless regions (road, sky).

Recommended initialization parameters:
- Filter points by distance to the target camera center (keep within ~60 m).
- Downsample to ≤500k points with voxel downsampling (voxel size ≈ 0.1 m).

---

### Step 3 — Pose-Supervised Gaussian Optimization

Since poses are provided, skip the tracker entirely. Run only the mapper with fixed poses:

```python
# Pseudocode for the per-sample optimization loop
gaussians = GaussianModel(sh_degree=0, config=cfg)
gaussians.initialize_from_lidar(lidar_xyz)
optimizer = gaussians.training_setup(cfg['mapping']['Training'])

for iteration in range(cfg['mapping']['iters']):        # e.g., 1000 iters
    viewpoint = random.choice(input_viewpoints)
    render_pkg = render(viewpoint, gaussians, pipe, background)
    loss = l1_loss(render_pkg['render'], viewpoint.gt_color)
    loss += cfg['lambda_ssim'] * (1.0 - ssim(render_pkg['render'], viewpoint.gt_color))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    gaussians.update_learning_rate(iteration)
    gaussians.adaptive_density_control(iteration, cfg)  # densify + prune
```

Key config settings for the competition:
- `sh_degree: 0` — faster; color is view-independent for a quasi-static scene.
- `iters: 800–1500` — more than standard SLAM per-frame budget, since this is offline.
- `lambda_ssim: 0.2` — standard GS weighting.

---

### Step 4 — Render the Target View

After optimization, render from the target pose using the existing pipeline from `eval_map/nvs_wild_slam.py`:

```python
target_intrinsic = meta['intrinsics'][meta['target_camera']]
target_pose_c2w  = np.array(meta['poses_c2w']['target'][meta['target_camera']])
target_w2c       = torch.tensor(np.linalg.inv(target_pose_c2w), dtype=torch.float32, device='cuda')

viewpoint = get_temp_viewpoint(target_intrinsic, full_resol=True)
viewpoint.update_RT(target_w2c[:3, :3], target_w2c[:3, 3])

with torch.no_grad():
    render_pkg = render(viewpoint, gaussians, pipe, background)
    prediction  = torch.clamp(render_pkg['render'], 0.0, 1.0)

output_rgb = (prediction.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
cv2.imwrite(output_path, cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
```

---

### Step 5 — Entry Point Script

Create `scripts_run/run_nvs_competition.py`:

```
usage: python scripts_run/run_nvs_competition.py \
           --data_root /path/to/dataset/test \
           --output_dir /path/to/predictions \
           --config configs/NVS/competition.yaml
```

The script:
1. Iterates over all sample folders in `data_root`.
2. For each sample: loads data → initializes Gaussians → optimizes → renders → saves.
3. Writes one PNG per sample named `<sample_id>.png`.

---

## Key Files to Create / Modify

| Action | File |
|--------|------|
| Create | `src/utils/datasets_nvs.py` — competition data loader |
| Create | `src/utils/lidar_init.py` — LiDAR-to-Gaussian seeding |
| Create | `configs/NVS/competition.yaml` — optimized config for the task |
| Create | `scripts_run/run_nvs_competition.py` — batch inference script |
| Modify | `eval_map/utils.py` — add `get_temp_viewpoint_nvs()` variant accepting `cx/cy` keys |
| Reuse  | `eval_map/utils.py` — `get_temp_viewpoint()` (already fits) |
| Reuse  | `thirdparty/gaussian_splatting/gaussian_renderer/__init__.py` — `render()` |
| Reuse  | `thirdparty/gaussian_splatting/scene/gaussian_model.py` — `GaussianModel` |

---

## Exploring `eval_map/nvs_wild_slam.py` for Reuse

### What the Script Does

`eval_map/nvs_wild_slam.py` is a **Novel View Synthesis evaluation pipeline** for WildGS-SLAM. It:

1. **Loads a pre-trained Gaussian Splatting model** (`final_gs.ply`) produced by a full SLAM run.
2. **Aligns the estimated trajectory to ground-truth** using scale recovery and an SE3 transformation (`T_we_wg`).
3. **Renders novel views** from static camera poses (from `nvs/groundtruth.txt`) that were **not** in the training trajectory.
4. **Computes metrics** (PSNR, SSIM, LPIPS) against ground-truth images at those novel poses.
5. **Saves visualizations** (rendered vs. GT side-by-side).

### Rendering Pipeline Template (Immediately Reusable)

Lines 74–92 show the exact recipe for rendering from an arbitrary pose using a loaded Gaussian model:

```python
gaussians = GaussianModel(0, config=None)
gaussians.load_ply(os.path.join(exp_folder, 'final_gs.ply'))
pipe = get_render_pipline_params(exp_folder)

viewpoint = get_temp_viewpoint(intrinsics, full_resol=full_resol, exp_cfg=cfg)
viewpoint.update_RT(T_c_we[:3, :3], T_c_we[:3, 3])

with torch.no_grad():
    rendering_pkg = render(viewpoint, gaussians, pipe, background)
    image = torch.clamp(rendering_pkg["render"], 0.0, 1.0)
```

This is **exactly** what the competition pipeline needs for the final rendering step.

### `get_temp_viewpoint()` in `eval_map/utils.py` (Directly Reusable)

This utility shows how to construct a `Camera` object from **arbitrary intrinsics** (fx, fy, cx, cy, W, H) without needing an actual image or depth — just a pose and projection matrix.

### Coordinate Transform Logic

Lines 52–66 demonstrate how to align estimated SLAM coordinates to ground-truth world coordinates. For the competition, poses are given directly in a common world frame, so this alignment step is not needed.

### Metric Computation (Directly Reusable for Validation)

Lines 106–108 show PSNR/SSIM/LPIPS computation:

```python
psnr_score  = psnr(image[mask].unsqueeze(0),  gt_image[mask].unsqueeze(0)).item()
ssim_score  = ssim(image.unsqueeze(0),         gt_image.unsqueeze(0)).item()
lpips_score = cal_lpips(image.unsqueeze(0),    gt_image.unsqueeze(0)).item()
```

### Distortion Handling (Important Pattern)

Line 98 applies `cv2.undistort()` to GT images before metric computation — confirming that both GT images and input views should be undistorted before being passed to the GS renderer.

---

## Key Reusable Components

| Component | File | Use in Competition |
|-----------|------|-------------------|
| `get_temp_viewpoint()` | `eval_map/utils.py` | Build target `Camera` from `meta.json` intrinsics |
| `render()` call pattern | `eval_map/nvs_wild_slam.py` L90–92 | Final rendering step |
| `GaussianModel.load_ply()` | `eval_map/nvs_wild_slam.py` L74–75 | Load optimized model after training |
| `get_render_pipline_params()` | `eval_map/utils.py` | Load pipeline params from config |
| PSNR/SSIM calculation | `eval_map/nvs_wild_slam.py` L106–108 | Local evaluation on train split |
| `cv2.undistort` pattern | `eval_map/nvs_wild_slam.py` L98 | Undistort input and GT images |

---

## What's Missing (Must Be Added)

- **LiDAR point cloud → Gaussian initialization** — `lidar.npz` → `GaussianModel.create_from_pcd()`
- **Multi-view photometric optimization loop** — 12 input views; no SLAM tracking needed
- **Dynamic scene handling** — mask out other vehicles/pedestrians between t0 and t1
- **Competition data loader** — parse `meta.json`, load 12 input images, build `Camera` objects

---

## How to Adapt the Existing Script for the Competition

| Current Script | Needed Adaptation |
|---|---|
| Loads `final_gs.ply` from SLAM output | Initialize Gaussians from LiDAR + optimize per sample |
| Aligns SLAM trajectory to GT | Not needed — poses are given in world frame |
| Uses Wild-SLAM Mocap dataset format | Replace with competition `meta.json` / `lidar.npz` format |
| Evaluates on static NVS cameras | Render single target camera at t_mid |
| Iterates over 10 Wild-SLAM scenes | Iterate over competition samples |
| Uses fixed intrinsics from dataset | Read per-camera intrinsics from `meta.json` |

---

## Expected Challenges and Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Dynamic objects (cars, pedestrians) in input views | Use uncertainty MLP or simple temporal consistency mask |
| Large scene scale (60 m radius) | Voxel-downsample LiDAR; adjust `scene_extent` in config |
| Limited views (12 images) for GS optimization | LiDAR initialization; lower SH degree; regularization |
| Time budget per sample (inference) | `fast_mode: True`; cap to 1000 iters; half-resolution optimization |
| Sky/background not covered by LiDAR | Add background Gaussians at large radius or use SkyBox |
| Distortion in input images | `cv2.undistort` before passing to the GS renderer |

---

## Coordinate System Notes

- **`meta.json` poses** are `camera-to-world` (c2w, OpenCV convention: x→right, y→down, z→forward).
- **GS renderer** expects `world-to-camera` rotation `R` and translation `T` (i.e., the columns of the w2c matrix).
- **LiDAR xyz** is already in the same world frame as the poses — no additional transform needed.
- Inversion: `w2c = np.linalg.inv(c2w)`.

---

## Expected Performance Targets

| Metric | Target | Approach |
|--------|--------|----------|
| PSNR | 25–28 dB | LiDAR init + 12-view optimization |
| Inference time | 30–60s per sample | 1000 iters, half-res optimization |
| GPU memory | <12 GB | ≤500k initial Gaussians, SH degree 0 |

---

## Competition Config (`configs/NVS/competition.yaml`)

Key parameters:

```yaml
mapping:
  Training:
    ssim_loss: True
    alpha: 1.0              # no depth loss (LiDAR provides geometry)
    mapping_itr_num: 1000
    gaussian_update_every: 100
    gaussian_th: 0.7
    gaussian_extent: 1.0
    size_threshold: 20
  opt_params:
    position_lr_init: 0.00016
    position_lr_final: 0.0000016
    feature_lr: 0.0025
    opacity_lr: 0.05
    scaling_lr: 0.001
    rotation_lr: 0.001
    lambda_dssim: 0.2
    densification_interval: 100
    densify_until_iter: 700
    opacity_reset_interval: 300
  model_params:
    sh_degree: 0
  pipeline_params:
    convert_SHs_python: False
    compute_cov3D_python: False
```

---

## Dynamic Object Handling (Enhancement Strategies)

Two strategies, from simple to advanced:

1. **Temporal consistency mask** (simple): For each Gaussian, if it was visible in a t0 camera but not t1 (or vice versa), mark it as potentially dynamic and down-weight its opacity loss.

2. **Uncertainty MLP** (reuse existing): WildGS-SLAM's uncertainty model (`src/utils/dyn_uncertainty/`) predicts per-pixel dynamic probability from DINOv2 features. Apply it to input images and mask out high-uncertainty regions before computing photometric loss.

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Sky not covered by LiDAR | Add background plane Gaussians at z=100m behind all cameras |
| Dynamic objects in input views | Temporal consistency check + uncertainty MLP |
| Distortion model mismatch | Apply `cv2.undistort()` to all inputs before processing |
| 12 views insufficient for convergence | LiDAR provides geometry; optimization only learns appearance |
| Large scenes (highway) | Increase voxel size; filter points by frustum of target camera |

---

## Summary

`eval_map/nvs_wild_slam.py` is essentially a **reference implementation for the rendering and evaluation half** of the competition pipeline. It proves that the existing codebase can render novel views from arbitrary poses using a pre-built Gaussian model and compute standard image quality metrics.

The main gap is that it assumes a Gaussian model already exists from a SLAM run, whereas the competition requires **building** that model per-sample from 12 input views + LiDAR. The rendering/evaluation code, however, can be adopted nearly verbatim.

The implementation requires:
1. A new data loader for the competition format
2. LiDAR-based Gaussian initialization
3. A pose-supervised optimization loop (no tracker)
4. Target view rendering (reusing existing code)
5. A batch inference entry-point script
