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
3. ~~Applies distortion correction~~ → **Not needed** (PINHOLE model, no distortion).
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
| Sky/background not covered by LiDAR | Add background Gaussians at large radius or use SkyBox |
| Target at arbitrary time between t0/t1 | Interpolate dynamic objects; static scene dominates |
| 3.5M LiDAR points too many for GPU | Voxel downsample to ~200k–500k points |

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
| PSNR | 25–30 dB | LiDAR init + 12-view optimization, more iterations |
| Inference time | 2–5 min per sample (no constraint) | 3000–5000 iters, full-res optimization |
| GPU memory | <12 GB | Voxel-downsample to ≤500k Gaussians, SH degree 0 |

---

## Competition Config (`configs/NVS/competition.yaml`)

Key parameters:

```yaml
mapping:
  Training:
    ssim_loss: True
    alpha: 1.0              # no depth loss (LiDAR provides geometry)
    mapping_itr_num: 3000
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
| 12 views insufficient for convergence | LiDAR provides geometry; optimization only learns appearance |
| Large scenes (highway) | Increase voxel size; filter points by frustum of target camera |
| Target at non-midpoint time | Use timestamp ratio `(t_target - t0) / (t1 - t0)` for motion interpolation |

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

---

## Clarified Assumptions (Answers to Open Questions)

| # | Question | Answer |
|---|----------|--------|
| 1 | Camera distortion model? | **No distortion** — all cameras use PINHOLE model with empty `distortion_coeffs`. No `cv2.undistort()` needed. |
| 2 | Target time relative to t0/t1? | **Any time between t0 and t1** (not necessarily the midpoint). The exact timestamp is in `meta.json` → `timestamps_ns.target`. |
| 3 | Time constraints? | **No strict time constraints** — just feasible execution time across the full dataset. Can run more optimization iterations. |
| 4 | LiDAR format? | **Dense raw point cloud** — `lidar.npz` contains `xyz` array with ~3.5M points (aggregated from ~21 sweeps in a 3× extended window around [t0, t1]). |
| 5 | Ground-truth availability? | **Available for train dataset** at `data/train/<sample_id>/target/<camera_name>.jpg`. GT pose is between cameras at the interpolated time. |

### Implications for Implementation

- **Simplification**: Remove all distortion handling code (`cv2.undistort` calls) — images can be used directly.
- **More iterations**: No time budget means we can run 2000–5000 optimization iterations for better quality.
- **Dense initialization**: 3.5M LiDAR points provides extremely strong geometry; voxel downsample to ~200k–500k for GPU memory.
- **Temporal interpolation**: Target time varies per sample — must handle arbitrary t ∈ [t0, t1], not just midpoint.
- **Validation**: Can compute PSNR locally using GT images from the train split.

---

## Actual Data Format (from `data/train/` example)

### Directory Structure
```
data/train/<sample_id>/
├── meta.json           # all metadata (poses, intrinsics, timestamps)
├── input/
│   ├── lidar.npz       # dense point cloud (~3.5M points, xyz in world frame)
│   ├── t0/             # 6 camera images at time t0
│   │   ├── front.jpg
│   │   ├── left_fwd.jpg
│   │   ├── left_bwd.jpg
│   │   ├── right_fwd.jpg
│   │   ├── right_bwd.jpg
│   │   └── rear.jpg
│   └── t1/             # 6 camera images at time t1
│       ├── front.jpg
│       ├── ...
│       └── rear.jpg
└── target/             # ground-truth (train only)
    └── <target_camera>.jpg   # e.g., left_fwd.jpg
```

### `meta.json` Key Fields
```json
{
  "sample_id": "...",
  "delta_s": 1.0,                    // time delta between t0 and t1 in seconds
  "target_camera": "left_fwd",       // which camera to render
  "frame_convention": "poses are camera-to-world (OpenCV: x-right, y-down, z-forward)",
  "lidar_info": {
    "n_points": 3548884,             // ~3.5M dense points
    "n_sweeps": 21                   // aggregated from 21 LiDAR sweeps
  },
  "timestamps_ns": {
    "t0": 1759586461631681000,
    "t1": 1759586462631688000,
    "target": 1759586462131660000    // target time (between t0 and t1)
  },
  "intrinsics": {
    "<camera_name>": {
      "fx": ..., "fy": ..., "cx": ..., "cy": ...,
      "width": 1024, "height": 540-548,
      "distortion_model": "PINHOLE",
      "distortion_coeffs": []        // always empty — no distortion
    }
  },
  "poses_c2w": {
    "t0": { "<camera_name>": [[4x4 matrix]] },
    "t1": { "<camera_name>": [[4x4 matrix]] },
    "target": { "<target_camera>": [[4x4 matrix]] }
  }
}
```

### Key Observations from Real Data
- **6 cameras**: front, left_fwd, left_bwd, right_fwd, right_bwd, rear
- **Resolution**: ~1024×540–548 (varies by camera)
- **World frame**: SDG world_3d, derived from car_frame FLU (forward-left-up)
- **LiDAR coverage**: 3× extended window (one delta before t0, one after t1)
- **Target pose**: Only one camera in `poses_c2w.target` (the `target_camera` specified)
