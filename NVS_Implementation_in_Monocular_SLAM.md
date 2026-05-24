# Implementing Novel View Synthesis in Monocular SLAM

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
| Reuse  | `eval_map/utils.py` — `get_temp_viewpoint()` (already fits) |
| Reuse  | `thirdparty/gaussian_splatting/gaussian_renderer/__init__.py` — `render()` |
| Reuse  | `thirdparty/gaussian_splatting/scene/gaussian_model.py` — `GaussianModel` |

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
