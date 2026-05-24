# Exploring `eval_map/nvs_wild_slam.py` for the NVS Task

## What the Script Does

`eval_map/nvs_wild_slam.py` is a **Novel View Synthesis evaluation pipeline** for WildGS-SLAM. It:

1. **Loads a pre-trained Gaussian Splatting model** (`final_gs.ply`) produced by a full SLAM run.
2. **Aligns the estimated trajectory to ground-truth** using scale recovery and an SE3 transformation (`T_we_wg`).
3. **Renders novel views** from static camera poses (from `nvs/groundtruth.txt`) that were **not** in the training trajectory.
4. **Computes metrics** (PSNR, SSIM, LPIPS) against ground-truth images at those novel poses.
5. **Saves visualizations** (rendered vs. GT side-by-side).

---

## How It Is Directly Useful for the Competition NVS Task

### 1. Rendering Pipeline Template (Immediately Reusable)

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

This is **exactly** what the competition pipeline needs for the final rendering step — construct a `Camera` from the target pose/intrinsics in `meta.json`, load Gaussians, render.

---

### 2. `get_temp_viewpoint()` in `eval_map/utils.py` (Directly Reusable)

This utility (lines 45–78 of `utils.py`) shows how to construct a `Camera` object from **arbitrary intrinsics** (fx, fy, cx, cy, W, H) without needing an actual image or depth — just a pose and projection matrix:

```python
def get_temp_viewpoint(static_cam_cfg, full_resol=False, exp_cfg=None):
    # Scales intrinsics to output resolution if needed
    # Builds projection matrix and fovx/fovy
    # Returns a Camera object ready for render()
```

This is exactly what's needed to render the competition's target view, where you only have the target pose and intrinsics from `meta.json` but no GT image.

---

### 3. Coordinate Transform Logic (Conceptually Reusable)

Lines 52–66 demonstrate how to align estimated SLAM coordinates to ground-truth world coordinates:

- **Scale recovery** from `metrics_full_traj.txt`
- **Per-frame `T_we_wg` computation** (estimated-world → GT-world)
- **Rotation averaging** via `scipy.spatial.transform.Rotation.mean()`

```python
T_we_wg = np.eye(4)
T_we_wg[:3, :3] = R.from_matrix(np.array(Rot)).mean().as_matrix()
T_we_wg[:3, 3]  = np.mean(Trans, axis=0)
```

For the competition, poses are given directly in a common world frame, so this alignment step is not needed — but the pattern is essential if you first run WildGS-SLAM on the input views and then render from the estimated map.

---

### 4. Metric Computation (Directly Reusable for Validation)

Lines 106–108 show PSNR/SSIM/LPIPS computation using the same functions from `thirdparty/gaussian_splatting/`:

```python
psnr_score  = psnr(image[mask].unsqueeze(0),  gt_image[mask].unsqueeze(0)).item()
ssim_score  = ssim(image.unsqueeze(0),         gt_image.unsqueeze(0)).item()
lpips_score = cal_lpips(image.unsqueeze(0),    gt_image.unsqueeze(0)).item()
```

For local validation on the competition's train split, this exact code evaluates predicted images against GT targets without any additional tooling.

---

### 5. Distortion Handling (Important Pattern)

Line 98 applies `cv2.undistort()` to GT images before metric computation:

```python
input_rgb = cv2.undistort(input_rgb, K, np.array(this_intrinsic['coeffs']))
```

The competition data also has per-camera distortion coefficients in `meta.json`. This pattern confirms that both GT images and input views should be undistorted before being passed to the GS renderer.

---

## How to Adapt It for the Competition

| Current Script | Needed Adaptation |
|---|---|
| Loads `final_gs.ply` from SLAM output | Initialize Gaussians from LiDAR + optimize per sample |
| Aligns SLAM trajectory to GT | Not needed — poses are given in world frame |
| Uses Wild-SLAM Mocap dataset format | Replace with competition `meta.json` / `lidar.npz` format |
| Evaluates on static NVS cameras | Render single target camera at t_mid |
| Iterates over 10 Wild-SLAM scenes | Iterate over competition samples |
| Uses fixed intrinsics from dataset | Read per-camera intrinsics from `meta.json` |

---

## Key Reusable Components (Copy Directly)

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

## Summary

`eval_map/nvs_wild_slam.py` is essentially a **reference implementation for the rendering and evaluation half** of the competition pipeline. It proves that the existing codebase can render novel views from arbitrary poses using a pre-built Gaussian model and compute standard image quality metrics.

The main gap is that it assumes a Gaussian model already exists from a SLAM run, whereas the competition requires **building** that model per-sample from 12 input views + LiDAR. The rendering/evaluation code, however, can be adopted nearly verbatim.
