# Plan: Running WildGS-SLAM Without Gaussian Splatting

**Goal**: Strip WildGS-SLAM to a pure **tracking + sparse 3D point cloud** system (no Gaussian Splatting), similar in spirit to DROID-SLAM, but keeping WildGS's monocular metric depth regularization and uncertainty-aware keyframe filtering.

---

## Is It Feasible?

**Yes.** The tracking subsystem is already independent of Gaussian Splatting. The tracker, frontend, backend, factor graph, and bundle adjustment are entirely self-contained and produce camera poses + dense disparity maps without any rendering. The mapper runs as a separate process and only communicates through a one-way pipe (tracker → mapper). The `DepthVideo` shared state does not depend on gaussians at all.

The mapper, however, is currently integrated with GaussianModel in two important ways:
1. **Covisibility management**: `render()` is called to count `n_touched` pixels per keyframe, used to decide which keyframes to include in the local optimization window.
2. **Uncertainty feedback loop**: The uncertainty MLP is trained using rendered-vs-real image discrepancies during mapping; the resulting uncertainty mask feeds back into tracking via `video.update_all_uncertainty_mask()`.

Both can be replaced without losing the tracking functionality.

---

## Step-by-Step Plan

### Step 1: Remove the Mapper Process

- Delete or replace `src/mapper.py` with a lightweight stub (`StubMapper`) that:
  - Receives pipe messages from the tracker.
  - Sends `"continue"` immediately without any GS operations.
  - Does not instantiate `GaussianModel`.
- In `slam.py::run()`, instantiate `StubMapper` instead of `Mapper`.
- In `slam.py::terminate()`, skip calls to `mapper.final_refine`, `mapper.save_all_kf_figs`, `mapper.gaussians.save_ply`.

### Step 2: Decouple Uncertainty from the Mapper

Two options:

**Option A — Simple (disable uncertainty)**:
- Set `uncertainty_params.activate: false` in the config.
- The system falls back to pure geometry-based tracking.
- The `valid_depth_mask` and monocular depth regularization (`BA_with_scale_shift`) still work correctly.
- No changes to `depth_video.py` or `factor_graph.py` needed.

**Option B — Retain uncertainty without GS**:
- Train the uncertainty MLP using photometric error from **direct frame warping** instead of GS rendering:
  - Use `DepthVideo.reproject(ii, jj)` (already implemented) to warp frame `j` onto frame `i`.
  - Compare warped image with real frame `i` as a photometric residual.
  - Use this residual as the supervision signal for the MLP (heteroscedastic loss).
- This is classical direct SLAM photometric loss — the geometric infrastructure is already in place.
- Requires adding a `warp_image()` function using `pops.projective_transform` and bilinear sampling.

Recommendation: Start with Option A to get a working baseline, then add Option B if uncertainty is desired.

### Step 3: Build a Sparse Point Cloud from the Depth Video

`DepthVideo` already stores all necessary data:
- `disps_up`: upsampled inverse depth maps per keyframe (`[buffer, H, W]`)
- `poses`: SE3 keyframe poses (`[buffer, 7]` as quaternions)
- `valid_depth_mask`: mask of reliable depth pixels (`[buffer, H, W]`)
- `intrinsics`: camera intrinsics per keyframe

Add a `PointCloudExporter` class (or standalone function) in `src/utils/` that:
1. For each keyframe `k` up to `video.counter.value`:
   - Converts `disps_up[k]` → metric depth: `depth = 1.0 / disps_up[k]`
   - Applies `valid_depth_mask[k]` to select reliable pixels.
   - Unprojects pixels to 3D using `pops.projective_transform` or manual unprojection with intrinsics.
   - Transforms to world frame using `poses[k]`.
   - Optionally colors points using `video.images[k]`.
2. Concatenates all keyframe point clouds.
3. Saves as `.ply` using standard PLY libraries (e.g., `open3d` or manual binary write).

Replace `mapper.gaussians.save_ply(...)` in `slam.py::terminate()` with the point cloud exporter call.

### Step 4: Adapt `slam.py::terminate()`

Keep:
- Final global BA: `self.backend()` — no changes needed.
- Trajectory filling: `self.traj_filler` — no changes needed.
- Evaluation: `kf_traj_eval` / `full_traj_eval` — no changes needed.
- Video saving: `self.video.save_video(...)` — no changes needed.

Remove:
- `self.mapper.final_refine(...)` — GS-specific.
- `self.mapper.save_all_kf_figs(...)` — GS-specific.
- `self.mapper.gaussians.save_ply(...)` — replace with point cloud exporter.
- GUI process spawning (or replace with a simple point cloud viewer).

### Step 5: Remove GS-Specific Dependencies

From `setup.py` / `requirements.txt`:
- Remove `diff-gaussian-rasterization` (CUDA extension).
- Remove `simple-knn` (CUDA extension used by GaussianModel).

Keep:
- `droid_backends` (essential for BA and correlation sampling).
- `lietorch` (SE3 pose arithmetic).
- `torch_scatter` (BA linear system assembly).
- `thirdparty/depth_anything_v2` (monocular metric depth prior).
- DINOv2 (if using uncertainty — Option B above).

`thirdparty/gaussian_splatting/` can be left in place but its imports removed from the active code paths.

### Step 6: Simplify the Covisibility Window (Optional)

The current mapper uses rendered visibility (`n_touched`) to select which keyframes to include in the local optimization window. Without GS, replace with a simpler strategy:

- **Temporal window**: always include the N most recent keyframes.
- **Distance-based selection**: additionally include K far-away keyframes using `video.distance()` (already available) to ensure good loop coverage.

This logic can live in the `StubMapper` or directly in the tracker's online BA trigger.

---

## What Stays Unchanged

The following files require **zero modifications**:

| File | Reason |
|---|---|
| `src/motion_filter.py` | Pure feature extraction + keyframe selection |
| `src/frontend.py` | Pure local BA |
| `src/backend.py` | Pure global BA + loop closure |
| `src/factor_graph.py` | Pure factor graph management |
| `src/depth_video.py` | Shared state store; GS-independent |
| `src/geom/ba.py` | Bundle adjustment math |
| `src/geom/projective_ops.py` | Projection math |
| `src/modules/droid_net/` | DroidNet feature extraction + update |
| `src/utils/mono_priors/` | Metric depth + DINOv2 extraction |
| `src/trajectory_filler.py` | Non-keyframe pose interpolation |
| `src/utils/eval_traj.py` | ATE evaluation |

---

## Summary of Files to Modify / Create

| Action | File |
|---|---|
| Replace | `src/mapper.py` → `StubMapper` |
| Modify | `src/slam.py` → use `StubMapper`, update `terminate()` |
| Create | `src/utils/point_cloud_exporter.py` |
| Modify (optional) | `setup.py` — remove GS CUDA extensions |
| Config change | Set `uncertainty_params.activate: false` (Option A) |

**Estimated effort**: Medium (~300–400 lines of new/modified code). The tracking stack is already modular and independent of GS. The main work is: writing the `StubMapper` (~50 lines), implementing the point cloud exporter (~100 lines), and adapting `slam.py::terminate()` (~50 lines of removals/replacements).
