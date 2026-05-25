"""
NVS Competition Pipeline — Phase 1: Minimal Working Pipeline

Given a sample directory with:
  - meta.json (poses, intrinsics, target camera)
  - input/t0/*.jpg, input/t1/*.jpg (12 input images)
  - input/lidar.npz (dense point cloud)

Pipeline:
  1. Load data from meta.json
  2. Initialize Gaussians from LiDAR point cloud
  3. Optimize Gaussians using 12 input views (pose-supervised)
  4. Render the target view
  5. Save the predicted image
"""

import os
import sys
import json
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from munch import munchify

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import (
    getProjectionMatrix2,
    BasicPointCloud,
)
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov
from thirdparty.gaussian_splatting.utils.sh_utils import RGB2SH
from thirdparty.gaussian_splatting.utils.general_utils import inverse_sigmoid
from src.utils.camera_utils import Camera


def load_config():
    """Load the NVS competition config."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "NVS", "competition.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_sample(sample_dir):
    """Load all data for a single competition sample.

    Returns:
        meta: dict from meta.json
        images: dict {(time, cam_name): np.ndarray HxWx3 RGB}
        lidar_xyz: np.ndarray (N, 3)
        lidar_intensity: np.ndarray (N,) or None
    """
    with open(os.path.join(sample_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    images = {}
    for t in ["t0", "t1"]:
        t_dir = os.path.join(sample_dir, "input", t)
        for cam_name in ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]:
            img_path = os.path.join(t_dir, f"{cam_name}.jpg")
            if os.path.exists(img_path):
                img_bgr = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                images[(t, cam_name)] = img_rgb

    lidar_path = os.path.join(sample_dir, "input", "lidar.npz")
    lidar_data = np.load(lidar_path)
    lidar_xyz = lidar_data["xyz"]
    lidar_intensity = lidar_data.get("intensity", None)

    return meta, images, lidar_xyz, lidar_intensity


def make_viewpoint(intrinsics_dict, pose_c2w, image_rgb=None, uid=0, device="cuda"):
    """Create a Camera viewpoint from intrinsics and a c2w pose.

    Args:
        intrinsics_dict: dict with fx, fy, cx, cy, width, height
        pose_c2w: 4x4 np.ndarray camera-to-world matrix
        image_rgb: optional HxWx3 uint8 numpy array (RGB)
        uid: camera index
        device: torch device

    Returns:
        Camera object ready for render()
    """
    fx = intrinsics_dict["fx"]
    fy = intrinsics_dict["fy"]
    cx = intrinsics_dict["cx"]
    cy = intrinsics_dict["cy"]
    W = intrinsics_dict["width"]
    H = intrinsics_dict["height"]

    # Build projection matrix
    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
    ).transpose(0, 1).to(device=device)

    fovx = focal2fov(fx, W)
    fovy = focal2fov(fy, H)

    # Convert image to tensor if provided
    color_tensor = None
    if image_rgb is not None:
        color_tensor = (
            torch.from_numpy(image_rgb).float().permute(2, 0, 1).to(device) / 255.0
        )

    identity = torch.eye(4, device=device)

    viewpoint = Camera(
        uid,
        color_tensor,  # gt_color
        None,          # est_depth
        identity,      # gt_T (placeholder)
        projection_matrix,
        fx, fy, cx, cy,
        fovx, fovy,
        H, W,
        features=None,
        device=device,
    )

    # Set the actual pose (world-to-camera)
    w2c = np.linalg.inv(pose_c2w)
    R = torch.tensor(w2c[:3, :3], dtype=torch.float32, device=device)
    T = torch.tensor(w2c[:3, 3], dtype=torch.float32, device=device)
    viewpoint.update_RT(R, T)

    return viewpoint


def initialize_gaussians_from_lidar(
    lidar_xyz, lidar_intensity=None, cfg=None,
    voxel_size=0.1, max_points=500000, device="cuda"
):
    """Initialize a GaussianModel from LiDAR point cloud.

    Args:
        lidar_xyz: (N, 3) numpy array of world-frame points
        lidar_intensity: (N,) optional intensity values
        cfg: config dict
        voxel_size: voxel downsampling size in meters
        max_points: maximum number of points to keep
        device: torch device

    Returns:
        GaussianModel initialized with LiDAR geometry
    """
    import open3d as o3d

    # Create Open3D point cloud for voxel downsampling
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_xyz.astype(np.float64))

    # Voxel downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd_down.points).astype(np.float32)

    # If still too many points, randomly subsample
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]

    print(f"  LiDAR init: {lidar_xyz.shape[0]} -> {len(points)} points (voxel={voxel_size}m)")

    # Initialize colors to gray (will be optimized)
    colors = np.ones((len(points), 3), dtype=np.float32) * 0.5

    # Create GaussianModel
    sh_degree = cfg["mapping"]["model_params"]["sh_degree"] if cfg else 0
    gaussians = GaussianModel(sh_degree, config=None)
    gaussians.active_sh_degree = sh_degree

    # Convert to SH
    fused_point_cloud = torch.from_numpy(points).float().to(device)
    fused_color = RGB2SH(torch.from_numpy(colors).float().to(device))

    features = torch.zeros(
        (fused_color.shape[0], 3, (sh_degree + 1) ** 2), dtype=torch.float32, device=device
    )
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    # Compute initial scales from nearest-neighbor distances
    from simple_knn._C import distCUDA2

    dist2 = torch.clamp_min(
        distCUDA2(fused_point_cloud), 0.0000001
    ) * 0.01  # small initial scale

    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

    # Identity rotations
    rots = torch.zeros((len(points), 4), device=device)
    rots[:, 0] = 1.0

    # Opacity initialized to 0.5
    opacities = inverse_sigmoid(
        0.5 * torch.ones((len(points), 1), dtype=torch.float32, device=device)
    )

    # Set parameters
    from torch import nn
    gaussians._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(
        features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
    )
    gaussians._features_rest = nn.Parameter(
        features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
    )
    gaussians._scaling = nn.Parameter(scales.requires_grad_(True))
    gaussians._rotation = nn.Parameter(rots.requires_grad_(True))
    gaussians._opacity = nn.Parameter(opacities.requires_grad_(True))
    gaussians.max_radii2D = torch.zeros((len(points),), device=device)
    gaussians.unique_kfIDs = torch.zeros((len(points),)).int()
    gaussians.n_obs = torch.zeros((len(points),)).int()

    # Setup optimizer
    gaussians.init_lr(6.0)
    opt_params = munchify(cfg["mapping"]["opt_params"])
    gaussians.training_setup(opt_params)

    return gaussians


def optimize_gaussians(gaussians, viewpoints, cfg, device="cuda"):
    """Optimize Gaussians using the input viewpoints.

    Args:
        gaussians: initialized GaussianModel
        viewpoints: list of Camera objects with gt images
        cfg: config dict
        device: torch device
    """
    pipe = munchify(cfg["mapping"]["pipeline_params"])
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    n_iters = cfg["mapping"]["Training"]["mapping_itr_num"]
    lambda_dssim = cfg["mapping"]["opt_params"]["lambda_dssim"]
    densify_until = cfg["mapping"]["opt_params"]["densify_until_iter"]
    densify_interval = cfg["mapping"]["opt_params"]["densification_interval"]
    opacity_reset_interval = cfg["mapping"]["opt_params"]["opacity_reset_interval"]
    densify_grad_threshold = cfg["mapping"]["opt_params"]["densify_grad_threshold"]

    gaussian_extent = cfg["mapping"]["Training"]["gaussian_extent"]
    size_threshold = cfg["mapping"]["Training"]["size_threshold"]
    min_opacity = cfg["mapping"]["Training"]["gaussian_th"]

    print(f"  Optimizing {gaussians.get_xyz.shape[0]} Gaussians for {n_iters} iterations...")

    for iteration in range(1, n_iters + 1):
        # Pick a random viewpoint
        viewpoint = random.choice(viewpoints)

        # Render
        render_pkg = render(viewpoint, gaussians, pipe, background)
        if render_pkg is None:
            continue

        image = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_points = render_pkg["viewspace_points"]

        # Compute loss
        gt_image = viewpoint.original_image
        l1_loss = F.l1_loss(image, gt_image)

        if cfg["mapping"]["Training"]["ssim_loss"]:
            from thirdparty.gaussian_splatting.utils.loss_utils import ssim
            ssim_loss = 1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss
        else:
            loss = l1_loss

        loss.backward()

        with torch.no_grad():
            # Update learning rate
            gaussians.update_learning_rate(iteration)

            # Densification
            if iteration < densify_until:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_points, visibility_filter)

                if iteration % densify_interval == 0:
                    gaussians.densify_and_prune(
                        densify_grad_threshold,
                        min_opacity,
                        gaussian_extent,
                        size_threshold,
                    )

                if iteration % opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 500 == 0:
            print(f"    Iter {iteration}/{n_iters}: loss={loss.item():.4f}, "
                  f"n_gaussians={gaussians.get_xyz.shape[0]}")


def render_target_view(gaussians, meta, cfg, device="cuda"):
    """Render the target view and return as uint8 RGB numpy array.

    Args:
        gaussians: optimized GaussianModel
        meta: meta.json dict
        cfg: config dict
        device: torch device

    Returns:
        np.ndarray (H, W, 3) uint8 RGB image
    """
    pipe = munchify(cfg["mapping"]["pipeline_params"])
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    target_camera_name = meta["target_camera"]
    target_intrinsics = meta["intrinsics"][target_camera_name]
    target_pose_c2w = np.array(meta["poses_c2w"]["target"][target_camera_name], dtype=np.float64)

    viewpoint = make_viewpoint(target_intrinsics, target_pose_c2w, image_rgb=None, uid=99, device=device)

    with torch.no_grad():
        render_pkg = render(viewpoint, gaussians, pipe, background)
        if render_pkg is None:
            # Return black image if no Gaussians visible
            H = target_intrinsics["height"]
            W = target_intrinsics["width"]
            return np.zeros((H, W, 3), dtype=np.uint8)

        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

    output_rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return output_rgb


def process_sample(sample_dir, output_dir, cfg, device="cuda"):
    """Process a single competition sample end-to-end.

    Args:
        sample_dir: path to sample directory
        output_dir: path to save predictions
        cfg: config dict
        device: torch device

    Returns:
        sample_id: str
    """
    # Load data
    meta, images, lidar_xyz, lidar_intensity = load_sample(sample_dir)
    sample_id = meta["sample_id"]
    print(f"Processing sample: {sample_id}")

    # Build input viewpoints
    viewpoints = []
    uid = 0
    for t in ["t0", "t1"]:
        poses = meta["poses_c2w"][t]
        for cam_name in ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]:
            if (t, cam_name) not in images:
                continue
            if cam_name not in poses:
                continue

            pose_c2w = np.array(poses[cam_name], dtype=np.float64)
            intrinsics = meta["intrinsics"][cam_name]
            img_rgb = images[(t, cam_name)]

            viewpoint = make_viewpoint(intrinsics, pose_c2w, image_rgb=img_rgb, uid=uid, device=device)
            viewpoints.append(viewpoint)
            uid += 1

    print(f"  Loaded {len(viewpoints)} input viewpoints")

    # Initialize Gaussians from LiDAR
    gaussians = initialize_gaussians_from_lidar(
        lidar_xyz, lidar_intensity, cfg=cfg,
        voxel_size=0.1, max_points=500000, device=device,
    )

    # Optimize
    optimize_gaussians(gaussians, viewpoints, cfg, device=device)

    # Render target
    output_rgb = render_target_view(gaussians, meta, cfg, device=device)

    # Save prediction
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sample_id}.jpg")
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {output_path}")

    # Cleanup GPU memory
    del gaussians, viewpoints
    torch.cuda.empty_cache()

    return sample_id
