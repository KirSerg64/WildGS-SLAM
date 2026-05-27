import torch

from src.utils.camera_utils import Camera

def get_optical_axis_world(cam: Camera) -> torch.Tensor:
    # In OpenGL convention, camera looks along -Z in camera space.
    # R is W2C rotation, so the camera forward in world space is -R^T @ [0,0,1]
    # = negative 3rd column of R^T = negative 3rd row of R
    return -cam.R[2, :]  # shape (3,)

def cameras_may_intersect(cam_a: Camera, cam_b: Camera) -> bool:
    axis_a = get_optical_axis_world(cam_a)
    axis_b = get_optical_axis_world(cam_b)

    cos_angle = torch.clamp(
        (axis_a @ axis_b) / (axis_a.norm() * axis_b.norm()), -1.0, 1.0
    )
    angle_between = torch.acos(cos_angle)  # radians

    half_fov_a = max(cam_a.FoVx, cam_a.FoVy) / 2
    half_fov_b = max(cam_b.FoVx, cam_b.FoVy) / 2

    return angle_between.item() < (half_fov_a + half_fov_b)