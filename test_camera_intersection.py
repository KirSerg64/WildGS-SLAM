#!/usr/bin/env python3
"""
Test script for camera FOV intersection checking vs. the target camera.

Loads t0 and t1 poses for all 6 rig cameras, plus the explicit target camera
pose (poses_c2w["target"][target_camera]).  Checks each of the 12 cameras
against the target with cameras_may_intersect() and draws a BEV diagram:

  Circle  = t0 camera
  Diamond = t1 camera
  Cyan crosshair = target camera (t_target, between t0 and t1)
  Green ring around marker = FOV intersects target
  Colored cone = intersecting  |  Light gray cone = no intersection

Output: test_camera_intersection.jpg  (project root)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov, getProjectionMatrix2
from src.utils.camera_utils import Camera
from submission.utils import cameras_may_intersect

# ---------------------------------------------------------------------------
META_JSON = (
    # PROJECT_ROOT
    # / "data/train/2025-10-04_15_57_04_16_44_38_luka_1759586458800193000__000/meta.json"
    PROJECT_ROOT
    / "data/test/2025-10-01_10_56_41_13_02_27_natelio_1759313641599971000__000/meta.json"    
)
OUTPUT_JPG = PROJECT_ROOT / "test_camera_intersection.jpg"

CAMERA_NAMES = ["front", "left_fwd", "left_bwd", "right_fwd", "right_bwd", "rear"]

ABBREV = {
    "front":     "fr",
    "left_fwd":  "lf",
    "left_bwd":  "lb",
    "right_fwd": "rf",
    "right_bwd": "rb",
    "rear":      "re",
}

# BGR colours per camera
CAM_COLORS = {
    "front":     (0,   180,   0),
    "left_fwd":  (0,   130, 200),
    "left_bwd":  (0,    50, 180),
    "right_fwd": (200, 100,   0),
    "right_bwd": (180,  40,   0),
    "rear":      (140,   0, 200),
}

# Outward world-XY nudge direction for BEV label placement
LABEL_DIR = {
    "front":     np.array([ 0.0,  1.0]),
    "left_fwd":  np.array([-1.0,  0.5]),
    "left_bwd":  np.array([-1.0, -0.5]),
    "right_fwd": np.array([ 1.0,  0.5]),
    "right_bwd": np.array([ 1.0, -0.5]),
    "rear":      np.array([ 0.0, -1.0]),
}

# ---------------------------------------------------------------------------
# Camera construction
# ---------------------------------------------------------------------------

def c2w_to_w2c(c2w: np.ndarray):
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    return R, t


def build_camera(intr: dict, c2w_mat, device="cpu") -> Camera:
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    W, H   = intr["width"], intr["height"]
    fovx   = focal2fov(fx, W)
    fovy   = focal2fov(fy, H)
    proj   = getProjectionMatrix2(
        znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
    ).transpose(0, 1)
    c2w    = np.array(c2w_mat, dtype=np.float64)
    R_w2c, t_w2c = c2w_to_w2c(c2w)
    cam = Camera(
        uid=0, color=None, depth=None,
        gt_T=torch.eye(4, dtype=torch.float32),
        projection_matrix=proj,
        fx=fx, fy=fy, cx=cx, cy=cy,
        fovx=fovx, fovy=fovy,
        image_height=H, image_width=W,
        device=device,
    )
    cam.update_RT(
        torch.tensor(R_w2c, dtype=torch.float32),
        torch.tensor(t_w2c, dtype=torch.float32),
    )
    return cam


# ---------------------------------------------------------------------------
# BEV geometry helpers
# ---------------------------------------------------------------------------

def fov_boundary_dirs_xy(c2w: np.ndarray, fovx: float):
    """Left/right FOV boundary unit vectors in world XY (OpenCV +Z forward)."""
    half = fovx / 2.0
    left_cam  = np.array([-math.sin(half), 0.0, math.cos(half)])
    right_cam = np.array([ math.sin(half), 0.0, math.cos(half)])
    R = c2w[:3, :3]
    return (R @ left_cam)[:2], (R @ right_cam)[:2]


class BEVTransform:
    """World XY -> image pixel.  World +X = image right,  World +Y = image up."""
    def __init__(self, centroid, scale, h, w):
        self.centroid = centroid
        self.scale    = scale
        self.h, self.w = h, w

    def __call__(self, wx, wy):
        ix = int((wx - self.centroid[0]) * self.scale + self.w / 2)
        iy = int(self.h / 2 - (wy - self.centroid[1]) * self.scale)
        return ix, iy

    def ray_end(self, origin_xy, dir_xy, length_m):
        n = np.linalg.norm(dir_xy)
        if n < 1e-9:
            return self(*origin_xy)
        d = dir_xy / n * length_m
        return self(origin_xy[0] + d[0], origin_xy[1] + d[1])


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def draw_diamond(img, cx, cy, size, fill_color):
    pts = np.array([
        [cx, cy - size], [cx + size, cy],
        [cx, cy + size], [cx - size, cy],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], fill_color)
    cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


def put_label(img, text, cx, cy, off_px, font, fscl, fthk, text_color):
    """White-background text box centred at (cx + off_px[0], cy + off_px[1])."""
    tw, th = cv2.getTextSize(text, font, fscl, fthk)[0]
    lx = cx + off_px[0] - tw // 2
    ly = cy + off_px[1] + th // 2
    cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (255, 255, 255), -1)
    cv2.putText(img, text, (lx, ly), font, fscl, text_color, fthk, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# BEV diagram
# ---------------------------------------------------------------------------

def draw_bev(
    cams_t0: dict,       # name -> {"c2w": ndarray, "cam": Camera}
    cams_t1: dict,
    target: dict,        # {"c2w": ndarray, "cam": Camera}
    target_name: str,
    intersecting: set,   # set of ("t0"|"t1", cam_name)
    img_size: int = 1100,
    fov_len_m: float = 3.0,
) -> np.ndarray:

    img  = np.full((img_size, img_size, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    all_pos = (
        [cams_t0[n]["c2w"][:2, 3] for n in CAMERA_NAMES]
        + [cams_t1[n]["c2w"][:2, 3] for n in CAMERA_NAMES]
        + [target["c2w"][:2, 3]]
    )
    centroid = np.mean(all_pos, axis=0)
    # Fit all camera positions plus FOV cone reach inside the view
    max_dist     = max(np.linalg.norm(np.array(p) - centroid) for p in all_pos)
    view_radius_m = max_dist + fov_len_m + 1.0
    scale    = (img_size / 2 - 85) / view_radius_m
    T        = BEVTransform(centroid, scale, img_size, img_size)

    # ---- 1 m grid -----------------------------------------------------------
    for dm in np.arange(-view_radius_m, view_radius_m + 1, 1.0):
        cv2.line(img, T(centroid[0] - view_radius_m, centroid[1] + dm),
                      T(centroid[0] + view_radius_m, centroid[1] + dm), (215, 215, 215), 1)
        cv2.line(img, T(centroid[0] + dm, centroid[1] - view_radius_m),
                      T(centroid[0] + dm, centroid[1] + view_radius_m), (215, 215, 215), 1)

    # ---- t0->t1 trajectory lines per camera ---------------------------------
    for name in CAMERA_NAMES:
        p0 = cams_t0[name]["c2w"][:2, 3]
        p1 = cams_t1[name]["c2w"][:2, 3]
        col = tuple(int(c * 0.45 + 200 * 0.55) for c in CAM_COLORS[name])
        cv2.line(img, T(*p0), T(*p1), col, 1, cv2.LINE_AA)

    # ---- FOV cones (filled, blended) ----------------------------------------
    overlay = img.copy()

    for ts, cams in [("t0", cams_t0), ("t1", cams_t1)]:
        for name in CAMERA_NAMES:
            c2w = cams[name]["c2w"]
            cam = cams[name]["cam"]
            pos = c2w[:2, 3]
            cx_i, cy_i = T(*pos)
            hits = (ts, name) in intersecting

            ld, rd = fov_boundary_dirs_xy(c2w, cam.FoVx)
            pt_l = T.ray_end(pos, ld, fov_len_m)
            pt_r = T.ray_end(pos, rd, fov_len_m)

            cone_col = CAM_COLORS[name] if hits else (215, 215, 215)
            cv2.fillPoly(overlay,
                         [np.array([[cx_i, cy_i], pt_l, pt_r], dtype=np.int32)],
                         cone_col)

    # Target cone (cyan)
    tpos = target["c2w"][:2, 3]
    tcx, tcy = T(*tpos)
    tld, trd = fov_boundary_dirs_xy(target["c2w"], target["cam"].FoVx)
    pt_tl = T.ray_end(tpos, tld, fov_len_m)
    pt_tr = T.ray_end(tpos, trd, fov_len_m)
    cv2.fillPoly(overlay, [np.array([[tcx, tcy], pt_tl, pt_tr], dtype=np.int32)], (60, 220, 220))

    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # ---- FOV boundary lines and forward arrows -------------------------------
    for ts, cams in [("t0", cams_t0), ("t1", cams_t1)]:
        for name in CAMERA_NAMES:
            c2w = cams[name]["c2w"]
            cam = cams[name]["cam"]
            pos = c2w[:2, 3]
            cx_i, cy_i = T(*pos)
            hits = (ts, name) in intersecting

            ld, rd = fov_boundary_dirs_xy(c2w, cam.FoVx)
            pt_l = T.ray_end(pos, ld, fov_len_m)
            pt_r = T.ray_end(pos, rd, fov_len_m)
            pt_f = T.ray_end(pos, c2w[:2, 2], fov_len_m)

            line_col = CAM_COLORS[name] if hits else (185, 185, 185)
            cv2.line(img, (cx_i, cy_i), pt_l, line_col, 1, cv2.LINE_AA)
            cv2.line(img, (cx_i, cy_i), pt_r, line_col, 1, cv2.LINE_AA)
            cv2.arrowedLine(img, (cx_i, cy_i), pt_f, line_col, 1,
                            cv2.LINE_AA, tipLength=0.22)

    # Target boundary lines (cyan, thicker)
    pt_tf = T.ray_end(tpos, target["c2w"][:2, 2], fov_len_m)
    cv2.line(img, (tcx, tcy), pt_tl, (0, 155, 155), 2, cv2.LINE_AA)
    cv2.line(img, (tcx, tcy), pt_tr, (0, 155, 155), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, (tcx, tcy), pt_tf, (0, 140, 140), 2,
                    cv2.LINE_AA, tipLength=0.18)

    # ---- camera markers (circle = t0, diamond = t1) -------------------------
    R0   = 9    # t0 circle radius
    R1   = 9    # t1 diamond half-size
    RTGT = 13
    RGLOW = 5

    for ts, cams in [("t0", cams_t0), ("t1", cams_t1)]:
        for name in CAMERA_NAMES:
            pos    = cams[name]["c2w"][:2, 3]
            cx_i, cy_i = T(*pos)
            col    = CAM_COLORS[name]
            hits   = (ts, name) in intersecting

            if hits:
                glow_r = R0 + RGLOW if ts == "t0" else R1 + RGLOW + 1
                cv2.circle(img, (cx_i, cy_i), glow_r, (0, 200, 70), 2, cv2.LINE_AA)

            if ts == "t0":
                cv2.circle(img, (cx_i, cy_i), R0, col, -1, cv2.LINE_AA)
                cv2.circle(img, (cx_i, cy_i), R0, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                draw_diamond(img, cx_i, cy_i, R1, col)

    # Target marker (cyan crosshair)
    cv2.circle(img, (tcx, tcy), RTGT + 6, (0, 175, 175), 2, cv2.LINE_AA)
    cv2.circle(img, (tcx, tcy), RTGT, (0, 210, 210), -1, cv2.LINE_AA)
    cv2.circle(img, (tcx, tcy), RTGT, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, (tcx - RTGT - 5, tcy), (tcx + RTGT + 5, tcy), (0, 0, 0), 1)
    cv2.line(img, (tcx, tcy - RTGT - 5), (tcx, tcy + RTGT + 5), (0, 0, 0), 1)

    # ---- labels -------------------------------------------------------------
    label_px = max(22, int(scale * 0.35))

    for ts, cams in [("t0", cams_t0), ("t1", cams_t1)]:
        for name in CAMERA_NAMES:
            pos    = cams[name]["c2w"][:2, 3]
            cx_i, cy_i = T(*pos)
            hits   = (ts, name) in intersecting
            text   = f"{ABBREV[name]}.{ts}"
            d      = LABEL_DIR[name]
            # BEV flips world Y, so negate Y component of world nudge
            off    = (int(d[0] * label_px), int(-d[1] * label_px))
            tcol   = (0, 110, 30) if hits else (110, 110, 110)
            put_label(img, text, cx_i, cy_i, off, font, 0.38, 1, tcol)

    put_label(img, f"TARGET.{ABBREV[target_name]}",
              tcx, tcy, (0, -(RTGT + 14)), font, 0.42, 1, (0, 100, 100))

    # ---- top-left legend ----------------------------------------------------
    cv2.putText(img, "Camera BEV - FOV vs Target",
                (14, 26), font, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(img, f"Target: {target_name} at t_target  (cyan crosshair)",
                (14, 46), font, 0.40, (0, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(img, "Circle=t0  Diamond=t1  |  Green ring = intersects target",
                (14, 62), font, 0.38, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(img, "Colored cone = intersects  |  Gray cone = no overlap",
                (14, 78), font, 0.38, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(img, "Line = t0->t1 trajectory per camera  |  1 m grid",
                (14, 93), font, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

    # ---- compass (bottom-right) ---------------------------------------------
    ox, oy = img_size - 75, img_size - 70
    al = 32
    cv2.arrowedLine(img, (ox, oy), (ox + al, oy),   (0, 0, 160), 2, tipLength=0.30)
    cv2.arrowedLine(img, (ox, oy), (ox, oy - al),   (0, 130, 0), 2, tipLength=0.30)
    cv2.putText(img, "+X", (ox + al + 4, oy + 5), font, 0.38, (0, 0, 160), 1)
    cv2.putText(img, "+Y", (ox - 4, oy - al - 4),  font, 0.38, (0, 130, 0), 1)

    # ---- result table (bottom-left) -----------------------------------------
    row_h  = 14
    col1_w = 50
    colw   = 50
    n_rows = len(CAMERA_NAMES) + 1
    box_h  = n_rows * row_h + 10
    box_y  = img_size - box_h - 6
    box_x2 = 10 + col1_w + colw * 2

    cv2.rectangle(img, (6, box_y), (box_x2, img_size - 4), (255, 255, 255), -1)
    cv2.rectangle(img, (6, box_y), (box_x2, img_size - 4), (180, 180, 180), 1)

    y0 = box_y + row_h
    cv2.putText(img, "cam", (10,                  y0), font, 0.34, (30, 30, 30), 1)
    cv2.putText(img, "t0",  (10 + col1_w,         y0), font, 0.34, (30, 30, 30), 1)
    cv2.putText(img, "t1",  (10 + col1_w + colw,  y0), font, 0.34, (30, 30, 30), 1)
    cv2.line(img, (6, y0 + 3), (box_x2, y0 + 3), (200, 200, 200), 1)

    for k, name in enumerate(CAMERA_NAMES):
        ry     = box_y + (k + 2) * row_h + 3
        t0_hit = ("t0", name) in intersecting
        t1_hit = ("t1", name) in intersecting
        cv2.putText(img, ABBREV[name], (10, ry), font, 0.33, (50, 50, 50), 1)
        cv2.putText(img, "YES" if t0_hit else "NO",
                    (10 + col1_w, ry), font, 0.33,
                    (0, 130, 0) if t0_hit else (150, 0, 0), 1)
        cv2.putText(img, "YES" if t1_hit else "NO",
                    (10 + col1_w + colw, ry), font, 0.33,
                    (0, 130, 0) if t1_hit else (150, 0, 0), 1)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Camera FOV intersection BEV visualizer")
    parser.add_argument(
        "meta_json",
        nargs="?",
        default=str(META_JSON),
        help="Path to meta.json (default: train sample)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JPG path (default: next to meta.json)",
    )
    args = parser.parse_args()

    meta_path  = Path(args.meta_json)
    output_jpg = Path(args.out) if args.out else meta_path.parent / "bev_intersection.jpg"

    print(f"Reading {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    target_name = meta["target_camera"]
    intrinsics  = meta["intrinsics"]

    def load_cameras(poses: dict) -> dict:
        return {
            name: {
                "c2w": np.array(poses[name], dtype=np.float64),
                "cam": build_camera(intrinsics[name], poses[name]),
            }
            for name in CAMERA_NAMES
        }

    cams_t0 = load_cameras(meta["poses_c2w"]["t0"])
    cams_t1 = load_cameras(meta["poses_c2w"]["t1"])

    # Explicit target camera pose at t_target
    target = {
        "c2w": np.array(meta["poses_c2w"]["target"][target_name], dtype=np.float64),
        "cam": build_camera(intrinsics[target_name],
                            meta["poses_c2w"]["target"][target_name]),
    }

    # Intersection test: each of the 12 cameras vs. target
    print(f"\nFOV intersection vs. TARGET '{target_name}' (t_target):")
    print(f"  {'camera':<12}  {'t0':>6}  {'t1':>6}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}")

    intersecting: set = set()
    for name in CAMERA_NAMES:
        t0_hit = cameras_may_intersect(cams_t0[name]["cam"], target["cam"])
        t1_hit = cameras_may_intersect(cams_t1[name]["cam"], target["cam"])
        if t0_hit:
            intersecting.add(("t0", name))
        if t1_hit:
            intersecting.add(("t1", name))
        print(f"  {name:<12}  {'YES' if t0_hit else 'NO':>6}  {'YES' if t1_hit else 'NO':>6}")

    bev = draw_bev(cams_t0, cams_t1, target, target_name, intersecting)
    cv2.imwrite(str(output_jpg), bev)
    print(f"\nSaved -> {output_jpg}")


if __name__ == "__main__":
    main()
