from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_depth(depth_path: Path) -> np.ndarray:
    depth = np.load(depth_path)
    return depth.astype(np.float32)


def load_frame_metadata(frame_json: Path) -> dict:
    return json.loads(frame_json.read_text(encoding="utf-8"))


def backproject_depth(
    depth: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict,
    min_depth_m: float = 0.05,
    max_depth_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= min_depth_m)
    if max_depth_m is not None:
        valid &= z <= max_depth_m
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[ys, xs].astype(np.float32) / 255.0
    return points, colors


def camera_to_world(points_camera: np.ndarray, rotation_3x3: list[list[float]], position_world: dict) -> np.ndarray:
    if points_camera.size == 0:
        return points_camera
    rotation = np.asarray(rotation_3x3, dtype=np.float32)
    translation = np.asarray(
        [position_world["x"], position_world["y"], position_world["z"]], dtype=np.float32
    )
    # Isaac Sim stores camera basis in world coordinates where:
    # row0 = camera right, row1 = camera up, row2 = backward.
    # Our backprojection uses image coordinates with +y downward,
    # so world = t + x * right - y * up - z * backward.
    basis = np.stack([rotation[0], -rotation[1], -rotation[2]], axis=1)
    return (basis @ points_camera.T).T + translation


def world_to_camera(points_world: np.ndarray, rotation_3x3: list[list[float]], position_world: dict) -> np.ndarray:
    if points_world.size == 0:
        return points_world
    rotation = np.asarray(rotation_3x3, dtype=np.float32)
    translation = np.asarray(
        [position_world["x"], position_world["y"], position_world["z"]], dtype=np.float32
    )
    basis = np.stack([rotation[0], -rotation[1], -rotation[2]], axis=1)
    return ((points_world - translation) @ basis).astype(np.float32)


def project_world_to_image(
    points_world: np.ndarray,
    rotation_3x3: list[list[float]],
    position_world: dict,
    intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = world_to_camera(points_world, rotation_3x3, position_world)
    z = points_camera[:, 2]
    valid = z > 1e-6
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    u = fx * points_camera[:, 0] / z + cx
    v = fy * points_camera[:, 1] / z + cy
    pixels = np.stack([u, v], axis=1).astype(np.float32)
    return pixels, valid


def write_ascii_ply(path: Path, points_xyzrgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(points_xyzrgb[:, 3:] * 255.0, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points_xyzrgb)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points_xyzrgb[:, :3], rgb, strict=True):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
