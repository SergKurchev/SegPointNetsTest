from __future__ import annotations

import numpy as np


def filter_points_to_bush_volume(
    points_world: np.ndarray,
    colors: np.ndarray,
    bush_world: dict,
    xy_radius_m: float = 0.6,
    z_margin_down_m: float = 0.12,
    z_margin_up_m: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    if points_world.size == 0:
        return points_world, colors
    bush_center = np.array([bush_world["x"], bush_world["y"], bush_world["z"]], dtype=np.float32)
    xy_dist = np.linalg.norm(points_world[:, :2] - bush_center[None, :2], axis=1)
    z_ok = (
        (points_world[:, 2] >= bush_center[2] - z_margin_down_m)
        & (points_world[:, 2] <= bush_center[2] + z_margin_up_m)
    )
    keep = (xy_dist <= xy_radius_m) & z_ok
    return points_world[keep], colors[keep]


def filter_points_to_centered_ellipsoid(
    points_world: np.ndarray,
    colors: np.ndarray,
    center_world: np.ndarray,
    radii_m: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if points_world.size == 0:
        return points_world, colors
    center_world = np.asarray(center_world, dtype=np.float32)
    radii = np.maximum(np.asarray(radii_m, dtype=np.float32), 1e-6)
    normalized = (points_world - center_world[None, :]) / radii[None, :]
    keep = np.sum(normalized * normalized, axis=1) <= 1.0
    return points_world[keep], colors[keep]


def filter_points_to_sphere(
    points_world: np.ndarray,
    colors: np.ndarray,
    center_world: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if points_world.size == 0:
        return points_world, colors
    distance = np.linalg.norm(points_world - center_world[None, :], axis=1)
    keep = distance <= radius_m
    return points_world[keep], colors[keep]


def keep_largest_connected_component(
    points_world: np.ndarray,
    colors: np.ndarray,
    radius_m: float = 0.035,
    min_component_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points_world) <= min_component_size:
        return points_world, colors

    remaining = set(range(len(points_world)))
    best_component: list[int] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        queue = [seed]
        while queue:
            current = queue.pop()
            if not remaining:
                continue
            indices = np.fromiter(remaining, dtype=np.int32)
            distances = np.linalg.norm(points_world[indices] - points_world[current], axis=1)
            neighbors = indices[distances <= radius_m]
            for neighbor in neighbors.tolist():
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        if len(component) > len(best_component):
            best_component = component

    if len(best_component) < min_component_size:
        return points_world, colors
    best_idx = np.array(sorted(best_component), dtype=np.int32)
    return points_world[best_idx], colors[best_idx]


def voxel_downsample(points_xyzrgb: np.ndarray, voxel_size_m: float = 0.0075) -> np.ndarray:
    if len(points_xyzrgb) == 0:
        return points_xyzrgb
    voxel_keys = np.floor(points_xyzrgb[:, :3] / voxel_size_m).astype(np.int32)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    output = np.zeros((len(unique_keys), points_xyzrgb.shape[1]), dtype=np.float32)
    counts = np.bincount(inverse)
    for dim in range(points_xyzrgb.shape[1]):
        output[:, dim] = np.bincount(inverse, weights=points_xyzrgb[:, dim]) / np.maximum(counts, 1)
    return output


def trim_points_by_radius(
    points_xyzrgb: np.ndarray,
    center_world: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    if len(points_xyzrgb) == 0:
        return points_xyzrgb
    center_world = np.asarray(center_world, dtype=np.float32)
    distances = np.linalg.norm(points_xyzrgb[:, :3] - center_world[None, :], axis=1)
    return points_xyzrgb[distances <= radius_m]
