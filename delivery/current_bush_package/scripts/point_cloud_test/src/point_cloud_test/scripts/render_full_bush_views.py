from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from point_cloud_test.geometry import backproject_depth, camera_to_world, load_depth, load_frame_metadata, load_rgb
from point_cloud_test.point_cloud_ops import filter_points_to_bush_volume, filter_points_to_centered_ellipsoid, voxel_downsample


def _set_equal_aspect(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _style_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    except AttributeError:
        pass


def build_full_bush_cloud(
    orbit_root: Path,
    frame_stride: int = 2,
    pixel_stride: int = 8,
    voxel_size_m: float = 0.008,
) -> np.ndarray:
    coord_dir = orbit_root / "coords"
    frame_paths = sorted(coord_dir.glob("frame_*.json"))[::frame_stride]
    if not frame_paths:
        raise FileNotFoundError(f"No frame metadata found under {coord_dir}")

    merged_clouds: list[np.ndarray] = []
    for frame_path in frame_paths:
        frame_meta = load_frame_metadata(frame_path)
        rgb = load_rgb(orbit_root / "images" / frame_meta["rgb_file"])
        depth = load_depth(orbit_root / "depth_est" / f"distance_to_image_plane_{frame_meta['frame_id']:04d}.npy")

        mask = np.zeros(depth.shape, dtype=bool)
        mask[::pixel_stride, ::pixel_stride] = True
        points_camera, colors = backproject_depth(
            depth,
            rgb,
            mask,
            frame_meta["intrinsics_px"],
            min_depth_m=0.2,
            max_depth_m=6.0,
        )
        if len(points_camera) == 0:
            continue

        points_world = camera_to_world(
            points_camera,
            frame_meta["camera_gt"]["rotation_matrix_3x3"],
            frame_meta["camera_gt"]["position_world_m"],
        )
        points_world, colors = filter_points_to_bush_volume(
            points_world,
            colors,
            frame_meta["bush_world"],
            xy_radius_m=0.65,
            z_margin_down_m=0.15,
            z_margin_up_m=0.65,
        )
        if len(points_world) == 0:
            continue

        bush_center = np.array(
            [
                frame_meta["bush_world"]["x"],
                frame_meta["bush_world"]["y"],
                frame_meta["bush_world"]["z"],
            ],
            dtype=np.float32,
        )
        points_world, colors = filter_points_to_centered_ellipsoid(
            points_world,
            colors,
            bush_center,
            radii_m=(0.58, 0.58, 0.42),
        )
        if len(points_world) == 0:
            continue

        merged_clouds.append(np.concatenate([points_world, colors], axis=1).astype(np.float32))

    if not merged_clouds:
        raise RuntimeError("No points survived bush filtering")

    return voxel_downsample(np.concatenate(merged_clouds, axis=0), voxel_size_m=voxel_size_m)


def render_full_bush_views(
    orbit_root: Path,
    output_path: Path,
    frame_stride: int = 2,
    pixel_stride: int = 8,
) -> Path:
    cloud = build_full_bush_cloud(
        orbit_root=orbit_root,
        frame_stride=frame_stride,
        pixel_stride=pixel_stride,
    )
    xyz = cloud[:, :3]
    rgb = np.clip(cloud[:, 3:], 0.0, 1.0)

    fig = plt.figure(figsize=(15, 5))
    views = [
        ("Front", 18, -90),
        ("Side", 16, 0),
        ("Top-Oblique", 62, -55),
    ]
    for subplot_idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, subplot_idx, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1.7, alpha=0.92, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        _set_equal_aspect(ax, xyz)
        _style_axes(ax)
        ax.set_title(title)

    fig.suptitle("Full bush point cloud from orbital RGB-D frames", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    orbit_root = Path(r"D:\MyProjects\Skoltech\Perception_In_Robotics\FP\strawberry_orbit_data_new\data")
    output_path = Path(r"D:\MyProjects\Robostrawberry\point_cloud_test\artifacts\proof\full_bush_3_views.png")
    result = render_full_bush_views(orbit_root=orbit_root, output_path=output_path)
    print(f"Saved {result}")


if __name__ == "__main__":
    main()
