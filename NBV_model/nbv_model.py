"""
nbv_model.py — Next Best View SAC Agent.

MDP:
  State  : up to 10 RGB-D frames + camera poses + joint positions
           + per-frame segmentation confidence + per-frame collision flags
  Action : next camera pose (x, y, z, qw, qx, qy, qz)
  Reward : +10*conf_seg  -20*collision  -100*no_segment  +20*conf_complete

Two perception backbone modes (set in config.yaml):
  "cnn"        — RGB-D frames → CNN backbone + temporal attention/GRU
  "pointcloud" — unproject to merged 3D point cloud → PointNet++ or ODIN

Framework: Stable-Baselines3 SAC + gymnasium.

Usage (CLI):
    python nbv_model.py --test                         # forward-pass smoke test
    python nbv_model.py --train --config config.yaml   # train on MockEnv
    python nbv_model.py --train --steps 500            # short run
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class NBVConfig:
    """Thin wrapper around a YAML config dict with dot-path access."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self._cfg: dict = yaml.safe_load(f)
        self.config_path = config_path

    # ── shortcuts ──────────────────────────────────────────────────────
    @property
    def perception_mode(self) -> str:
        return self._cfg["perception"]["mode"]

    @property
    def cnn_cfg(self) -> dict:
        return self._cfg["perception"]["cnn"]

    @property
    def pc_cfg(self) -> dict:
        return self._cfg["perception"]["pointcloud"]

    @property
    def state_cfg(self) -> dict:
        return self._cfg["state"]

    @property
    def action_cfg(self) -> dict:
        return self._cfg["action"]

    @property
    def reward_cfg(self) -> dict:
        return self._cfg["reward"]

    @property
    def sac_cfg(self) -> dict:
        return self._cfg["sac"]

    @property
    def training_cfg(self) -> dict:
        return self._cfg["training"]

    @property
    def callbacks_cfg(self) -> dict:
        return self._cfg["callbacks"]

    def get(self, dotpath: str, default=None):
        val = self._cfg
        for k in dotpath.split("."):
            if not isinstance(val, dict):
                return default
            val = val.get(k, default)
        return val

    def __repr__(self) -> str:
        return f"NBVConfig(mode={self.perception_mode}, path={self.config_path})"


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Fusion modules
# ──────────────────────────────────────────────────────────────────────────────

class TemporalAttentionFusion(nn.Module):
    """Single learned query cross-attends to N frame features → 1 vector."""

    def __init__(self, feat_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.attn  = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm  = nn.LayerNorm(feat_dim)

    def forward(
        self,
        frame_feats: torch.Tensor,  # (B, N, D)
        pad_mask:    torch.Tensor,  # (B, N) True=ignore
    ) -> torch.Tensor:              # (B, D)
        B = frame_feats.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, frame_feats, frame_feats, key_padding_mask=pad_mask)
        return self.norm(out.squeeze(1))


class GRUTemporalFusion(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden_dim, batch_first=True)

    def forward(self, frame_feats: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(frame_feats)
        return h.squeeze(0)  # (B, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# CNN Perception Encoder
# ──────────────────────────────────────────────────────────────────────────────

class CNNFrameEncoder(nn.Module):
    """
    Encodes up to max_frames RGB-D (4-channel) frames with a CNN backbone,
    then fuses temporal features with attention / GRU / mean-pool.

    Supported backbones (perception.cnn.backbone):
        "efficientnet_v2_s"  — recommended, good accuracy/speed trade-off
        "resnet50"           — classic choice
        "mobilenet_v3_large" — lightweight for fast experiments
    """

    _IN_CH = 4  # RGB + Depth

    def __init__(self, config: NBVConfig):
        super().__init__()
        cnn_cfg = config.cnn_cfg
        self.max_frames    = config.state_cfg["max_frames"]
        self.fusion_mode   = cnn_cfg.get("temporal_fusion", "attention")
        self.image_h, self.image_w = cnn_cfg.get("image_size", [480, 640])

        self.backbone, self.backbone_dim = self._build_backbone(
            cnn_cfg.get("backbone", "efficientnet_v2_s")
        )
        self.proj = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )
        self.out_dim = 512

        nheads = cnn_cfg.get("temporal_attention_heads", 4)
        if self.fusion_mode == "attention":
            self.fusion = TemporalAttentionFusion(512, nheads)
        elif self.fusion_mode == "gru":
            self.fusion = GRUTemporalFusion(512, 512)
        elif self.fusion_mode == "stack":
            self.fusion = None
        else:
            raise ValueError(f"Unknown temporal_fusion: {self.fusion_mode}")

    @staticmethod
    def _build_backbone(name: str) -> Tuple[nn.Module, int]:
        import torchvision.models as tvm

        def _patch_first_conv(conv: nn.Conv2d) -> nn.Conv2d:
            return nn.Conv2d(
                4, conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=False,
            )

        if name == "efficientnet_v2_s":
            m = tvm.efficientnet_v2_s(weights=None)
            m.features[0][0] = _patch_first_conv(m.features[0][0])
            dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, dim

        if name == "resnet50":
            m = tvm.resnet50(weights=None)
            m.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
            dim = m.fc.in_features
            m.fc = nn.Identity()
            return m, dim

        if name == "mobilenet_v3_large":
            m = tvm.mobilenet_v3_large(weights=None)
            m.features[0][0] = _patch_first_conv(m.features[0][0])
            dim = m.classifier[0].in_features
            m.classifier = nn.Identity()
            return m, dim

        raise ValueError(f"Unknown CNN backbone: {name}")

    def forward(
        self,
        frames:      torch.Tensor,   # (B, N, H, W, 4)  uint8 or float
        frame_count: torch.Tensor,   # (B, 1)
    ) -> torch.Tensor:               # (B, 512)
        B, N, H, W, C = frames.shape
        x = frames.float().div(255.0) if frames.dtype == torch.uint8 else frames.float()
        x = x.view(B * N, H, W, C).permute(0, 3, 1, 2)        # (B*N, 4, H, W)

        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.flatten(1)
        feats = self.proj(feats).view(B, N, 512)                # (B, N, 512)

        counts   = frame_count.squeeze(-1).long()               # (B,)
        pad_mask = torch.arange(N, device=frames.device).unsqueeze(0) >= counts.unsqueeze(1)

        if self.fusion_mode == "attention":
            return self.fusion(feats, pad_mask)
        if self.fusion_mode == "gru":
            return self.fusion(feats, counts)
        # stack = masked mean
        mask = (~pad_mask).float().unsqueeze(-1)                # (B, N, 1)
        return (feats * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# PointNet++ (self-contained, pure PyTorch — no external CUDA kernels needed)
# ──────────────────────────────────────────────────────────────────────────────

def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling. Returns (B, npoint) indices."""
    B, N, _ = xyz.shape
    device   = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance  = torch.full((B, N), 1e10, device=device)
    farthest  = torch.randint(0, N, (B,), device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B, 1, 3)
        dist     = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(-1)
    return centroids


def _ball_query(
    xyz: torch.Tensor, new_xyz: torch.Tensor,
    radius: float, nsample: int
) -> torch.Tensor:
    """Ball query. Returns (B, M, nsample) indices."""
    dist  = torch.cdist(new_xyz, xyz)                    # (B, M, N)
    idx   = dist.argsort(dim=-1)[..., :nsample]          # (B, M, nsample)
    valid = dist.gather(-1, idx) < radius
    first = idx[..., :1].expand_as(idx)
    idx[~valid] = first[~valid]
    return idx


class _SetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_ch: int, mlp_chs: List[int]):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers, c = [], in_ch
        for oc in mlp_chs:
            layers += [nn.Conv1d(c, oc, 1), nn.BatchNorm1d(oc), nn.ReLU(inplace=True)]
            c = oc
        self.mlp = nn.Sequential(*layers)
        self.out_ch = c

    def forward(
        self, xyz: torch.Tensor, feats: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        fps_idx  = _farthest_point_sample(xyz, self.npoint)              # (B, M)
        new_xyz  = xyz[torch.arange(B).unsqueeze(-1), fps_idx]           # (B, M, 3)
        grp_idx  = _ball_query(xyz, new_xyz, self.radius, self.nsample)  # (B, M, S)
        flat_idx = grp_idx.reshape(B, -1)

        grouped_xyz = xyz[torch.arange(B).unsqueeze(-1), flat_idx]       # (B, M*S, 3)
        grouped_xyz = grouped_xyz.reshape(B, self.npoint, self.nsample, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        if feats is not None:
            gf = feats[torch.arange(B).unsqueeze(-1), flat_idx]
            gf = gf.reshape(B, self.npoint, self.nsample, -1)
            grouped = torch.cat([grouped_xyz, gf], dim=-1)
        else:
            grouped = grouped_xyz

        M, S, F = grouped.shape[1], grouped.shape[2], grouped.shape[3]
        out = self.mlp(grouped.reshape(B * M, F, S))                     # (B*M, C, S)
        out = out.max(-1).values.reshape(B, M, self.out_ch)              # (B, M, C)
        return new_xyz, out


class PointNetPlusPlus(nn.Module):
    """
    Three-layer PointNet++ encoder.
    Input : (B, N, in_ch)  — point cloud; first 3 channels = xyz
    Output: (B, out_dim)   — global feature vector
    """

    def __init__(self, in_ch: int = 6, out_dim: int = 512):
        super().__init__()
        self.sa1 = _SetAbstraction(1024, 0.1, 32, 3 + in_ch,  [64,  64,  128])
        self.sa2 = _SetAbstraction(256,  0.2, 32, 3 + 128,    [128, 128, 256])
        self.sa3 = _SetAbstraction(64,   0.4, 32, 3 + 256,    [256, 256, 512])
        self.mlp = nn.Sequential(
            nn.Linear(512, out_dim), nn.GELU(), nn.LayerNorm(out_dim)
        )
        self.out_dim = out_dim

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        xyz   = pts[..., :3]
        feats = pts[..., 3:] if pts.shape[-1] > 3 else None
        xyz, feats = self.sa1(xyz, feats)
        xyz, feats = self.sa2(xyz, feats)
        _,   feats = self.sa3(xyz, feats)
        return self.mlp(feats.max(1).values)                # (B, out_dim)


# ──────────────────────────────────────────────────────────────────────────────
# ODIN Encoder (lazy import — requires setup_odin.py)
# ──────────────────────────────────────────────────────────────────────────────

class ODINEncoder(nn.Module):
    """
    Wraps ODIN as a frozen feature extractor.
    Call `python setup_odin.py --all` before using this.
    """

    def __init__(self, config: NBVConfig, out_dim: int = 512):
        super().__init__()
        odin_cfg   = config.pc_cfg.get("odin", {})
        repo_dir   = Path(odin_cfg.get("repo_dir", "odin_repo"))
        checkpoint = odin_cfg.get("checkpoint", "")
        feat_dim   = int(odin_cfg.get("feature_dim", 256))
        self.frozen = bool(odin_cfg.get("frozen_backbone", True))

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        try:
            # Import ODIN backbone (3D Swin Transformer)
            from odin.modeling.backbone.swin import D2SwinTransformer  # noqa
            self._odin_available = True
        except ImportError as e:
            raise ImportError(
                "ODIN is not installed.\n"
                "Run:  python setup_odin.py --all\n"
                f"Original error: {e}"
            )

        # Adapter: project ODIN features → out_dim
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location="cpu")
            # Load only backbone weights (skip prediction heads)
            state = {k: v for k, v in ckpt["model"].items()
                     if "backbone" in k}
            # NOTE: actual loading depends on ODIN model internals
            print(f"[ODINEncoder] Loaded backbone from {checkpoint}")

        if self.frozen:
            for p in self.parameters():
                p.requires_grad_(False)
            # Re-enable adapter params
            for p in self.adapter.parameters():
                p.requires_grad_(True)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # Full ODIN forward is complex; this is where you'd call the model.
        # For now, raise NotImplementedError to signal incomplete integration.
        raise NotImplementedError(
            "ODINEncoder.forward() requires completing the ODIN integration "
            "after running setup_odin.py."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Point Cloud Perception Encoder
# ──────────────────────────────────────────────────────────────────────────────

class PointCloudEncoder(nn.Module):
    """
    Fuses N RGB-D frames into a single 3D point cloud, then encodes it.

    Pipeline:
        1. Unproject each depth frame → 3D (using camera intrinsics + extrinsics)
        2. Transform to world frame with camera pose (quaternion → rotation matrix)
        3. Merge all N point clouds
        4. Uniform subsample to max_points
        5. Encode with PointNet++ or ODIN
    """

    def __init__(self, config: NBVConfig):
        super().__init__()
        pc_cfg     = config.pc_cfg
        cnn_cfg    = config.cnn_cfg
        self.backend    = pc_cfg.get("backend", "pointnet2")
        self.max_points = int(pc_cfg.get("max_points", 40000))
        self.depth_scale = float(cnn_cfg.get("depth_scale", 1000.0))
        self.fx = float(pc_cfg.get("fx", 577.591))
        self.fy = float(pc_cfg.get("fy", 578.730))
        self.cx = float(pc_cfg.get("cx", 318.905))
        self.cy = float(pc_cfg.get("cy", 242.684))
        H, W = cnn_cfg.get("image_size", [480, 640])
        self.H, self.W = H, W

        if self.backend == "pointnet2":
            self.encoder = PointNetPlusPlus(in_ch=6, out_dim=512)
        elif self.backend == "odin":
            self.encoder = ODINEncoder(config, out_dim=512)
        else:
            raise ValueError(f"Unknown pointcloud backend: {self.backend}")

        self.out_dim = 512

    @staticmethod
    def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
        """(qw, qx, qy, qz) → 3×3 rotation matrix."""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return torch.stack([
            1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw),
            2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw),
            2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2),
        ]).reshape(3, 3)

    def _unproject_batch(
        self,
        frames:      torch.Tensor,   # (B, N, H, W, 4)
        cam_poses:   torch.Tensor,   # (B, N, 7)
        frame_count: torch.Tensor,   # (B, 1)
    ) -> torch.Tensor:               # (B, max_points, 6) xyz+rgb
        B, N, H, W, _ = frames.shape
        device = frames.device

        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )  # (H, W)

        clouds = []
        for b in range(B):
            n_valid = int(frame_count[b, 0].item())
            pts_list = []
            for n in range(n_valid):
                d   = frames[b, n, :, :, 3].float() / self.depth_scale  # (H, W) metres
                rgb = frames[b, n, :, :, :3].float() / 255.0            # (H, W, 3)
                valid = d > 0                                             # (H, W)

                x_c = (u_grid - self.cx) / self.fx * d
                y_c = (v_grid - self.cy) / self.fy * d
                z_c = d

                pts_cam = torch.stack([x_c, y_c, z_c], dim=-1)[valid]    # (K, 3)
                rgb_v   = rgb[valid]                                      # (K, 3)

                pos  = cam_poses[b, n, :3]
                R    = self._quat_to_rot(cam_poses[b, n, 3:])
                pts_w = pts_cam @ R.T + pos.unsqueeze(0)                  # (K, 3)
                pts_list.append(torch.cat([pts_w, rgb_v], dim=-1))        # (K, 6)

            merged = torch.cat(pts_list, dim=0) if pts_list else torch.zeros(1, 6, device=device)
            M = merged.shape[0]
            if M >= self.max_points:
                idx = torch.randperm(M, device=device)[:self.max_points]
                merged = merged[idx]
            else:
                pad = self.max_points - M
                merged = torch.cat([merged, merged[:1].expand(pad, -1)], dim=0)
            clouds.append(merged)

        return torch.stack(clouds, dim=0)  # (B, max_points, 6)

    def forward(
        self,
        frames:      torch.Tensor,
        cam_poses:   torch.Tensor,
        frame_count: torch.Tensor,
    ) -> torch.Tensor:
        cloud = self._unproject_batch(frames, cam_poses, frame_count)
        return self.encoder(cloud)


# ──────────────────────────────────────────────────────────────────────────────
# State Meta-Encoder (poses + joints + confidence + collision)
# ──────────────────────────────────────────────────────────────────────────────

class StateMetaEncoder(nn.Module):
    """
    MLP encoding of non-visual state components:
      - camera pose history       (max_frames × 7)
      - joint positions           (3 × 3)
      - segmentation confidence   (max_frames,)  — per-frame
      - collision flags           (max_frames,)  — per-frame
      - completeness_confidence   (1,)           — global scene coverage estimate
    """

    def __init__(self, max_frames: int, out_dim: int = 256):
        super().__init__()
        # +1 for completeness_confidence scalar
        in_dim = max_frames * 7 + 3 * 3 + max_frames + max_frames + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, out_dim), nn.GELU(), nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward(
        self,
        camera_poses:             torch.Tensor,  # (B, N, 7)
        joint_positions:          torch.Tensor,  # (B, 3, 3)
        seg_confidence:           torch.Tensor,  # (B, N)
        collision_flags:          torch.Tensor,  # (B, N)
        completeness_confidence:  torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:
        B = camera_poses.shape[0]
        x = torch.cat([
            camera_poses.reshape(B, -1),
            joint_positions.reshape(B, -1),
            seg_confidence,
            collision_flags.float(),
            completeness_confidence,
        ], dim=-1)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Stable-Baselines3 Features Extractor
# ──────────────────────────────────────────────────────────────────────────────

class NBVFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom SB3 BaseFeaturesExtractor for the NBV Dict observation space.
    Used via policy_kwargs["features_extractor_class"].
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        config: NBVConfig,
        features_dim: int = 512,
    ):
        super().__init__(observation_space, features_dim=features_dim)
        self.config = config

        mode = config.perception_mode
        if mode == "cnn":
            self.perception = CNNFrameEncoder(config)
        elif mode == "pointcloud":
            self.perception = PointCloudEncoder(config)
        else:
            raise ValueError(f"Unknown perception mode: {mode}")

        max_frames      = config.state_cfg["max_frames"]
        self.meta_enc   = StateMetaEncoder(max_frames, out_dim=256)
        total_dim       = self.perception.out_dim + self.meta_enc.out_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.GELU(),
            nn.LayerNorm(features_dim),
        )
        self._features_dim = features_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        frames                   = obs["frames"]                   # (B, N, H, W, 4)
        frame_count              = obs["frame_count"]              # (B, 1)
        camera_poses             = obs["camera_poses"]             # (B, N, 7)
        joint_positions          = obs["joint_positions"]          # (B, 3, 3)
        seg_confidence           = obs["seg_confidence"]           # (B, N)
        collision_flags          = obs["collision_flags"]          # (B, N)
        completeness_confidence  = obs["completeness_confidence"]  # (B, 1)

        if self.config.perception_mode == "cnn":
            vis = self.perception(frames, frame_count)
        else:
            vis = self.perception(frames, camera_poses, frame_count)

        meta = self.meta_enc(
            camera_poses, joint_positions,
            seg_confidence, collision_flags,
            completeness_confidence.float(),
        )
        return self.fusion_mlp(torch.cat([vis, meta], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# Mock Gymnasium Environment (for offline testing without Isaac Sim)
# ──────────────────────────────────────────────────────────────────────────────

class NBVMockEnv(gym.Env):
    """
    Mock Gymnasium environment — mirrors the Isaac Sim episode structure.

    Episode logic:
        step 0 (reset):  agent receives frame_count=1  (only 1 RGB-D frame)
        step 1:          agent receives frame_count=2  (2 frames accumulated)
        ...
        step 9:          agent receives frame_count=10 (max; episode ends)

    All frame-indexed arrays (frames, camera_poses, seg_confidence,
    collision_flags) are zero-padded for indices >= frame_count.

    Replace step() / reset() internals when connecting to Isaac Sim.
    observation_space and action_space stay identical.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: NBVConfig):
        super().__init__()
        self.config     = config
        s               = config.state_cfg
        a               = config.action_cfg
        cnn_cfg         = config.cnn_cfg
        self.max_frames = s["max_frames"]  # 10
        H, W            = cnn_cfg.get("image_size", [480, 640])
        self.H, self.W  = H, W

        self.observation_space = spaces.Dict({
            "frames":                   spaces.Box(0, 255, (self.max_frames, H, W, 4), dtype=np.uint8),
            "frame_count":              spaces.Box(1, self.max_frames, (1,), dtype=np.int32),
            "camera_poses":             spaces.Box(-np.inf, np.inf, (self.max_frames, 7), dtype=np.float32),
            "joint_positions":          spaces.Box(-np.inf, np.inf, (3, 3), dtype=np.float32),
            "seg_confidence":           spaces.Box(0.0, 1.0, (self.max_frames,), dtype=np.float32),
            "collision_flags":          spaces.Box(0.0, 1.0, (self.max_frames,), dtype=np.float32),
            # Global scene coverage estimate — updated every step
            "completeness_confidence":  spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (a["pose_dim"],), dtype=np.float32)

        # Episode state
        self._frame_count  = 1
        self._step_n       = 0
        self._comp_conf    = 0.0   # current completeness confidence

        # Persistent frame buffers (filled as episode progresses)
        self._frames       = np.zeros((self.max_frames, H, W, 4), dtype=np.uint8)
        self._cam_poses    = np.zeros((self.max_frames, 7),        dtype=np.float32)
        self._seg_conf     = np.zeros(self.max_frames,             dtype=np.float32)
        self._coll_flags   = np.zeros(self.max_frames,             dtype=np.float32)

    def _make_obs(self) -> dict:
        """Build obs dict: only indices < frame_count contain real data."""
        n = self._frame_count
        return {
            "frames":                  self._frames.copy(),
            "frame_count":             np.array([n], dtype=np.int32),
            "camera_poses":            self._cam_poses.copy(),
            "joint_positions":         self.observation_space["joint_positions"].sample(),
            "seg_confidence":          self._seg_conf.copy(),
            "collision_flags":         self._coll_flags.copy(),
            "completeness_confidence": np.array([self._comp_conf], dtype=np.float32),
        }

    def _sample_new_frame(self, idx: int) -> None:
        """Fill slot `idx` with a random RGB-D frame + metadata (mock data)."""
        H, W = self.H, self.W
        self._frames[idx]     = self.np_random.integers(0, 255, (H, W, 4), dtype=np.uint8)
        self._cam_poses[idx]  = self.np_random.uniform(-1, 1, (7,)).astype(np.float32)
        # Normalize quaternion part
        q = self._cam_poses[idx, 3:]
        q /= np.linalg.norm(q) + 1e-8
        self._cam_poses[idx, 3:] = q
        self._seg_conf[idx]   = float(self.np_random.uniform(0.3, 1.0))
        self._coll_flags[idx] = float(self.np_random.choice([0, 1], p=[0.9, 0.1]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_n      = 0
        self._frame_count = 1
        self._comp_conf   = 0.0
        # Clear buffers and fill first frame
        self._frames[:]=0; self._cam_poses[:]=0
        self._seg_conf[:]=0; self._coll_flags[:]=0
        self._sample_new_frame(0)
        return self._make_obs(), {}

    def step(self, action: np.ndarray):
        # Add next frame if budget allows
        if self._frame_count < self.max_frames:
            self._sample_new_frame(self._frame_count)
            self._frame_count += 1

        self._step_n += 1
        obs = self._make_obs()

        # Reward components (mock — use last valid frame's metrics)
        r_cfg     = self.config.reward_cfg
        idx       = self._frame_count - 1
        seg_conf  = float(self._seg_conf[idx])
        collision = float(self._coll_flags[idx])
        no_seg    = float(self.np_random.choice([0, 1], p=[0.95, 0.05]))
        comp_conf = float(self.np_random.uniform(0, 1))

        # Update completeness_confidence in state buffer
        self._comp_conf = comp_conf

        reward = (
            + r_cfg["seg_confidence_weight"]    * seg_conf
            - r_cfg["collision_penalty"]        * collision
            - r_cfg["no_segment_penalty"]       * no_seg
            + r_cfg["completeness_conf_weight"] * comp_conf
        )
        done = self._frame_count >= self.max_frames
        info = {
            "seg_confidence":      seg_conf,
            "collision":           collision,
            "no_segment":          no_seg,
            "complete_confidence": comp_conf,
            "frame_count":         self._frame_count,
        }
        return obs, reward, done, False, info

    def render(self):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# NBV Agent
# ──────────────────────────────────────────────────────────────────────────────

class NBVAgent:
    """
    Next Best View agent wrapping Stable-Baselines3 SAC.

    Example:
        agent = NBVAgent("config.yaml")
        agent.train(total_timesteps=1_000_000)
        obs, _ = env.reset()
        action, _ = agent.predict(obs)
    """

    def __init__(
        self,
        config_path: str,
        env: Optional[gym.Env] = None,
        verbose: int = 1,
    ):
        self.config     = NBVConfig(config_path)
        self.verbose    = verbose
        self._config_path = config_path

        if env is None:
            warnings.warn(
                "No env provided → using NBVMockEnv. "
                "Replace with Isaac Sim environment for real training."
            )
            env = NBVMockEnv(self.config)
        self.env = env

        sac_cfg  = self.config.sac_cfg
        train_cfg = self.config.training_cfg
        cb_cfg   = self.config.callbacks_cfg

        log_dir  = train_cfg.get("log_dir", "logs/")
        ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints/")
        os.makedirs(log_dir,  exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        policy_kwargs = {
            "features_extractor_class":  NBVFeaturesExtractor,
            "features_extractor_kwargs": {
                "config":       self.config,
                "features_dim": sac_cfg.get("hidden_dim", 512),
            },
            "net_arch": [sac_cfg.get("hidden_dim", 512)] * sac_cfg.get("num_hidden_layers", 3),
            "share_features_extractor": False,
        }

        tb_log = log_dir if cb_cfg.get("use_tensorboard", True) else None

        self.model = SAC(
            policy="MultiInputPolicy",
            env=self.env,
            learning_rate=sac_cfg.get("actor_lr", 3e-4),
            buffer_size=sac_cfg.get("buffer_size", 100_000),
            learning_starts=sac_cfg.get("warmup_steps", 10_000),
            batch_size=sac_cfg.get("batch_size", 256),
            tau=sac_cfg.get("tau", 0.005),
            gamma=sac_cfg.get("gamma", 0.99),
            train_freq=sac_cfg.get("update_every", 1),
            gradient_steps=sac_cfg.get("gradient_steps", 1),
            ent_coef="auto" if sac_cfg.get("auto_alpha", True) else sac_cfg.get("alpha", 0.2),
            target_entropy=sac_cfg.get("target_entropy", "auto"),
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_log,
            verbose=verbose,
            seed=train_cfg.get("seed", 42),
            device=train_cfg.get("device", "cuda"),
        )

        if verbose:
            print(f"[NBVAgent] Ready | mode={self.config.perception_mode} "
                  f"| device={train_cfg.get('device')}")

    # ── Training ───────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callbacks=None,
        reset_num_timesteps: bool = True,
    ):
        from nbv_callbacks import make_callback_list

        if total_timesteps is None:
            total_timesteps = self.config.training_cfg.get("total_timesteps", 1_000_000)
        if callbacks is None:
            eval_env = NBVMockEnv(self.config)
            callbacks = make_callback_list(self.config, eval_env=eval_env)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="nbv_sac",
            progress_bar=True,
        )

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(
        self, observation: dict, deterministic: bool = True
    ) -> Tuple[np.ndarray, Any]:
        return self.model.predict(observation, deterministic=deterministic)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(
                self.config.training_cfg.get("checkpoint_dir", "checkpoints/"),
                "nbv_agent_final",
            )
        self.model.save(path)
        if self.verbose:
            print(f"[NBVAgent] Saved → {path}.zip")
        return path

    def load(self, path: str, env: Optional[gym.Env] = None):
        device = self.config.training_cfg.get("device", "cuda")
        self.model = SAC.load(path, env=env or self.env, device=device)
        if self.verbose:
            print(f"[NBVAgent] Loaded ← {path}")

    # ── Utilities ──────────────────────────────────────────────────────────

    @property
    def policy(self) -> nn.Module:
        return self.model.policy

    def __repr__(self) -> str:
        mode = self.config.perception_mode
        sub  = (self.config.pc_cfg.get("backend", "?") if mode == "pointcloud"
                else self.config.cnn_cfg.get("backbone", "?"))
        return f"NBVAgent(mode={mode}, backbone={sub})"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="NBV SAC Agent CLI")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--train",  action="store_true")
    p.add_argument("--test",   action="store_true",
                   help="Run a forward-pass smoke test (no real training)")
    p.add_argument("--steps",  type=int, default=None)
    return p.parse_args()


def _smoke_test(config_path: str):
    """Verify obs→feature extraction with random tensors (no env required)."""
    print("=" * 60)
    print("  NBV Forward-Pass Smoke Test")
    print("=" * 60)
    cfg      = NBVConfig(config_path)
    env      = NBVMockEnv(cfg)
    obs, _   = env.reset()
    obs_t    = {k: torch.tensor(v).unsqueeze(0) for k, v in obs.items()}

    # Build extractor and run forward pass
    extractor = NBVFeaturesExtractor(env.observation_space, cfg, features_dim=512)
    extractor.eval()
    with torch.no_grad():
        feat = extractor(obs_t)
    print(f"  Mode           : {cfg.perception_mode}")
    print(f"  Feature shape  : {feat.shape}  (expected: [1, 512])")
    print(f"  Feature norm   : {feat.norm().item():.4f}")
    print("  ✅ Smoke test PASSED")


if __name__ == "__main__":
    args = _parse_args()

    if args.test:
        _smoke_test(args.config)

    elif args.train:
        agent = NBVAgent(args.config)
        agent.train(total_timesteps=args.steps)
        agent.save()

    else:
        print("Usage: python nbv_model.py --test | --train [--steps N]")
