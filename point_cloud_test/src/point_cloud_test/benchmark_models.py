from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


NUM_CLASSES = 4


class OdinPointNet(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 128, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        point_features = self.point_mlp(points)
        global_feature = point_features.max(dim=1, keepdim=True).values.expand_as(point_features)
        return self.head(torch.cat([point_features, global_feature], dim=-1))


class PointTransformerV3Mini(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 96, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.embed = nn.Linear(in_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.encoder(self.embed(points))
        return self.classifier(features)


class OneFormer3DMini(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 96, num_queries: int = 16, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.embed = nn.Linear(in_channels, hidden_dim)
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )
        self.point_proj = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(self.embed(points))
        queries = self.query_embed.expand(points.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        context = decoded.mean(dim=1, keepdim=True).expand(memory.shape[0], memory.shape[1], memory.shape[2])
        return self.point_proj(torch.cat([memory, context], dim=-1))


@dataclass(slots=True)
class OpenYolo3DModel:
    point_classifier: LogisticRegression
    cluster_classifier: RandomForestClassifier

    @classmethod
    def fit(cls, train_samples: list[tuple[np.ndarray, np.ndarray]]) -> "OpenYolo3DModel":
        x_points = []
        y_points = []
        x_clusters = []
        y_clusters = []

        for points, labels in train_samples:
            point_features = _normalize_points(points)
            x_points.append(point_features)
            y_points.append((labels > 0).astype(np.int32))
            for class_id in [1, 2, 3]:
                mask = labels == class_id
                if mask.sum() < 8:
                    continue
                x_clusters.append(_cluster_features(point_features[mask]))
                y_clusters.append(class_id)

        point_classifier = LogisticRegression(max_iter=300, class_weight="balanced")
        point_classifier.fit(np.concatenate(x_points), np.concatenate(y_points))

        cluster_classifier = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced")
        if x_clusters:
            cluster_classifier.fit(np.stack(x_clusters), np.asarray(y_clusters))
        else:
            cluster_classifier.fit(np.zeros((1, 10), dtype=np.float32), np.asarray([1]))
        return cls(point_classifier=point_classifier, cluster_classifier=cluster_classifier)

    def predict(self, points: np.ndarray) -> np.ndarray:
        features = _normalize_points(points)
        foreground = self.point_classifier.predict(features).astype(np.int32)
        predictions = np.zeros(len(points), dtype=np.int64)
        if foreground.sum() == 0:
            return predictions

        clustering = DBSCAN(eps=0.08, min_samples=12).fit(features[foreground == 1, :3])
        cluster_labels = clustering.labels_
        fg_indices = np.where(foreground == 1)[0]
        for cluster_id in np.unique(cluster_labels):
            if cluster_id < 0:
                continue
            point_idx = fg_indices[cluster_labels == cluster_id]
            class_id = int(self.cluster_classifier.predict([_cluster_features(features[point_idx])])[0])
            predictions[point_idx] = class_id
        return predictions


def _normalize_points(points_xyzrgb: np.ndarray) -> np.ndarray:
    xyz = points_xyzrgb[:, :3].astype(np.float32)
    rgb = points_xyzrgb[:, 3:].astype(np.float32)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = max(np.linalg.norm(xyz, axis=1).max(), 1e-6)
    xyz = xyz / scale
    return np.concatenate([xyz, rgb], axis=1)


def _cluster_features(points_xyzrgb: np.ndarray) -> np.ndarray:
    xyz = points_xyzrgb[:, :3]
    rgb = points_xyzrgb[:, 3:]
    return np.concatenate(
        [
            xyz.mean(axis=0),
            xyz.std(axis=0),
            rgb.mean(axis=0),
            np.asarray([len(points_xyzrgb)], dtype=np.float32),
        ]
    ).astype(np.float32)


def save_torch_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_torch_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
