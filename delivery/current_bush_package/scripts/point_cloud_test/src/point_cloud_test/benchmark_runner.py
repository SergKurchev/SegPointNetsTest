from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .benchmark_data import (
    PointCloudSegmentationDataset,
    collate_batch,
    create_split,
    load_sample,
    load_split,
    save_split,
)
from .benchmark_models import (
    NUM_CLASSES,
    OdinPointNet,
    OneFormer3DMini,
    OpenYolo3DModel,
    PointTransformerV3Mini,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from .evaluation import compute_metrics, save_summary


@dataclass(slots=True)
class BenchmarkConfig:
    dataset_root: Path
    output_root: Path
    points_per_sample: int = 2048
    batch_size: int = 4
    epochs: int = 12
    learning_rate: float = 1e-3
    device: str = "cuda"


def prepare_dataset(dataset_root: Path) -> Path:
    split = create_split(dataset_root)
    return save_split(dataset_root, split)


def run_benchmark(config: BenchmarkConfig, models: list[str] | None = None) -> dict[str, dict]:
    if models is None:
        models = ["odin", "pointtransformerv3", "oneformer3d", "open-yolo-3d"]
    config.output_root.mkdir(parents=True, exist_ok=True)
    split = load_split(config.dataset_root)
    results: dict[str, dict] = {}

    for model_name in models:
        if model_name == "open-yolo-3d":
            result = _train_eval_openyolo(model_name, config, split)
        else:
            result = _train_eval_torch(model_name, config, split)
        results[model_name] = result

    summary_payload = {
        name: {
            "overall_accuracy": result["metrics"].overall_accuracy,
            "macro_iou": result["metrics"].macro_iou,
            "macro_f1": result["metrics"].macro_f1,
            "fruit_macro_iou": _fruit_macro(result["metrics"].per_class_iou),
            "fruit_macro_f1": _fruit_macro(result["metrics"].per_class_f1),
            "per_class_iou": result["metrics"].per_class_iou,
            "per_class_f1": result["metrics"].per_class_f1,
            "train_seconds": result["train_seconds"],
            "inference_seconds": result["inference_seconds"],
        }
        for name, result in results.items()
    }
    summary_path = config.output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    save_summary(config.output_root / "evaluation_summary.json", {k: v["metrics"] for k, v in results.items()})
    return summary_payload


def _build_torch_model(model_name: str) -> torch.nn.Module:
    if model_name == "odin":
        return OdinPointNet()
    if model_name == "pointtransformerv3":
        return PointTransformerV3Mini()
    if model_name == "oneformer3d":
        return OneFormer3DMini()
    raise ValueError(model_name)


def _train_eval_torch(model_name: str, config: BenchmarkConfig, split) -> dict:
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = _build_torch_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loader = DataLoader(
        PointCloudSegmentationDataset(config.dataset_root, split.train, config.points_per_sample),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        PointCloudSegmentationDataset(config.dataset_root, split.val, config.points_per_sample),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        PointCloudSegmentationDataset(config.dataset_root, split.test, config.points_per_sample),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_batch,
    )

    class_weights = _estimate_class_weights(config.dataset_root, split.train).to(device)
    best_val = -1.0
    checkpoint_path = config.output_root / model_name / "best_model.pt"
    start_train = time.perf_counter()
    for _epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            points = batch["points"].to(device)
            labels = batch["labels"].to(device)
            logits = model(points)
            loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1), weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_metrics = _eval_torch_loader(model, val_loader, device)
        if val_metrics.macro_iou > best_val:
            best_val = val_metrics.macro_iou
            save_torch_checkpoint(model, checkpoint_path)
    train_seconds = time.perf_counter() - start_train

    model = load_torch_checkpoint(_build_torch_model(model_name), checkpoint_path, device)
    start_infer = time.perf_counter()
    metrics, predictions = _predict_torch_test(model_name, model, test_loader, device, config, split.test)
    inference_seconds = time.perf_counter() - start_infer
    return {
        "metrics": metrics,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "predictions": predictions,
    }


def _estimate_class_weights(dataset_root: Path, sample_names: list[str]) -> torch.Tensor:
    counts = np.ones(NUM_CLASSES, dtype=np.float64)
    for sample_name in sample_names:
        _points, labels = load_sample(dataset_root, sample_name)
        bincount = np.bincount(labels, minlength=NUM_CLASSES)
        counts[: len(bincount)] += bincount
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _fruit_macro(metric_by_class: dict[str, float]) -> float:
    fruit_classes = ["ripe", "half-ripe", "unripe"]
    return float(np.mean([metric_by_class[name] for name in fruit_classes]))


def _eval_torch_loader(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            labels = batch["labels"].cpu().numpy().reshape(-1)
            pred = model(points).argmax(dim=-1).cpu().numpy().reshape(-1)
            all_true.append(labels)
            all_pred.append(pred)
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def _predict_torch_test(model_name, model, test_loader, device, config, sample_names):
    model.eval()
    all_true = []
    all_pred = []
    prediction_dir = config.output_root / "predictions" / model_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    with torch.no_grad():
        for batch, sample_name in zip(test_loader, sample_names, strict=True):
            points = batch["points"].to(device)
            labels = batch["labels"].cpu().numpy().reshape(-1)
            pred = model(points).argmax(dim=-1).cpu().numpy().reshape(-1)
            np.save(prediction_dir / f"{sample_name}.npy", pred)
            all_true.append(labels)
            all_pred.append(pred)
            saved[sample_name] = str(prediction_dir / f"{sample_name}.npy")
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_pred)), saved


def _train_eval_openyolo(model_name: str, config: BenchmarkConfig, split) -> dict:
    train_samples = [load_sample(config.dataset_root, name) for name in split.train]
    start_train = time.perf_counter()
    model = OpenYolo3DModel.fit(train_samples)
    train_seconds = time.perf_counter() - start_train
    model_dir = config.output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    prediction_dir = config.output_root / "predictions" / model_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    all_true = []
    all_pred = []
    saved = {}
    start_infer = time.perf_counter()
    for sample_name in split.test:
        points, labels = load_sample(config.dataset_root, sample_name)
        pred = model.predict(points)
        np.save(prediction_dir / f"{sample_name}.npy", pred)
        all_true.append(labels)
        all_pred.append(pred)
        saved[sample_name] = str(prediction_dir / f"{sample_name}.npy")
    inference_seconds = time.perf_counter() - start_infer
    return {
        "metrics": compute_metrics(np.concatenate(all_true), np.concatenate(all_pred)),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "predictions": saved,
    }
