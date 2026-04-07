from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CopyItem:
    source: Path
    relative_target: Path


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _seg_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _point_cloud_root() -> Path:
    return _workspace_root() / "point_cloud_test"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _copy_item(item: CopyItem, destination_root: Path) -> dict[str, object]:
    destination = destination_root / item.relative_target
    if item.source.is_dir():
        shutil.copytree(item.source, destination, dirs_exist_ok=True)
        kind = "directory"
        size_bytes = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    else:
        _ensure_parent(destination)
        shutil.copy2(item.source, destination)
        kind = "file"
        size_bytes = destination.stat().st_size
    return {
        "source": str(item.source),
        "target": str(destination),
        "kind": kind,
        "size_bytes": int(size_bytes),
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metrics_table(summary: dict) -> str:
    headers = [
        "Model",
        "overall_accuracy",
        "macro_iou",
        "macro_f1",
        "fruit_macro_iou",
        "fruit_macro_f1",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for model_name, payload in summary.items():
        row = [model_name]
        for key in headers[1:]:
            value = payload.get(key, 0.0)
            row.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _best_model_line(summary: dict) -> str:
    best_by_fruit = max(summary.items(), key=lambda item: float(item[1].get("fruit_macro_iou", 0.0)))
    best_by_macro = max(summary.items(), key=lambda item: float(item[1].get("macro_iou", 0.0)))
    return (
        f"- Best by `fruit_macro_iou`: `{best_by_fruit[0]}` ({float(best_by_fruit[1]['fruit_macro_iou']):.4f})\n"
        f"- Best by `macro_iou`: `{best_by_macro[0]}` ({float(best_by_macro[1]['macro_iou']):.4f})"
    )


def _build_summary_markdown(
    benchmark_summary: dict,
    benchmark_manifest: dict,
    clustering_summary: dict,
    delivery_root: Path,
) -> str:
    dataset_root = delivery_root / "results" / "point_cloud_test" / "artifacts" / "dataset"
    export_root = delivery_root / "results" / "SegPointNetsTest" / "data_exports" / "strawberry_benchmark"
    return f"""# Current Bush Delivery

## Scope

This package collects the current state of work for the two strawberry point-cloud tasks:

1. merged RGB-D point cloud generation with segment-level ground truth;
2. benchmark preparation and comparison of 4 point-cloud segmentation models.

Everything here was copied into a clean delivery folder so it can be added to git separately from the older repo contents.

## What is included

- Dataset generator and evaluation code from `point_cloud_test`.
- Export/adaptation scripts from `SegPointNetsTest` for PointTransformerV3, OneFormer3D, ODIN, and Open-YOLO-3D.
- Generated dataset samples with `points_xyzrgb`, `segment_id`, and `class_id`.
- Exported benchmark layouts for the 4 target models.
- Prediction files, JSON reports, and visual proofs.

## Current dataset status

- Source dataset root in this package: `{dataset_root}`
- Export root in this package: `{export_root}`
- Splits: train={len(benchmark_manifest['splits']['train'])}, val={len(benchmark_manifest['splits']['val'])}, test={len(benchmark_manifest['splits']['test'])}
- Multiview frames exported: {benchmark_manifest['multiview_summary']['num_frames']}
- Frame stride used for multiview export: {benchmark_manifest['multiview_summary']['frame_stride']}
- ReID proof for merged bush segments: old co-visibility violations={clustering_summary['old_global_co_visibility_violations']}, new={clustering_summary['new_global_co_visibility_violations']}

## Current benchmark metrics

{_format_metrics_table(benchmark_summary)}

## Best model snapshot

{_best_model_line(benchmark_summary)}

## Important note

The point-cloud samples do contain segment-level ground truth and ripeness class labels, but the ripeness label itself is currently derived heuristically from color statistics, not from a manually annotated biological ground truth source. That is the main quality limitation of the current dataset snapshot.

## Main entrypoints

- `scripts/point_cloud_test/src/point_cloud_test/scripts/generate_dataset.py`
- `scripts/point_cloud_test/src/point_cloud_test/scripts/evaluate_models.py`
- `scripts/SegPointNetsTest/tools/prepare_strawberry_benchmark.py`
- `scripts/SegPointNetsTest/tools/evaluate_strawberry_predictions.py`
- `scripts/SegPointNetsTest/scripts/run_pointcept_strawberry.ps1`
- `scripts/SegPointNetsTest/scripts/run_oneformer3d_strawberry.ps1`
"""


def _build_instructions_markdown(delivery_root: Path) -> str:
    return f"""# Instructions

## Open the packaged results

- Main summary: `{delivery_root / 'SUMMARY.md'}`
- Package manifest: `{delivery_root / 'package_manifest.json'}`
- Point-cloud dataset: `{delivery_root / 'results' / 'point_cloud_test' / 'artifacts' / 'dataset'}`
- 4-model benchmark exports: `{delivery_root / 'results' / 'SegPointNetsTest' / 'data_exports' / 'strawberry_benchmark'}`

## Regenerate the merged dataset

```powershell
cd D:\\MyProjects\\Robostrawberry\\point_cloud_test
python -m point_cloud_test.scripts.generate_dataset --max-samples 32 --views-per-sample 4
```

## Rebuild the 4-model export package

```powershell
cd D:\\MyProjects\\Robostrawberry\\SegPointNetsTest
python tools\\prepare_strawberry_benchmark.py --force
```

## Re-evaluate saved predictions

```powershell
cd D:\\MyProjects\\Robostrawberry\\SegPointNetsTest
python tools\\evaluate_strawberry_predictions.py --predictions-root artifacts\\predictions --models pointtransformerv3 oneformer3d odin openyolo3d --split test
```

## Rebuild this clean delivery folder

```powershell
cd D:\\MyProjects\\Robostrawberry\\SegPointNetsTest
python tools\\package_current_bush_delivery.py
```
"""


def _package_items() -> list[CopyItem]:
    seg_root = _seg_repo_root()
    point_root = _point_cloud_root()
    return [
        CopyItem(point_root / "README.md", Path("scripts/point_cloud_test/README.md")),
        CopyItem(point_root / "MODEL_EVAL_GUIDE.md", Path("scripts/point_cloud_test/MODEL_EVAL_GUIDE.md")),
        CopyItem(point_root / "pyproject.toml", Path("scripts/point_cloud_test/pyproject.toml")),
        CopyItem(point_root / "src", Path("scripts/point_cloud_test/src")),
        CopyItem(seg_root / "README.md", Path("scripts/SegPointNetsTest/README.md")),
        CopyItem(seg_root / "tools" / "prepare_strawberry_benchmark.py", Path("scripts/SegPointNetsTest/tools/prepare_strawberry_benchmark.py")),
        CopyItem(seg_root / "tools" / "evaluate_strawberry_predictions.py", Path("scripts/SegPointNetsTest/tools/evaluate_strawberry_predictions.py")),
        CopyItem(seg_root / "tools" / "collect_pointcept_predictions.py", Path("scripts/SegPointNetsTest/tools/collect_pointcept_predictions.py")),
        CopyItem(seg_root / "tools" / "export_oneformer3d_predictions.py", Path("scripts/SegPointNetsTest/tools/export_oneformer3d_predictions.py")),
        CopyItem(seg_root / "tools" / "package_current_bush_delivery.py", Path("scripts/SegPointNetsTest/tools/package_current_bush_delivery.py")),
        CopyItem(seg_root / "scripts", Path("scripts/SegPointNetsTest/scripts")),
        CopyItem(seg_root / "configs" / "strawberry", Path("scripts/SegPointNetsTest/configs/strawberry")),
        CopyItem(point_root / "artifacts" / "dataset", Path("results/point_cloud_test/artifacts/dataset")),
        CopyItem(point_root / "artifacts" / "proof", Path("results/point_cloud_test/artifacts/proof")),
        CopyItem(point_root / "artifacts" / "benchmark_current_bush", Path("results/point_cloud_test/artifacts/benchmark_current_bush")),
        CopyItem(point_root / "artifacts" / "benchmark" / "metrics_overview.png", Path("results/point_cloud_test/artifacts/benchmark/metrics_overview.png")),
        CopyItem(point_root / "artifacts" / "benchmark" / "runtime_comparison.png", Path("results/point_cloud_test/artifacts/benchmark/runtime_comparison.png")),
        CopyItem(point_root / "artifacts" / "benchmark" / "sample_0001_qualitative.png", Path("results/point_cloud_test/artifacts/benchmark/sample_0001_qualitative.png")),
        CopyItem(point_root / "artifacts" / "reid" / "strawberry_remap.json", Path("results/point_cloud_test/artifacts/reid/strawberry_remap.json")),
        CopyItem(seg_root / "data_exports" / "strawberry_benchmark", Path("results/SegPointNetsTest/data_exports/strawberry_benchmark")),
        CopyItem(seg_root / "artifacts" / "predictions", Path("results/SegPointNetsTest/artifacts/predictions")),
        CopyItem(seg_root / "artifacts" / "reports", Path("results/SegPointNetsTest/artifacts/reports")),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the current strawberry point-cloud task results into a clean delivery folder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_seg_repo_root() / "delivery" / "current_bush_package",
        help="Where to assemble the clean delivery package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_items = [_copy_item(item, output_dir) for item in _package_items()]

    benchmark_summary = _load_json(_point_cloud_root() / "artifacts" / "benchmark_current_bush" / "benchmark_summary.json")
    benchmark_manifest = _load_json(_seg_repo_root() / "data_exports" / "strawberry_benchmark" / "strawberry_benchmark_manifest.json")
    clustering_summary = _load_json(_point_cloud_root() / "artifacts" / "proof" / "clustering_proof_summary.json")

    summary_markdown = _build_summary_markdown(benchmark_summary, benchmark_manifest, clustering_summary, output_dir)
    (output_dir / "SUMMARY.md").write_text(summary_markdown, encoding="utf-8")
    (output_dir / "INSTRUCTIONS.md").write_text(_build_instructions_markdown(output_dir), encoding="utf-8")

    package_manifest = {
        "output_dir": str(output_dir),
        "copied_items": copied_items,
        "benchmark_models": sorted(benchmark_summary.keys()),
        "split_sizes": {name: len(values) for name, values in benchmark_manifest["splits"].items()},
        "multiview_frames": int(benchmark_manifest["multiview_summary"]["num_frames"]),
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")
    print(json.dumps(package_manifest, indent=2))


if __name__ == "__main__":
    main()
