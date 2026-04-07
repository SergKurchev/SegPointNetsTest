# Current Bush Delivery

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

- Source dataset root in this package: `artifacts/dataset`
- Export root in this package: `SegPointNetsTest/data_exports/strawberry_benchmark`
- Splits: train=19, val=6, test=7
- Multiview frames exported: 150
- Frame stride used for multiview export: 2
- ReID proof for merged bush segments: old co-visibility violations=1706, new=0

## Current benchmark metrics

| Model | overall_accuracy | macro_iou | macro_f1 | fruit_macro_iou | fruit_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| odin | 0.0481 | 0.0122 | 0.0232 | 0.0162 | 0.0309 |
| pointtransformerv3 | 0.0748 | 0.0832 | 0.1402 | 0.1110 | 0.1869 |
| oneformer3d | 0.1838 | 0.0864 | 0.1543 | 0.0693 | 0.1250 |
| open-yolo-3d | 0.4157 | 0.1158 | 0.1705 | 0.0169 | 0.0328 |

## Best model snapshot

- Best by `fruit_macro_iou`: `pointtransformerv3` (0.1110)
- Best by `macro_iou`: `open-yolo-3d` (0.1158)

## Important note

The point-cloud samples do contain segment-level ground truth and ripeness class labels, but the ripeness label itself is currently derived heuristically from color statistics, not from a manually annotated biological ground truth source. That is the main quality limitation of the current dataset snapshot.

## Main entrypoints

- `src/point_cloud_test/scripts/generate_dataset.py`
- `src/point_cloud_test/scripts/evaluate_models.py`
- `SegPointNetsTest/tools/prepare_strawberry_benchmark.py`
- `SegPointNetsTest/tools/evaluate_strawberry_predictions.py`
- `SegPointNetsTest/scripts/run_pointcept_strawberry.ps1`
- `SegPointNetsTest/scripts/run_oneformer3d_strawberry.ps1`
