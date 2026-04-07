## Strawberry bush benchmark

This repository now contains an export pipeline that prepares the merged strawberry bush dataset for the 4 official models we compare:

- `PointTransformerV3` via `Pointcept DefaultDataset`
- `OneFormer3D` via a ScanNet-style `points/ + semantic_mask/ + instance_mask/ + pkl` layout
- `ODIN` via a ScanNet-context-like multiview RGB-D scene with a COCO-like JSON
- `OpenYOLO3D` via a Replica-like scene folder with RGB, depth, poses, intrinsics and a reconstructed bush point cloud

### Export all inputs

```powershell
python tools\prepare_strawberry_benchmark.py --force
```

Main export manifest:

- `data_exports/strawberry_benchmark/strawberry_benchmark_manifest.json`

### Evaluate point predictions

The evaluator expects predictions in the form:

```text
<predictions-root>/
  pointtransformerv3/sample_0000.npy
  oneformer3d/sample_0000.npy
  odin/sample_0000.npy
  openyolo3d/sample_0000.npy
```

Each `*.npy` file should contain one semantic class id per point in the corresponding `point_cloud_test/artifacts/dataset/sample_xxxx.npz`.

Run evaluation:

```powershell
python tools\evaluate_strawberry_predictions.py `
  --predictions-root artifacts\predictions `
  --models pointtransformerv3 oneformer3d odin openyolo3d `
  --split test
```

### Prepared export roots

- `data_exports/strawberry_benchmark/pointcept`
- `data_exports/strawberry_benchmark/oneformer3d`
- `data_exports/strawberry_benchmark/multiview/odin_frames`
- `data_exports/strawberry_benchmark/multiview/openyolo3d_scene`
Обзор Топ 4 3D семантических сегментаторов
