# Instructions

## Open the packaged results

- Main summary: `SUMMARY.md`
- Package manifest: `package_manifest.json`
- Point-cloud dataset: `artifacts/dataset`
- 4-model benchmark exports: `SegPointNetsTest/data_exports/strawberry_benchmark`

## Regenerate the merged dataset

```powershell
cd point_cloud_test
python -m point_cloud_test.scripts.generate_dataset --max-samples 32 --views-per-sample 4
```

## Rebuild the 4-model export package

```powershell
cd point_cloud_test\SegPointNetsTest
python tools\prepare_strawberry_benchmark.py --force
```

## Re-evaluate saved predictions

```powershell
cd point_cloud_test\SegPointNetsTest
python tools\evaluate_strawberry_predictions.py --predictions-root artifacts\predictions --models pointtransformerv3 oneformer3d odin openyolo3d --split test
```

## Rebuild this packaged delivery snapshot

```powershell
cd point_cloud_test\SegPointNetsTest
python tools\package_current_bush_delivery.py
```
