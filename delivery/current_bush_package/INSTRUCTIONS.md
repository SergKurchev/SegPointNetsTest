# Instructions

## Open the packaged results

- Main summary: `D:\MyProjects\Robostrawberry\SegPointNetsTest\delivery\current_bush_package\SUMMARY.md`
- Package manifest: `D:\MyProjects\Robostrawberry\SegPointNetsTest\delivery\current_bush_package\package_manifest.json`
- Point-cloud dataset: `D:\MyProjects\Robostrawberry\SegPointNetsTest\delivery\current_bush_package\results\point_cloud_test\artifacts\dataset`
- 4-model benchmark exports: `D:\MyProjects\Robostrawberry\SegPointNetsTest\delivery\current_bush_package\results\SegPointNetsTest\data_exports\strawberry_benchmark`

## Regenerate the merged dataset

```powershell
cd D:\MyProjects\Robostrawberry\point_cloud_test
python -m point_cloud_test.scripts.generate_dataset --max-samples 32 --views-per-sample 4
```

## Rebuild the 4-model export package

```powershell
cd D:\MyProjects\Robostrawberry\SegPointNetsTest
python tools\prepare_strawberry_benchmark.py --force
```

## Re-evaluate saved predictions

```powershell
cd D:\MyProjects\Robostrawberry\SegPointNetsTest
python tools\evaluate_strawberry_predictions.py --predictions-root artifacts\predictions --models pointtransformerv3 oneformer3d odin openyolo3d --split test
```

## Rebuild this clean delivery folder

```powershell
cd D:\MyProjects\Robostrawberry\SegPointNetsTest
python tools\package_current_bush_delivery.py
```
