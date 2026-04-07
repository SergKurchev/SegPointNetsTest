# point_cloud_test

Отдельный проект рядом с `strawberry_tracking` для двух задач:

1. Сборка объединённых point clouds из орбитального RGB-D датасета.
2. Подготовка point-cloud датасета с GT-сегментацией и обвязки для сравнения 4 моделей сегментации.

## Что внутри

- `src/point_cloud_test/dataset_builder.py` - генерация multiview point cloud датасета.
- `src/point_cloud_test/evaluation.py` - метрики и сводка результатов моделей.
- `src/point_cloud_test/scripts/generate_dataset.py` - CLI для генерации датасета.
- `src/point_cloud_test/scripts/evaluate_models.py` - CLI для запуска/сравнения моделей.

## Источники данных

По умолчанию код использует:

- RGB-D + GT poses: `..\Perception_In_Robotics\FP\strawberry_orbit_data_new\data`
- 2D instance masks: `..\strawberry_tracking\Robo Strawberry.coco-segmentation\train\_annotations.coco.json`

## Быстрый старт

```powershell
cd .
python -m point_cloud_test.scripts.generate_dataset --max-samples 3 --views-per-sample 4
python -m point_cloud_test.scripts.evaluate_models --dataset-root artifacts\dataset
```

## Формат датасета

Каждый sample сохраняется как:

- `sample_XXXX.npz`
  - `points_xyzrgb`: `float32`, shape `[N, 6]`
  - `segment_id`: `int32`, shape `[N]`
  - `class_id`: `int16`, shape `[N]`
- `sample_XXXX.json`
  - сводка по view-группе
  - список сегментов с `segment_id`, `class_name`, `num_points`

Поддерживаемые классы:

- `background`
- `ripe`
- `half-ripe`
- `unripe`

Класс зрелости сейчас вычисляется эвристически по цвету сегмента в RGB/HSV, потому что в доступной COCO-разметке есть instance masks, но нет отдельной GT-разметки зрелости.

## Замечания по задаче 2

`evaluate_models.py` рассчитан на внешний репозиторий `SegPointNetsTest`, если он лежит в:

- `.\SegPointNetsTest`

Если репозиторий не скачан локально, скрипт всё равно считает метрики по готовым prediction-файлам и формирует сравнение.
