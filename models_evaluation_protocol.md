# Политика сравнения моделей (Evaluation Protocol)

В данном документе описана стандартизированная методология тестирования и сравнения четырех архитектур для задачи 3D instance / panoptic segmentation:
1. Point Transformer V3
2. OneFormer3D
3. ODIN
4. Open-YOLO 3D

## 1. Загрузка весов и инициализация
- **Предобученные веса**: Каждая модель инициализируется актуальными предобученными весами от авторов оригинальных статей (предобученных на ScanNet, COCO и т.д.).
- **Воспроизводимость**: Строгая фиксация random seed для PyTorch, NumPy и других библиотек.

## 2. Дообучение на целевом датасете (Fine-tuning)
- **Процесс**: Модели обучаются (fine-tuning) на нашем проприетарном/целевом датасете.
- **Сходимость**: Обучение продолжается до полного схождения (convergence). Контроль осуществляется с помощью механизма Early Stopping по основной валидационной метрике PQ.
- **Сохранение чекпоинтов**: Каждую эпоху (или раз в N эпох) сохраняются веса модели («лучшая» модель обновляется при побитии рекорда метрики).

## 3. Трекинг метрик (Per-Epoch Logging)
На каждой валидационной эпохе обязательно вычисление и логирование следующего набора метрик.
**Метрики качества** (вычисляются в 3D пространстве на GT-объектах/вокселях):
- `PQ` (Panoptic Quality)
- `SQ` (Segmentation Quality)
- `RQ` (Recognition Quality)
- `mAP` (Mean Average Precision, среднее по порогам)
- `mAP@50` (mAP при пороге IoU 0.5)
- `mAP@25` (mAP при пороге IoU 0.25)

### Формулы метрик (3D)

**PQ (Panoptic Quality)**  
$PQ = \frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$  
где $IoU(p,g) = \frac{|p \cap g|}{|p \cup g|}$ считается над 3‑D масками (вокселями или точками).  

**SQ (Segmentation Quality)**  
$SQ = \frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP|}$  

**RQ (Recognition Quality)**  
$RQ = \frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$  

**mAP** (Mean Average Precision)  
Для каждого IoU‑порога $\tau$ (например 0.25, 0.5) вычисляем AP как площадь под PR‑кривой, затем усредняем по всем порогам:  
$mAP = \frac{1}{T}\sum_{t=1}^{T} AP_{\tau_t}$  

**mAP@50** – AP при $\tau = 0.5$.  
**mAP@25** – AP при $\tau = 0.25$.  

### Официальные реализации 

- **ScanNet Instance Evaluation** (mAP, mAP@50, mAP@25):  
  `https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py`  
  *Зависит от `util_3d.py`. Формат входа: `.txt` с путями к бинарным маскам, `label_id`, `confidence`.*

- **SemanticKITTI Panoptic Evaluation** (PQ, SQ, RQ, IoU, PQ†):  
  `https://github.com/PRBonn/semantic-kitti-api/blob/master/evaluate_panoptic.py`, and
  `https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/eval_np.py` 
  *Требует `auxiliary.eval_np.PanopticEval` и `config/semantic-kitti.yaml`. Формат входа: `.label` файлы с packed IDs.*

- **COCO-style mAP (адаптация под 3D)**:  
  `https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py`  
  *База для реализации mAP с кастомным 3D IoU.*

Эти скрипты вычисляют метрики на вокселизованных облаках точек или на точечных масках и могут быть напрямую интегрированы в цикл обучения.

**Производительность**:
- `Train Speed` (время прохождения одной эпохи или step time, в сек/батч или мин/эпоха)
- `Inference Speed` (время инференса на одну 3D-сцену, в сек/сцена, замеряется на тестовом наборе с выключенным подсчетом градиентов).


## 4. Сохранение результатов и визуализация
Финальный этап после того, как дообучение всех моделей завершено:

1. **Экспорт метрик (CSV)**:
   - Собираются лучшие (best) метрики по каждой из 4 моделей на валидационном сете.
   - Данные сохраняются в единый файл `final_metrics_comparison.csv` (столбцы: Model, PQ, SQ, RQ, mAP, mAP@50, mAP@25, Inf_Speed, Train_Speed).

2. **Построение графиков**:
   - Генерация сравнительных диаграмм (Bar Charts для сравнения итоговых показателей, Line Charts для кривых обучения).

3. **Экспорт визуализаций сегментации (Point Clouds)**:
   Для набора репрезентативных сцен (сэмплы 0000, 0003, 0005) из тестовой выборки результаты сегментации должны быть экспортированы в двух форматах:
   - **Статические изображения**: Рендер сцены в `3-х фиксированных камерах` (ракурсах), чтобы сразу видеть качество сегментации со всех сторон (в формате PNG/JPG).
   - **Интерактивные HTML**: Сохранение 3D-облаков с размеченными инстансами и классами в виде интерактивных HTML-файлов (например, с помощью библиотек Plotly, Open3D WebVisualizer или PyVista), чтобы можно было крутить предсказания моделей в браузере.
