from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class CocoAnnotation:
    annotation_id: int
    image_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    segmentation: list[list[float]]


@dataclass(slots=True)
class CocoImage:
    image_id: int
    file_name: str
    original_name: str
    width: int
    height: int


class CocoIndex:
    def __init__(self, annotation_path: Path) -> None:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.images_by_original_name: dict[str, CocoImage] = {}
        self.annotations_by_image_id: dict[int, list[CocoAnnotation]] = {}

        for raw_image in data["images"]:
            image = CocoImage(
                image_id=int(raw_image["id"]),
                file_name=raw_image["file_name"],
                original_name=raw_image.get("extra", {}).get("name", raw_image["file_name"]),
                width=int(raw_image["width"]),
                height=int(raw_image["height"]),
            )
            self.images_by_original_name[image.original_name] = image
            self.annotations_by_image_id[image.image_id] = []

        for raw_ann in data["annotations"]:
            ann = CocoAnnotation(
                annotation_id=int(raw_ann["id"]),
                image_id=int(raw_ann["image_id"]),
                category_id=int(raw_ann["category_id"]),
                bbox_xywh=tuple(float(v) for v in raw_ann["bbox"]),
                segmentation=raw_ann.get("segmentation", []),
            )
            self.annotations_by_image_id.setdefault(ann.image_id, []).append(ann)

    def annotations_for_original_name(self, image_name: str) -> tuple[CocoImage, list[CocoAnnotation]]:
        image = self.images_by_original_name[image_name]
        return image, self.annotations_by_image_id.get(image.image_id, [])


def _decode_rle_counts(encoded: str) -> list[int]:
    counts: list[int] = []
    current = 0
    shift = 0
    for character in encoded:
        value = ord(character) - 48
        current |= (value & 0x1F) << shift
        if value & 0x20:
            shift += 5
            continue
        counts.append(current)
        current = 0
        shift = 0
    return counts


def _rle_to_mask(segmentation: dict, height: int, width: int) -> np.ndarray:
    counts = segmentation.get("counts", [])
    if isinstance(counts, str):
        counts = _decode_rle_counts(counts)
    flat = np.zeros(height * width, dtype=np.uint8)
    value = 0
    index = 0
    for count in counts:
        end = min(index + int(count), flat.size)
        if value == 1:
            flat[index:end] = 1
        index = end
        value = 1 - value
        if index >= flat.size:
            break
    return flat.reshape((width, height), order="F").T.astype(bool)


def segmentation_to_mask(segmentation: list[list[float]] | dict, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        return _rle_to_mask(segmentation, height, width)
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask.astype(bool)
