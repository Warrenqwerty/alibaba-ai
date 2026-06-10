from __future__ import annotations

import json
import random
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


DEEPFASHION2_TO_PROJECT_CATEGORY = {
    1: 1,
    2: 1,
    3: 4,
    4: 4,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 3,
    10: 5,
    11: 5,
    12: 5,
    13: 5,
}


class DeepFashion2Dataset(Dataset):
    """Torch detection dataset adapter for DeepFashion2 annotations."""

    def __init__(
        self,
        image_dir: str | Path,
        anno_dir: str | Path,
        category_map: dict[int, int] | None = None,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.anno_dir = Path(anno_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.anno_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.anno_dir}")

        self.category_map = category_map or DEEPFASHION2_TO_PROJECT_CATEGORY
        self.augmentation = augmentation or {}
        self.annotation_paths = [
            path
            for path in sorted(self.anno_dir.glob("*.json"))
            if not path.name.startswith(".")
        ]
        if not self.annotation_paths:
            raise ValueError(f"No DeepFashion2 annotations found in {self.anno_dir}")

    def __len__(self) -> int:
        return len(self.annotation_paths)

    def get_labels(self, index: int) -> list[int]:
        """Return project labels present in one annotation file."""
        annotation = self._load_annotation(self.annotation_paths[index])
        labels: list[int] = []
        for item in self._iter_items(annotation):
            project_label = self.category_map.get(int(item["category_id"]))
            if project_label is not None:
                labels.append(project_label)
        return labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        annotation_path = self.annotation_paths[index]
        annotation = self._load_annotation(annotation_path)

        image = self._load_image(annotation, annotation_path)
        width, height = image.size
        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        labels: list[int] = []

        for item in self._iter_items(annotation):
            project_label = self.category_map.get(int(item["category_id"]))
            if project_label is None:
                continue

            mask = self._polygon_to_mask(item.get("segmentation", []), width, height)
            if mask.sum() == 0:
                continue

            box = self._resolve_box(item.get("bounding_box"), mask)
            if box is None:
                continue
            boxes.append([float(value) for value in box])
            labels.append(project_label)
            masks.append(mask)

        image, masks, boxes = self._apply_augmentation(image, masks, boxes)
        width, height = image.size

        if masks:
            mask_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            box_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            label_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (box_tensor[:, 2] - box_tensor[:, 0]) * (
                box_tensor[:, 3] - box_tensor[:, 1]
            )
        else:
            mask_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            label_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "masks": mask_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return F.to_tensor(image), target

    def _apply_augmentation(
        self,
        image: Image.Image,
        masks: list[np.ndarray],
        boxes: list[list[float]],
    ) -> tuple[Image.Image, list[np.ndarray], list[list[float]]]:
        if not bool(self.augmentation.get("enabled", False)):
            return image, masks, boxes

        image = self._apply_color_jitter(image)

        scale_range = self.augmentation.get("scale_jitter")
        if scale_range:
            image, masks, boxes = self._apply_scale_jitter(
                image,
                masks,
                boxes,
                float(scale_range[0]),
                float(scale_range[1]),
            )

        flip_prob = float(self.augmentation.get("horizontal_flip_prob", 0.0))
        if random.random() < flip_prob:
            image, masks, boxes = self._horizontal_flip(image, masks, boxes)

        return image, masks, boxes

    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        brightness = float(self.augmentation.get("brightness", 0.0))
        contrast = float(self.augmentation.get("contrast", 0.0))
        saturation = float(self.augmentation.get("saturation", 0.0))

        if brightness > 0:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(1.0 - brightness, 1.0 + brightness)
            )
        if contrast > 0:
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(1.0 - contrast, 1.0 + contrast)
            )
        if saturation > 0:
            image = ImageEnhance.Color(image).enhance(
                random.uniform(1.0 - saturation, 1.0 + saturation)
            )
        return image

    @staticmethod
    def _apply_scale_jitter(
        image: Image.Image,
        masks: list[np.ndarray],
        boxes: list[list[float]],
        min_scale: float,
        max_scale: float,
    ) -> tuple[Image.Image, list[np.ndarray], list[list[float]]]:
        scale = random.uniform(min_scale, max_scale)
        if abs(scale - 1.0) < 1e-3:
            return image, masks, boxes

        width, height = image.size
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized_image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
        resized_masks = [
            np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    new_size,
                    resample=Image.Resampling.NEAREST,
                )
            )
            > 0
            for mask in masks
        ]
        resized_boxes = [
            [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale]
            for box in boxes
        ]
        return resized_image, resized_masks, resized_boxes

    @staticmethod
    def _horizontal_flip(
        image: Image.Image,
        masks: list[np.ndarray],
        boxes: list[list[float]],
    ) -> tuple[Image.Image, list[np.ndarray], list[list[float]]]:
        width, _ = image.size
        flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped_masks = [np.fliplr(mask).copy() for mask in masks]
        flipped_boxes = [
            [width - box[2], box[1], width - box[0], box[3]]
            for box in boxes
        ]
        return flipped_image, flipped_masks, flipped_boxes

    @staticmethod
    def _load_annotation(annotation_path: Path) -> dict[str, Any]:
        try:
            with annotation_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (UnicodeDecodeError, JSONDecodeError) as error:
            raise ValueError(
                f"Invalid DeepFashion2 annotation file: {annotation_path}. "
                "Remove hidden macOS resource files such as ._*.json if present."
            ) from error
        if not isinstance(data, dict):
            raise ValueError(f"Invalid DeepFashion2 annotation mapping: {annotation_path}")
        return data

    def _load_image(
        self,
        annotation: dict[str, Any],
        annotation_path: Path,
    ) -> Image.Image:
        image_name = annotation.get("source") or f"{annotation_path.stem}.jpg"
        image_path = self.image_dir / image_name
        if not image_path.exists():
            image_path = self.image_dir / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found for {annotation_path}: {image_path}"
            )
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _iter_items(annotation: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            value
            for key, value in annotation.items()
            if key.startswith("item") and isinstance(value, dict)
        ]

    @staticmethod
    def _polygon_to_mask(polygons: list[Any], width: int, height: int) -> np.ndarray:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            if len(polygon) < 6:
                continue
            points = [
                (float(polygon[i]), float(polygon[i + 1]))
                for i in range(0, len(polygon), 2)
            ]
            draw.polygon(points, outline=1, fill=1)
        return np.asarray(mask, dtype=np.uint8)

    @staticmethod
    def _box_from_mask(mask: np.ndarray) -> list[float]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    @classmethod
    def _resolve_box(
        cls,
        annotation_box: list[Any] | None,
        mask: np.ndarray,
    ) -> list[float] | None:
        """Return a valid x1/y1/x2/y2 box, falling back to mask bounds."""
        if annotation_box is not None and len(annotation_box) == 4:
            box = [float(value) for value in annotation_box]
            if cls._is_valid_box(box):
                return box

        mask_box = cls._box_from_mask(mask)
        if cls._is_valid_box(mask_box):
            return mask_box
        return None

    @staticmethod
    def _is_valid_box(box: list[float]) -> bool:
        return box[2] > box[0] and box[3] > box[1]
