from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
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
    ) -> None:
        self.image_dir = Path(image_dir)
        self.anno_dir = Path(anno_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.anno_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.anno_dir}")

        self.category_map = category_map or DEEPFASHION2_TO_PROJECT_CATEGORY
        self.annotation_paths = sorted(self.anno_dir.glob("*.json"))
        if not self.annotation_paths:
            raise ValueError(f"No DeepFashion2 annotations found in {self.anno_dir}")

    def __len__(self) -> int:
        return len(self.annotation_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        annotation_path = self.annotation_paths[index]
        with annotation_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)

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

            box = item.get("bounding_box") or self._box_from_mask(mask)
            boxes.append([float(value) for value in box])
            labels.append(project_label)
            masks.append(mask)

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
