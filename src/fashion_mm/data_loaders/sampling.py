from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from fashion_mm.data_loaders.deepfashion2 import DeepFashion2Dataset


DEFAULT_HARD_REASONS = {
    "dress_confused_as_top",
    "top_confused_as_dress",
    "low_iou_dress",
    "low_iou_top",
    "low_iou_pants",
}


def build_balanced_sampler(
    dataset: DeepFashion2Dataset,
    categories: dict[int, str],
    hard_case_config: dict[str, Any] | None = None,
) -> WeightedRandomSampler:
    """Build an image sampler that upweights images containing rare classes."""
    foreground_ids = [label_id for label_id in categories if label_id != 0]
    class_counts = {label_id: 0 for label_id in foreground_ids}
    image_labels: list[list[int]] = []
    hard_case_weights = build_hard_case_weights(dataset, hard_case_config)

    for index in range(len(dataset)):
        labels = sorted(set(dataset.get_labels(index)))
        image_labels.append(labels)
        for label_id in labels:
            if label_id in class_counts:
                class_counts[label_id] += 1

    class_weights = {
        label_id: 1.0 / max(count, 1)
        for label_id, count in class_counts.items()
    }
    fallback_weight = min(class_weights.values()) if class_weights else 1.0
    sample_weights = []
    for labels in image_labels:
        weights = [
            class_weights[label_id]
            for label_id in labels
            if label_id in class_weights
        ]
        base_weight = max(weights) if weights else fallback_weight
        sample_weights.append(base_weight * hard_case_weights[index])

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_hard_case_weights(
    dataset: DeepFashion2Dataset,
    hard_case_config: dict[str, Any] | None,
) -> list[float]:
    """Return per-image multipliers loaded from failure-analysis JSON output."""
    weights = [1.0 for _ in range(len(dataset))]
    if not hard_case_config or not bool(hard_case_config.get("enabled", False)):
        return weights

    hard_case_path = Path(str(hard_case_config.get("path", "")))
    if not hard_case_path.exists():
        raise FileNotFoundError(f"Hard case file not found: {hard_case_path}")

    reasons = set(hard_case_config.get("reasons") or DEFAULT_HARD_REASONS)
    multiplier = float(hard_case_config.get("weight_multiplier", 2.0))
    if multiplier < 1.0:
        raise ValueError("hard_mining.weight_multiplier must be >= 1.0")

    with hard_case_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    hard_images = {
        str(case["image"])
        for case in payload.get("cases", [])
        if case.get("reason") in reasons and case.get("image")
    }
    annotation_names = {
        Path(image_name).with_suffix(".json").name for image_name in hard_images
    }
    for index, annotation_path in enumerate(dataset.annotation_paths):
        if annotation_path.name in annotation_names:
            weights[index] = multiplier
    return weights


def count_images_by_class(
    dataset: DeepFashion2Dataset,
    categories: dict[int, str],
) -> dict[int, int]:
    """Count how many images contain each foreground class."""
    class_counts = {label_id: 0 for label_id in categories if label_id != 0}
    for index in range(len(dataset)):
        for label_id in sorted(set(dataset.get_labels(index))):
            if label_id in class_counts:
                class_counts[label_id] += 1
    return class_counts
