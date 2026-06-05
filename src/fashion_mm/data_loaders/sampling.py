from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler

from fashion_mm.data_loaders.deepfashion2 import DeepFashion2Dataset


def build_balanced_sampler(
    dataset: DeepFashion2Dataset,
    categories: dict[int, str],
) -> WeightedRandomSampler:
    """Build an image sampler that upweights images containing rare classes."""
    foreground_ids = [label_id for label_id in categories if label_id != 0]
    class_counts = {label_id: 0 for label_id in foreground_ids}
    image_labels: list[list[int]] = []

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
        sample_weights.append(max(weights) if weights else fallback_weight)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


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
