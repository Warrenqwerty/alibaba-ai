from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FashionInstance:
    """One segmented clothing instance."""

    mask: np.ndarray
    box: Box
    label_id: int
    label_name: str
    score: float

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError(f"Instance mask must be 2D, got {self.mask.shape}")
        if len(self.box) != 4:
            raise ValueError("Box must contain four coordinates.")

    @property
    def area(self) -> int:
        """Return the foreground mask area in pixels."""
        return int(np.asarray(self.mask, dtype=bool).sum())

    def to_dict(self, include_mask: bool = False) -> dict[str, Any]:
        """Serialize the instance for JSON-friendly output."""
        data: dict[str, Any] = {
            "box": [float(value) for value in self.box],
            "label_id": int(self.label_id),
            "label_name": self.label_name,
            "score": float(self.score),
            "area": self.area,
        }
        if include_mask:
            data["mask"] = np.asarray(self.mask, dtype=np.uint8).tolist()
        return data


@dataclass(frozen=True)
class SegmentationResult:
    """Instance segmentation output for one input image."""

    image_size: tuple[int, int]
    instances: list[FashionInstance] = field(default_factory=list)
    inference_time_ms: float | None = None

    def filter_by_score(self, threshold: float) -> "SegmentationResult":
        """Return a copy containing only instances above the score threshold."""
        return SegmentationResult(
            image_size=self.image_size,
            instances=[
                instance for instance in self.instances if instance.score >= threshold
            ],
            inference_time_ms=self.inference_time_ms,
        )

    def to_dict(self, include_masks: bool = False) -> dict[str, Any]:
        """Serialize the result for logs, APIs, or offline evaluation."""
        return {
            "image_size": list(self.image_size),
            "inference_time_ms": self.inference_time_ms,
            "instances": [
                instance.to_dict(include_mask=include_masks)
                for instance in self.instances
            ],
        }
