from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttributeValuePrediction:
    label_index: int
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_index": self.label_index,
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class FineGrainedAttributePrediction:
    attribute_name: str
    value: AttributeValuePrediction
    alternatives: tuple[AttributeValuePrediction, ...]
    is_confident: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute_name": self.attribute_name,
            "label_index": self.value.label_index,
            "label": self.value.label,
            "confidence": self.value.confidence,
            "is_confident": self.is_confident,
            "alternatives": [value.to_dict() for value in self.alternatives],
        }


@dataclass(frozen=True)
class AttributeExtractionResult:
    image_size: tuple[int, int]
    region_box: tuple[int, int, int, int]
    mask_area: int
    mask_coverage: float
    predictions: tuple[FineGrainedAttributePrediction, ...]
    preprocessing_time_ms: float
    inference_time_ms: float
    total_time_ms: float
    backend: str
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "region_box": list(self.region_box),
            "mask_area": self.mask_area,
            "mask_coverage": self.mask_coverage,
            "attributes": [prediction.to_dict() for prediction in self.predictions],
            "preprocessing_time_ms": self.preprocessing_time_ms,
            "inference_time_ms": self.inference_time_ms,
            "total_time_ms": self.total_time_ms,
            "backend": self.backend,
            "status": self.status,
        }
