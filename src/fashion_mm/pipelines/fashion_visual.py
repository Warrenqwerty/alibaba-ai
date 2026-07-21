from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fashion_mm.models.attributes import AttributeExtractionResult
from fashion_mm.models.attributes import FashionAttributePredictor
from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.instance_segmentation import SegmentationResult
from fashion_mm.models.local_region import LocalRegionResult
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.utils.image_io import ImageInput


@dataclass(frozen=True)
class FashionVisualPipelineResult:
    """Combined output of PRD sections 3.1.1, 3.1.2, and 3.1.3."""

    segmentation: SegmentationResult
    local_region: LocalRegionResult
    attribute_extraction: AttributeExtractionResult | None
    total_pipeline_time_ms: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "segmentation": self.segmentation.to_dict(include_masks=False),
            "local_region": self.local_region.to_dict(include_mask=False),
            "attribute_extraction": (
                self.attribute_extraction.to_dict()
                if self.attribute_extraction is not None
                else None
            ),
            "total_pipeline_time_ms": self.total_pipeline_time_ms,
        }


class FashionVisualPipeline:
    """Run segmentation, language-guided localization, and region attributes."""

    def __init__(
        self,
        segmentation_predictor: FashionInstanceSegmentationPredictor,
        attribute_predictor: FashionAttributePredictor,
    ) -> None:
        self.segmentation_predictor = segmentation_predictor
        self.attribute_predictor = attribute_predictor

    def predict(
        self,
        image: ImageInput,
        query: str,
        *,
        attributes: list[str] | tuple[str, ...] | None = None,
    ) -> FashionVisualPipelineResult:
        start = time.perf_counter()
        segmentation = self.segmentation_predictor.predict(image)
        local_region = localize_region_from_instances(segmentation, query)
        if local_region.proposal is None:
            return FashionVisualPipelineResult(
                segmentation=segmentation,
                local_region=local_region,
                attribute_extraction=None,
                total_pipeline_time_ms=(time.perf_counter() - start) * 1000.0,
                status=local_region.status,
            )

        extraction = self.attribute_predictor.predict(
            image,
            local_region.proposal.proposal.mask,
            attributes=attributes,
        )
        return FashionVisualPipelineResult(
            segmentation=segmentation,
            local_region=local_region,
            attribute_extraction=extraction,
            total_pipeline_time_ms=(time.perf_counter() - start) * 1000.0,
            status="ok",
        )
