from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fashion_mm.models.instance_segmentation.result import FashionInstance
from fashion_mm.models.instance_segmentation.result import SegmentationResult
from fashion_mm.models.local_region.proposal import LocalRegionProposal
from fashion_mm.models.local_region.proposal import propose_local_region
from fashion_mm.models.local_region.query import ParsedRegionQuery
from fashion_mm.models.local_region.query import parse_region_query


@dataclass(frozen=True)
class LocalRegionResult:
    """Language-guided local-region localization output."""

    query: ParsedRegionQuery
    selected_instance: FashionInstance | None
    proposal: LocalRegionProposal | None
    status: str
    reason: str | None
    latency_ms: float

    def to_dict(self, include_mask: bool = False) -> dict[str, Any]:
        return {
            "query": {
                "text": self.query.query,
                "region": self.query.region,
                "garment_hint": self.query.garment_hint,
                "is_supported_region": self.query.is_supported_region,
            },
            "selected_instance": (
                self.selected_instance.to_dict(include_mask=False)
                if self.selected_instance is not None
                else None
            ),
            "region": (
                self.proposal.to_dict(include_mask=include_mask)
                if self.proposal is not None
                else None
            ),
            "status": self.status,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


def localize_region_from_instances(
    segmentation: SegmentationResult,
    query: str,
) -> LocalRegionResult:
    """Localize a queried local region from existing garment instances."""
    start = time.perf_counter()
    parsed_query = parse_region_query(query)
    if parsed_query.region is None:
        return _result(
            parsed_query,
            None,
            None,
            "unknown_region",
            "query does not mention a supported local clothing region",
            start,
        )

    selected_instance = select_garment_instance(segmentation, parsed_query)
    if selected_instance is None:
        return _result(
            parsed_query,
            None,
            None,
            "no_garment_instance",
            "no garment instance matched the query",
            start,
        )

    proposal = propose_local_region(
        selected_instance.mask,
        selected_instance.box,
        parsed_query.region,
    )
    status = "ok" if proposal.status == "ok" else proposal.status
    return _result(
        parsed_query,
        selected_instance,
        proposal,
        status,
        proposal.reason,
        start,
    )


def select_garment_instance(
    segmentation: SegmentationResult,
    query: ParsedRegionQuery,
) -> FashionInstance | None:
    """Select the garment instance to use as region-localization context."""
    if not segmentation.instances:
        return None

    candidates = segmentation.instances
    if query.garment_hint is not None:
        hinted = [
            instance
            for instance in segmentation.instances
            if instance.label_name == query.garment_hint
        ]
        if hinted:
            candidates = hinted

    return max(candidates, key=lambda instance: (instance.score, instance.area))


def _result(
    query: ParsedRegionQuery,
    selected_instance: FashionInstance | None,
    proposal: LocalRegionProposal | None,
    status: str,
    reason: str | None,
    start: float,
) -> LocalRegionResult:
    return LocalRegionResult(
        query=query,
        selected_instance=selected_instance,
        proposal=proposal,
        status=status,
        reason=reason,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
