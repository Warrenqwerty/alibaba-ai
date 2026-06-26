from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fashion_mm.models.instance_segmentation.result import FashionInstance
from fashion_mm.models.instance_segmentation.result import SegmentationResult
from fashion_mm.models.local_region.proposal import generate_open_vocab_candidates
from fashion_mm.models.local_region.query import ParsedRegionQuery
from fashion_mm.models.local_region.query import parse_region_query
from fashion_mm.models.local_region.ranker import HeuristicRegionRanker
from fashion_mm.models.local_region.ranker import LearnedRegionRanker
from fashion_mm.models.local_region.ranker import RankedRegionCandidate


@dataclass(frozen=True)
class LocalRegionResult:
    """Language-guided local-region localization output."""

    query: ParsedRegionQuery
    selected_instance: FashionInstance | None
    proposal: RankedRegionCandidate | None
    candidates: list[RankedRegionCandidate]
    ranker_backend: str
    status: str
    reason: str | None
    latency_ms: float

    def to_dict(self, include_mask: bool = False, max_candidates: int = 5) -> dict[str, Any]:
        return {
            "query": {
                "text": self.query.query,
                "region": self.query.region,
                "garment_hint": self.query.garment_hint,
                "is_supported_region": self.query.is_supported_region,
                "spatial_hints": list(self.query.spatial_hints),
                "attribute_hints": list(self.query.attribute_hints),
                "relation_hints": list(self.query.relation_hints),
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
            "candidate_regions": [
                candidate.to_dict(include_mask=False)
                for candidate in self.candidates[:max_candidates]
            ],
            "ranker_backend": self.ranker_backend,
            "status": self.status,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


def localize_region_from_instances(
    segmentation: SegmentationResult,
    query: str,
    ranker: HeuristicRegionRanker | LearnedRegionRanker | None = None,
) -> LocalRegionResult:
    """Localize a queried local region from existing garment instances."""
    start = time.perf_counter()
    parsed_query = parse_region_query(query)

    selected_instance = select_garment_instance(segmentation, parsed_query)
    if selected_instance is None:
        return _result(
            parsed_query,
            None,
            None,
            [],
            HeuristicRegionRanker.backend_name,
            "no_garment_instance",
            "no garment instance matched the query",
            start,
        )

    candidates = generate_open_vocab_candidates(
        selected_instance.mask,
        selected_instance.box,
    )
    ranker = ranker or HeuristicRegionRanker()
    ranked_candidates = ranker.rank(
        parsed_query,
        candidates,
        selected_instance.box,
        selected_instance.label_name,
    )
    proposal = ranked_candidates[0] if ranked_candidates else None
    status = "ok" if proposal is not None else "no_region_candidate"
    effective_backend = proposal.backend if proposal is not None else ranker.backend_name
    return _result(
        parsed_query,
        selected_instance,
        proposal,
        ranked_candidates,
        effective_backend,
        status,
        proposal.proposal.reason if proposal is not None else "no candidate regions generated",
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
    proposal: RankedRegionCandidate | None,
    candidates: list[RankedRegionCandidate],
    ranker_backend: str,
    status: str,
    reason: str | None,
    start: float,
) -> LocalRegionResult:
    return LocalRegionResult(
        query=query,
        selected_instance=selected_instance,
        proposal=proposal,
        candidates=candidates,
        ranker_backend=ranker_backend,
        status=status,
        reason=reason,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
