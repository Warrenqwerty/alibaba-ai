from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from fashion_mm.models.local_region.learned_ranker import BoxCandidate
from fashion_mm.models.local_region.learned_ranker import build_pair_feature
from fashion_mm.models.local_region.learned_ranker import build_candidate_record_feature
from fashion_mm.models.local_region.learned_ranker import CandidateListwiseScorer
from fashion_mm.models.local_region.learned_ranker import HashingTextRegionScorer
from fashion_mm.models.local_region.learned_ranker import LEARNED_RANKER_CANDIDATE_REGIONS
from fashion_mm.models.local_region.proposal import LocalRegionProposal
from fashion_mm.models.local_region.query import ParsedRegionQuery


REGION_TEXT_HINTS = {
    "whole_garment": ("整件", "这件", "衣服", "服饰", "整体"),
    "upper": ("上方", "上面", "上半", "胸前", "胸口"),
    "lower": ("下方", "下面", "下半", "底部"),
    "left": ("左边", "左侧", "左面"),
    "right": ("右边", "右侧", "右面"),
    "center": ("中间", "中央", "正面", "前面"),
    "neckline": ("领口", "衣领", "领型", "领子"),
    "hem": ("下摆", "衣摆", "裙摆", "裤脚"),
    "shoulder": ("肩部", "肩线", "肩膀"),
    "waist": ("腰部", "腰线", "收腰"),
    "left_cuff": ("左边袖口", "左侧袖口", "左袖口"),
    "right_cuff": ("右边袖口", "右侧袖口", "右袖口"),
    "left_pocket": ("左边口袋", "左边的口袋", "左侧口袋", "左侧的口袋", "左口袋"),
    "right_pocket": ("右边口袋", "右边的口袋", "右侧口袋", "右侧的口袋", "右口袋"),
    "zipper": ("拉链", "拉锁"),
    "button": ("纽扣", "扣子"),
    "pattern": ("图案", "印花", "花纹", "纹理", "碎花", "条纹", "格纹"),
    "decoration": ("装饰", "珠片", "亮片", "刺绣", "蝴蝶结"),
}

REGION_EQUIVALENTS = {
    "cuff": ("left_cuff", "right_cuff"),
    "decoration": ("decoration", "center", "upper", "whole_garment"),
    "pocket": ("left_pocket", "right_pocket", "left", "right", "center"),
}

HASH_RANKER_SUPPORTED_REGIONS = {"neckline", "hem"}
# Manual bbox evaluation showed the candidate-listwise weak-supervised ranker
# underperforms the heuristic online pipeline, so keep it disabled for inference.
CANDIDATE_LISTWISE_SUPPORTED_REGIONS: set[str] = set()


@dataclass(frozen=True)
class RankedRegionCandidate:
    """A candidate local region with a text-matching score."""

    proposal: LocalRegionProposal
    score: float
    reason: str
    backend: str

    def to_dict(self, include_mask: bool = False) -> dict:
        payload = self.proposal.to_dict(include_mask=include_mask)
        payload["match_score"] = self.score
        payload["match_reason"] = self.reason
        payload["ranker_backend"] = self.backend
        return payload


class HeuristicRegionRanker:
    """Dependency-light text-region ranker for the first 3.1.2 prototype.

    This is not the final PRD model. It keeps the pipeline open-vocabulary by
    ranking many generic region candidates from raw query text. The intended
    upgrade path is to replace this scorer with DINOv2/text-embedding similarity.
    """

    backend_name = "heuristic_text_region_ranker"

    def rank(
        self,
        query: ParsedRegionQuery,
        candidates: list[LocalRegionProposal],
        garment_box: tuple[float, float, float, float] | None = None,
        category_text: str | None = None,
    ) -> list[RankedRegionCandidate]:
        ranked = [
            RankedRegionCandidate(
                proposal=candidate,
                score=self._score_candidate(query, candidate),
                reason=self._reason(query, candidate),
                backend=self.backend_name,
            )
            for candidate in candidates
        ]
        return sorted(
            ranked,
            key=lambda item: (item.score, item.proposal.confidence, _area(item.proposal)),
            reverse=True,
        )

    def _score_candidate(
        self,
        query: ParsedRegionQuery,
        candidate: LocalRegionProposal,
    ) -> float:
        text = query.query
        score = candidate.confidence

        if query.region == candidate.region:
            score += 1.5
        if query.region in REGION_EQUIVALENTS and candidate.region in REGION_EQUIVALENTS[query.region]:
            score += 1.2
        if any(hint in text for hint in REGION_TEXT_HINTS.get(candidate.region, ())):
            score += 1.0

        score += self._spatial_score(text, candidate.region)
        score += self._attribute_score(text, candidate.region)
        return round(score, 4)

    def _spatial_score(self, text: str, region: str) -> float:
        score = 0.0
        if any(term in text for term in ("左边", "左侧", "左面", "左")):
            if region in {"left", "left_cuff", "left_pocket"}:
                score += 0.9
            if region in {"right", "right_cuff", "right_pocket"}:
                score -= 0.4
        if any(term in text for term in ("右边", "右侧", "右面", "右")):
            if region in {"right", "right_cuff", "right_pocket"}:
                score += 0.9
            if region in {"left", "left_cuff", "left_pocket"}:
                score -= 0.4
        if any(term in text for term in ("上方", "上面", "上半", "顶部")) and region in {
            "upper",
            "neckline",
            "shoulder",
        }:
            score += 0.5
        if any(term in text for term in ("下方", "下面", "下半", "底部")) and region in {
            "lower",
            "hem",
        }:
            score += 0.5
        return score

    def _attribute_score(self, text: str, region: str) -> float:
        if any(term in text for term in ("碎花", "印花", "图案", "花纹", "条纹", "格纹")):
            return 0.8 if region == "pattern" else 0.0
        if any(term in text for term in ("拉链", "拉锁")):
            return 0.9 if region == "zipper" else 0.0
        if any(term in text for term in ("扣子", "纽扣")):
            return 0.9 if region == "button" else 0.0
        if any(term in text for term in ("装饰", "珠片", "亮片", "刺绣", "蝴蝶结")):
            return 0.75 if region == "decoration" else 0.0
        return 0.0

    def _reason(self, query: ParsedRegionQuery, candidate: LocalRegionProposal) -> str:
        if query.region == candidate.region:
            return "exact parsed region match"
        if query.region in REGION_EQUIVALENTS and candidate.region in REGION_EQUIVALENTS[query.region]:
            return "parsed region matched equivalent candidate"
        if any(hint in query.query for hint in REGION_TEXT_HINTS.get(candidate.region, ())):
            return "raw query keyword matched candidate"
        return "generic open-vocabulary candidate"


class LearnedRegionRanker:
    """Checkpoint-backed lightweight text-region ranker for 3.1.2."""

    backend_name = "hybrid_learned_local_region_ranker"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.num_buckets = int(checkpoint.get("num_buckets", 256))
        self.hidden_dim = int(checkpoint.get("hidden_dim", 128))
        self.checkpoint_kind = _checkpoint_kind(checkpoint)
        if self.checkpoint_kind == "candidate_listwise":
            self.backend_name = "hybrid_candidate_listwise_context_ranker"
            self.supported_regions = CANDIDATE_LISTWISE_SUPPORTED_REGIONS
            self.model = CandidateListwiseScorer(
                num_buckets=self.num_buckets,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
        else:
            self.backend_name = "hybrid_learned_hash_text_geometry_ranker"
            self.supported_regions = HASH_RANKER_SUPPORTED_REGIONS
            self.model = HashingTextRegionScorer(
                num_buckets=self.num_buckets,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.fallback_ranker = HeuristicRegionRanker()

    def rank(
        self,
        query: ParsedRegionQuery,
        candidates: list[LocalRegionProposal],
        garment_box: tuple[float, float, float, float] | None = None,
        category_text: str | None = None,
    ) -> list[RankedRegionCandidate]:
        if query.region not in self.supported_regions:
            return self._fallback_rank(query, candidates, garment_box, category_text)
        if garment_box is None:
            raise ValueError("garment_box is required for LearnedRegionRanker")
        if self.checkpoint_kind == "candidate_listwise":
            return self._rank_candidate_listwise(
                query,
                candidates,
                garment_box,
                category_text,
            )

        ranked: list[RankedRegionCandidate] = []
        with torch.no_grad():
            for candidate in candidates:
                if candidate.box is None:
                    continue
                feature = build_pair_feature(
                    query.query,
                    BoxCandidate(candidate.region, candidate.box),
                    garment_box,
                    num_buckets=self.num_buckets,
                ).to(self.device)
                score = float(self.model(feature.unsqueeze(0)).detach().cpu()[0])
                ranked.append(
                    RankedRegionCandidate(
                        proposal=candidate,
                        score=round(score, 4),
                        reason="learned hash text-geometry score",
                        backend=self.backend_name,
                    )
                )

        return sorted(
            ranked,
            key=lambda item: (item.score, item.proposal.confidence, _area(item.proposal)),
            reverse=True,
        )

    def _fallback_rank(
        self,
        query: ParsedRegionQuery,
        candidates: list[LocalRegionProposal],
        garment_box: tuple[float, float, float, float] | None,
        category_text: str | None,
    ) -> list[RankedRegionCandidate]:
        ranked = self.fallback_ranker.rank(query, candidates, garment_box, category_text)
        return [
            RankedRegionCandidate(
                proposal=item.proposal,
                score=item.score,
                reason=f"heuristic fallback for unsupported learned region: {item.reason}",
                backend=item.backend,
            )
            for item in ranked
        ]

    def _rank_candidate_listwise(
        self,
        query: ParsedRegionQuery,
        candidates: list[LocalRegionProposal],
        garment_box: tuple[float, float, float, float],
        category_text: str | None,
    ) -> list[RankedRegionCandidate]:
        ranked: list[RankedRegionCandidate] = []
        trained_regions = set(LEARNED_RANKER_CANDIDATE_REGIONS)
        with torch.no_grad():
            for candidate in candidates:
                if candidate.box is None or candidate.region not in trained_regions:
                    continue
                feature = build_candidate_record_feature(
                    query.query,
                    candidate.region,
                    garment_box,
                    candidate.box,
                    query.region,
                    category_text=category_text,
                    num_buckets=self.num_buckets,
                ).to(self.device)
                score = float(self.model(feature.unsqueeze(0)).detach().cpu()[0])
                ranked.append(
                    RankedRegionCandidate(
                        proposal=candidate,
                        score=round(score, 4),
                        reason="learned listwise candidate context score",
                        backend=self.backend_name,
                    )
                )

        if not ranked:
            return self._fallback_rank(query, candidates, garment_box, category_text)
        return sorted(
            ranked,
            key=lambda item: (item.score, item.proposal.confidence, _area(item.proposal)),
            reverse=True,
        )


def _area(proposal: LocalRegionProposal) -> int:
    return int(proposal.mask.sum())


def _checkpoint_kind(checkpoint: dict) -> str:
    if "loss" in checkpoint or "softmax_temperature" in checkpoint:
        return "candidate_listwise"
    return "hash_text_geometry"
