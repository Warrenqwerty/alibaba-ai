from __future__ import annotations

from dataclasses import dataclass

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
    "pattern": ("图案", "印花", "花纹", "纹理", "碎花", "条纹", "格纹"),
}

REGION_EQUIVALENTS = {
    "cuff": ("left_cuff", "right_cuff"),
    "decoration": ("center", "upper", "whole_garment"),
    "pocket": ("left", "right", "center"),
}


@dataclass(frozen=True)
class RankedRegionCandidate:
    """A candidate local region with a text-matching score."""

    proposal: LocalRegionProposal
    score: float
    reason: str

    def to_dict(self, include_mask: bool = False) -> dict:
        payload = self.proposal.to_dict(include_mask=include_mask)
        payload["match_score"] = self.score
        payload["match_reason"] = self.reason
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
    ) -> list[RankedRegionCandidate]:
        ranked = [
            RankedRegionCandidate(
                proposal=candidate,
                score=self._score_candidate(query, candidate),
                reason=self._reason(query, candidate),
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
            if region in {"left", "left_cuff"}:
                score += 0.9
            if region in {"right", "right_cuff"}:
                score -= 0.4
        if any(term in text for term in ("右边", "右侧", "右面", "右")):
            if region in {"right", "right_cuff"}:
                score += 0.9
            if region in {"left", "left_cuff"}:
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
        if any(term in text for term in ("扣子", "纽扣", "拉链", "装饰", "珠片", "亮片")):
            return 0.45 if region in {"center", "upper", "whole_garment"} else 0.0
        return 0.0

    def _reason(self, query: ParsedRegionQuery, candidate: LocalRegionProposal) -> str:
        if query.region == candidate.region:
            return "exact parsed region match"
        if query.region in REGION_EQUIVALENTS and candidate.region in REGION_EQUIVALENTS[query.region]:
            return "parsed region matched equivalent candidate"
        if any(hint in query.query for hint in REGION_TEXT_HINTS.get(candidate.region, ())):
            return "raw query keyword matched candidate"
        return "generic open-vocabulary candidate"


def _area(proposal: LocalRegionProposal) -> int:
    return int(proposal.mask.sum())
