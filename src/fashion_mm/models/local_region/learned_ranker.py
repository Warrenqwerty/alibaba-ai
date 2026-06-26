from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


LEARNED_RANKER_CANDIDATE_REGIONS = (
    "whole_garment",
    "upper",
    "lower",
    "left",
    "right",
    "center",
    "neckline",
    "hem",
    "shoulder",
    "waist",
)


@dataclass(frozen=True)
class BoxCandidate:
    """One geometric candidate used by the lightweight learned ranker."""

    region: str
    box: tuple[float, float, float, float]


class HashingTextRegionScorer(nn.Module):
    """Small query-region scorer for weak 3.1.2 ranker training.

    This is intentionally dependency-light. It learns from query text, candidate
    region text, and normalized candidate geometry. It is a bridge toward a
    stronger CLIP/DINOv2-style text-region scorer.
    """

    def __init__(
        self,
        num_buckets: int = 256,
        hidden_dim: int = 128,
        geometry_dim: int = 6,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        input_dim = num_buckets * 2 + geometry_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class CandidateListwiseScorer(nn.Module):
    """Small listwise scorer for candidate-level weak supervision."""

    def __init__(
        self,
        num_buckets: int = 256,
        hidden_dim: int = 160,
        geometry_dim: int = 6,
        context_dim: int = 8,
        prior_dim: int = 3,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        input_dim = num_buckets * 3 + geometry_dim + context_dim + prior_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def build_pair_feature(
    query: str,
    candidate: BoxCandidate,
    garment_box: tuple[float, float, float, float],
    *,
    num_buckets: int = 256,
) -> torch.Tensor:
    """Build one ranker feature vector from text and candidate geometry."""
    return torch.cat(
        [
            hash_text(query, num_buckets),
            hash_text(candidate.region, num_buckets),
            torch.tensor(
                normalized_box_features(garment_box, candidate.box),
                dtype=torch.float32,
            ),
        ]
    )


def build_candidate_record_feature(
    query: str,
    candidate_region: str,
    garment_box: tuple[float, float, float, float],
    candidate_box: tuple[float, float, float, float],
    parsed_region: str | None,
    category_text: str | None = None,
    *,
    num_buckets: int = 256,
) -> torch.Tensor:
    """Build a candidate-level feature vector for listwise ranking."""
    return torch.cat(
        [
            hash_text(query, num_buckets),
            hash_text(candidate_region, num_buckets),
            torch.tensor(
                normalized_box_features(garment_box, candidate_box),
                dtype=torch.float32,
            ),
            torch.tensor(
                box_context_features(garment_box, candidate_box),
                dtype=torch.float32,
            ),
            torch.tensor(
                candidate_prior_features(candidate_region, parsed_region),
                dtype=torch.float32,
            ),
            hash_text(category_text or "", num_buckets),
        ]
    )


def box_context_features(
    garment_box: tuple[float, float, float, float],
    candidate_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return absolute box context features that vary by image instance."""
    gx1, gy1, gx2, gy2 = garment_box
    cx1, cy1, cx2, cy2 = candidate_box
    garment_width = max(gx2 - gx1, 1.0)
    garment_height = max(gy2 - gy1, 1.0)
    candidate_width = max(cx2 - cx1, 1.0)
    candidate_height = max(cy2 - cy1, 1.0)
    return (
        garment_width / 1000.0,
        garment_height / 1000.0,
        (garment_width * garment_height) / 1_000_000.0,
        math.log(garment_width / garment_height),
        candidate_width / 1000.0,
        candidate_height / 1000.0,
        (candidate_width * candidate_height) / 1_000_000.0,
        math.log(candidate_width / candidate_height),
    )


def candidate_prior_features(
    candidate_region: str,
    parsed_region: str | None,
) -> tuple[float, float, float]:
    """Return query-parser prior features for a candidate region."""
    exact_match = float(parsed_region is not None and candidate_region == parsed_region)
    side_stripped = candidate_region.removeprefix("left_").removeprefix("right_")
    side_match = float(parsed_region is not None and side_stripped == parsed_region)
    is_generic = float(
        candidate_region
        in {"whole_garment", "upper", "lower", "left", "right", "center", "waist"}
    )
    return (exact_match, side_match, is_generic)


def hash_text(text: str, num_buckets: int = 256) -> torch.Tensor:
    """Hash text characters into a normalized bag-of-characters vector."""
    vector = torch.zeros(num_buckets, dtype=torch.float32)
    stripped = text.strip()
    if not stripped:
        return vector
    for char in stripped:
        vector[ord(char) % num_buckets] += 1.0
    return vector / max(float(len(stripped)), 1.0)


def candidate_boxes_from_garment(
    garment_box: tuple[float, float, float, float],
    regions: tuple[str, ...] = LEARNED_RANKER_CANDIDATE_REGIONS,
) -> list[BoxCandidate]:
    """Generate box candidates from a garment box for weak ranker training."""
    x1, y1, x2, y2 = garment_box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)

    boxes = {
        "whole_garment": (x1, y1, x2, y2),
        "upper": (x1, y1, x2, y1 + height * 0.45),
        "lower": (x1, y1 + height * 0.55, x2, y2),
        "left": (x1, y1, x1 + width * 0.45, y2),
        "right": (x1 + width * 0.55, y1, x2, y2),
        "center": (
            x1 + width * 0.25,
            y1 + height * 0.25,
            x1 + width * 0.75,
            y1 + height * 0.75,
        ),
        "neckline": (
            x1 + width * 0.16,
            y1,
            x1 + width * 0.84,
            y1 + height * 0.22,
        ),
        "hem": (x1, y1 + height * 0.75, x2, y2),
        "shoulder": (x1, y1, x2, y1 + height * 0.22),
        "waist": (x1, y1 + height * 0.45, x2, y1 + height * 0.60),
    }
    return [
        BoxCandidate(region=region, box=boxes[region])
        for region in regions
        if region in boxes
    ]


def normalized_box_features(
    garment_box: tuple[float, float, float, float],
    candidate_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Return candidate geometry normalized inside the garment box."""
    gx1, gy1, gx2, gy2 = garment_box
    cx1, cy1, cx2, cy2 = candidate_box
    garment_width = max(gx2 - gx1, 1.0)
    garment_height = max(gy2 - gy1, 1.0)
    nx1 = (cx1 - gx1) / garment_width
    ny1 = (cy1 - gy1) / garment_height
    nx2 = (cx2 - gx1) / garment_width
    ny2 = (cy2 - gy1) / garment_height
    width = max(nx2 - nx1, 0.0)
    height = max(ny2 - ny1, 0.0)
    return (nx1, ny1, nx2, ny2, width, height)


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
