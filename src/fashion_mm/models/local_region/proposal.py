from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SUPPORTED_RULE_REGIONS = {"neckline", "cuff", "hem", "shoulder", "waist", "pattern"}
UNSUPPORTED_RULE_REGIONS = {"pocket", "decoration"}


@dataclass(frozen=True)
class LocalRegionProposal:
    """Rule-based local-region localization result inside one garment mask."""

    region: str
    mask: np.ndarray
    box: tuple[float, float, float, float] | None
    confidence: float
    source: Literal["rule_baseline", "landmark_pseudo_label"]
    status: Literal["ok", "empty_region", "unsupported_region", "unknown_region"]
    reason: str | None = None

    def to_dict(self, include_mask: bool = False) -> dict:
        payload = {
            "region": self.region,
            "box": list(self.box) if self.box is not None else None,
            "confidence": self.confidence,
            "source": self.source,
            "status": self.status,
            "reason": self.reason,
        }
        if include_mask:
            payload["mask"] = self.mask.astype(np.uint8).tolist()
        return payload


def propose_local_region(
    garment_mask: np.ndarray,
    garment_box: tuple[float, float, float, float] | list[float],
    region: str,
) -> LocalRegionProposal:
    """Create a local-region mask from one garment mask using baseline rules."""
    mask = np.asarray(garment_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"garment_mask must be 2D, got shape {mask.shape}")

    if region in UNSUPPORTED_RULE_REGIONS:
        return _empty_result(
            region,
            mask.shape,
            "unsupported_region",
            f"{region} requires a learned local detector or extra labels",
        )
    if region not in SUPPORTED_RULE_REGIONS:
        return _empty_result(region, mask.shape, "unknown_region", "unknown region")

    if region == "pattern":
        region_mask = mask.copy()
        confidence = 0.55
    else:
        x1, y1, x2, y2 = _clip_box(garment_box, mask.shape)
        window = _region_window(region, x1, y1, x2, y2, mask.shape)
        region_mask = mask & window
        confidence = _rule_confidence(region)

    box = _mask_to_box(region_mask)
    if box is None:
        return _empty_result(region, mask.shape, "empty_region", "rule produced empty mask")

    return LocalRegionProposal(
        region=region,
        mask=region_mask,
        box=box,
        confidence=confidence,
        source="rule_baseline",
        status="ok",
    )


def _region_window(
    region: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    window = np.zeros((height, width), dtype=bool)

    if region == "neckline":
        wx1 = x1 + int(box_width * 0.22)
        wx2 = x1 + int(box_width * 0.78)
        wy1 = y1
        wy2 = y1 + int(box_height * 0.18)
        window[wy1:wy2, wx1:wx2] = True
    elif region == "hem":
        wy1 = y1 + int(box_height * 0.75)
        window[wy1:y2, x1:x2] = True
    elif region == "waist":
        wy1 = y1 + int(box_height * 0.45)
        wy2 = y1 + int(box_height * 0.60)
        window[wy1:wy2, x1:x2] = True
    elif region == "shoulder":
        wy1 = y1
        wy2 = y1 + int(box_height * 0.35)
        left_x2 = x1 + int(box_width * 0.35)
        right_x1 = x1 + int(box_width * 0.65)
        window[wy1:wy2, x1:left_x2] = True
        window[wy1:wy2, right_x1:x2] = True
    elif region == "cuff":
        wy1 = y1 + int(box_height * 0.25)
        wy2 = y1 + int(box_height * 0.90)
        left_x2 = x1 + int(box_width * 0.22)
        right_x1 = x1 + int(box_width * 0.78)
        window[wy1:wy2, x1:left_x2] = True
        window[wy1:wy2, right_x1:x2] = True
    return window


def _clip_box(
    box: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )


def _mask_to_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _empty_result(
    region: str,
    image_shape: tuple[int, int],
    status: Literal["empty_region", "unsupported_region", "unknown_region"],
    reason: str,
) -> LocalRegionProposal:
    return LocalRegionProposal(
        region=region,
        mask=np.zeros(image_shape, dtype=bool),
        box=None,
        confidence=0.0,
        source="rule_baseline",
        status=status,
        reason=reason,
    )


def _rule_confidence(region: str) -> float:
    return {
        "neckline": 0.70,
        "hem": 0.72,
        "waist": 0.65,
        "shoulder": 0.62,
        "cuff": 0.58,
    }.get(region, 0.50)
