from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Literal

import numpy as np


SUPPORTED_RULE_REGIONS = {"neckline", "cuff", "hem", "shoulder", "waist", "pattern"}
UNSUPPORTED_RULE_REGIONS = {"pocket", "decoration"}
OPEN_VOCAB_CANDIDATE_REGIONS = (
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
    "left_cuff",
    "right_cuff",
    "left_pocket",
    "right_pocket",
    "zipper",
    "button",
    "pattern",
    "decoration",
)


@dataclass(frozen=True)
class LocalRegionProposal:
    """Rule-based local-region localization result inside one garment mask."""

    region: str
    mask: np.ndarray
    box: tuple[float, float, float, float] | None
    confidence: float
    source: Literal["rule_baseline", "landmark_pseudo_label", "open_vocab_candidate"]
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


def generate_open_vocab_candidates(
    garment_mask: np.ndarray,
    garment_box: tuple[float, float, float, float] | list[float],
    regions: Iterable[str] = OPEN_VOCAB_CANDIDATE_REGIONS,
    category_text: str | None = None,
) -> list[LocalRegionProposal]:
    """Generate generic region candidates for language-guided matching.

    These candidates are intentionally not limited to training-time part names.
    A later DINOv2/CLIP ranker can score them against arbitrary query text.
    """
    mask = np.asarray(garment_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"garment_mask must be 2D, got shape {mask.shape}")

    candidates: list[LocalRegionProposal] = []
    for region in regions:
        proposal = _propose_open_vocab_candidate(
            mask,
            garment_box,
            region,
            category_text=category_text,
        )
        if proposal.status == "ok":
            candidates.append(proposal)
    return candidates


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
        wx1 = x1 + int(box_width * 0.16)
        wx2 = x1 + int(box_width * 0.84)
        wy1 = y1
        wy2 = y1 + int(box_height * 0.22)
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
        wy2 = y1 + int(box_height * 0.22)
        window[wy1:wy2, x1:x2] = True
    elif region == "cuff":
        wy1 = y1 + int(box_height * 0.25)
        wy2 = y1 + int(box_height * 0.90)
        left_x2 = x1 + int(box_width * 0.22)
        right_x1 = x1 + int(box_width * 0.78)
        window[wy1:wy2, x1:left_x2] = True
        window[wy1:wy2, right_x1:x2] = True
    return window


def _propose_open_vocab_candidate(
    garment_mask: np.ndarray,
    garment_box: tuple[float, float, float, float] | list[float],
    region: str,
    category_text: str | None = None,
) -> LocalRegionProposal:
    if region == "whole_garment":
        region_mask = garment_mask.copy()
    elif region == "pattern":
        region_mask = garment_mask.copy()
    elif region in {"neckline", "hem", "shoulder"}:
        baseline_region = "neckline" if region == "neckline" else region
        proposal = propose_local_region(garment_mask, garment_box, baseline_region)
        return _as_open_vocab_source(proposal)
    elif region == "waist":
        region_mask = garment_mask & _waist_window(
            garment_box,
            garment_mask.shape,
            category_text=category_text,
        )
    elif region in {"left_cuff", "right_cuff"}:
        side = "left" if region == "left_cuff" else "right"
        broad_mask = garment_mask & _single_side_cuff_window(
            garment_box,
            garment_mask.shape,
            side=side,
        )
        region_mask = _terminal_cuff_mask(broad_mask, side=side)
    elif region in {"left_pocket", "right_pocket"}:
        region_mask = garment_mask & _single_side_pocket_window(
            garment_box,
            garment_mask.shape,
            side="left" if region == "left_pocket" else "right",
        )
    elif region == "zipper":
        region_mask = garment_mask & _vertical_trim_window(garment_box, garment_mask.shape)
    elif region == "button":
        region_mask = garment_mask & _button_placket_window(garment_box, garment_mask.shape)
    elif region == "decoration":
        region_mask = garment_mask & _decoration_window(garment_box, garment_mask.shape)
    else:
        x1, y1, x2, y2 = _clip_box(garment_box, garment_mask.shape)
        region_mask = garment_mask & _generic_spatial_window(
            region,
            x1,
            y1,
            x2,
            y2,
            garment_mask.shape,
        )

    box = _mask_to_box(region_mask)
    if box is None:
        return _empty_result(region, garment_mask.shape, "empty_region", "empty candidate")
    return LocalRegionProposal(
        region=region,
        mask=region_mask,
        box=box,
        confidence=_candidate_confidence(region),
        source="open_vocab_candidate",
        status="ok",
    )


def _as_open_vocab_source(proposal: LocalRegionProposal) -> LocalRegionProposal:
    return LocalRegionProposal(
        region=proposal.region,
        mask=proposal.mask,
        box=proposal.box,
        confidence=proposal.confidence,
        source="open_vocab_candidate",
        status=proposal.status,
        reason=proposal.reason,
    )


def _generic_spatial_window(
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

    if region == "upper":
        window[y1 : y1 + int(box_height * 0.45), x1:x2] = True
    elif region == "lower":
        window[y1 + int(box_height * 0.55) : y2, x1:x2] = True
    elif region == "left":
        window[y1:y2, x1 : x1 + int(box_width * 0.45)] = True
    elif region == "right":
        window[y1:y2, x1 + int(box_width * 0.55) : x2] = True
    elif region == "center":
        wx1 = x1 + int(box_width * 0.25)
        wx2 = x1 + int(box_width * 0.75)
        wy1 = y1 + int(box_height * 0.25)
        wy2 = y1 + int(box_height * 0.75)
        window[wy1:wy2, wx1:wx2] = True
    return window


def _single_side_cuff_window(
    box: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
    side: Literal["left", "right"],
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_box(box, image_shape)
    height, width = image_shape
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    window = np.zeros((height, width), dtype=bool)
    wy1 = y1 + int(box_height * 0.18)
    wy2 = y1 + int(box_height * 0.95)
    if side == "left":
        window[wy1:wy2, x1 : x1 + int(box_width * 0.34)] = True
    else:
        window[wy1:wy2, x1 + int(box_width * 0.66) : x2] = True
    return window


def _terminal_cuff_mask(mask: np.ndarray, side: Literal["left", "right"]) -> np.ndarray:
    """Keep the sleeve end instead of the whole side sleeve strip."""
    box = _mask_to_box(mask)
    if box is None:
        return mask
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    window = np.zeros(mask.shape, dtype=bool)
    if height >= width * 1.25:
        terminal_y1 = y1 + int(height * 0.68)
        window[terminal_y1:y2, x1:x2] = True
    elif side == "left":
        terminal_x2 = x1 + int(width * 0.42)
        window[y1:y2, x1:terminal_x2] = True
    else:
        terminal_x1 = x1 + int(width * 0.58)
        window[y1:y2, terminal_x1:x2] = True
    refined = mask & window
    return refined if refined.any() else mask


def _waist_window(
    box: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
    category_text: str | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_box(box, image_shape)
    height, width = image_shape
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    window = np.zeros((height, width), dtype=bool)
    category = (category_text or "").lower()
    if any(term in category for term in ("pants", "trousers", "shorts", "skirt")):
        start, end = 0.06, 0.26
    elif "dress" in category:
        start, end = 0.26, 0.46
    elif any(term in category for term in ("top", "shirt", "sleeve", "outerwear", "coat")):
        start, end = 0.52, 0.74
    else:
        start, end = 0.35, 0.55
    wy1 = y1 + int(box_height * start)
    wy2 = y1 + int(box_height * end)
    wx1 = x1 + int(box_width * 0.06)
    wx2 = x1 + int(box_width * 0.94)
    window[wy1:wy2, wx1:wx2] = True
    return window


def _single_side_pocket_window(
    box: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
    side: Literal["left", "right"],
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_box(box, image_shape)
    height, width = image_shape
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    window = np.zeros((height, width), dtype=bool)
    wy1 = y1 + int(box_height * 0.18)
    wy2 = y1 + int(box_height * 0.48)
    if side == "left":
        wx1 = x1 + int(box_width * 0.08)
        wx2 = x1 + int(box_width * 0.36)
    else:
        wx1 = x1 + int(box_width * 0.64)
        wx2 = x1 + int(box_width * 0.92)
    window[wy1:wy2, wx1:wx2] = True
    return window


def _vertical_trim_window(
    box: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_box(box, image_shape)
    height, width = image_shape
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    window = np.zeros((height, width), dtype=bool)
    wx1 = x1 + int(box_width * 0.45)
    wx2 = x1 + int(box_width * 0.55)
    wy1 = y1 + int(box_height * 0.12)
    wy2 = y1 + int(box_height * 0.88)
    window[wy1:wy2, wx1:wx2] = True
    return window


def _button_placket_window(
    box: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_box(box, image_shape)
    height, width = image_shape
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    window = np.zeros((height, width), dtype=bool)
    wx1 = x1 + int(box_width * 0.43)
    wx2 = x1 + int(box_width * 0.57)
    wy1 = y1 + int(box_height * 0.18)
    wy2 = y1 + int(box_height * 0.72)
    window[wy1:wy2, wx1:wx2] = True
    return window


def _decoration_window(
    box: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_box(box, image_shape)
    height, width = image_shape
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    window = np.zeros((height, width), dtype=bool)
    wx1 = x1 + int(box_width * 0.18)
    wx2 = x1 + int(box_width * 0.82)
    wy1 = y1 + int(box_height * 0.15)
    wy2 = y1 + int(box_height * 0.65)
    window[wy1:wy2, wx1:wx2] = True
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


def _candidate_confidence(region: str) -> float:
    return {
        "whole_garment": 0.45,
        "upper": 0.50,
        "lower": 0.50,
        "left": 0.48,
        "right": 0.48,
        "center": 0.46,
        "pattern": 0.50,
        "left_cuff": 0.55,
        "right_cuff": 0.55,
        "left_pocket": 0.52,
        "right_pocket": 0.52,
        "zipper": 0.50,
        "button": 0.50,
        "decoration": 0.48,
    }.get(region, _rule_confidence(region))
