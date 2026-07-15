from __future__ import annotations

from typing import Any

import numpy as np


def query_wearer_side(query_text: str) -> str | None:
    """Return the garment/wearer side named by a Chinese query."""
    if "左" in query_text:
        return "left"
    if "右" in query_text:
        return "right"
    return None


def desired_image_side(wearer_side: str) -> str:
    """Map garment/wearer side to image side for frontal or flat-lay views."""
    if wearer_side == "left":
        return "right"
    if wearer_side == "right":
        return "left"
    raise ValueError(f"Unsupported wearer side: {wearer_side}")


def detection_image_side(detection: dict[str, Any], image_width: int) -> str:
    """Return the image half containing a detection-box center."""
    x1, _, x2, _ = [float(value) for value in detection["bbox"]]
    return "left" if (x1 + x2) * 0.5 < image_width * 0.5 else "right"


def select_wearer_side_detection(
    detections: list[dict[str, Any]],
    *,
    query_text: str,
    image_width: int,
    min_score_ratio: float,
) -> tuple[dict[str, Any] | None, str]:
    """Select a credible detection on the query's garment/wearer side."""
    if not detections:
        return None, "no_detection"
    baseline = max(detections, key=lambda detection: float(detection["score"]))
    wearer_side = query_wearer_side(query_text)
    if wearer_side is None:
        return baseline, "query_has_no_side"
    if not 0.0 <= min_score_ratio <= 1.0:
        raise ValueError("min_score_ratio must be between 0 and 1")

    target_image_side = desired_image_side(wearer_side)
    minimum_score = float(baseline["score"]) * min_score_ratio
    compatible = [
        detection
        for detection in detections
        if float(detection["score"]) >= minimum_score
        and detection_image_side(detection, image_width) == target_image_side
    ]
    if not compatible:
        return baseline, "no_credible_side_candidate"
    return (
        max(compatible, key=lambda detection: float(detection["score"])),
        "side_candidate",
    )


def grounding_box_mask_coverage(
    box: list[float] | tuple[float, float, float, float],
    garment_mask: np.ndarray,
) -> float:
    """Return the fraction of a grounding box that lies inside a garment mask."""
    mask = np.asarray(garment_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"garment_mask must be 2D, got shape {mask.shape}")
    x1, y1, x2, y2 = [float(value) for value in box]
    left = max(0, min(mask.shape[1], int(np.floor(x1))))
    top = max(0, min(mask.shape[0], int(np.floor(y1))))
    right = max(0, min(mask.shape[1], int(np.ceil(x2))))
    bottom = max(0, min(mask.shape[0], int(np.ceil(y2))))
    if right <= left or bottom <= top:
        return 0.0
    return float(mask[top:bottom, left:right].mean())


def filter_grounding_detections_to_garment(
    detections: list[dict[str, Any]],
    garment_mask: np.ndarray,
    *,
    min_mask_coverage: float,
) -> list[dict[str, Any]]:
    """Keep detections with sufficient area inside the selected garment mask."""
    if not 0.0 <= min_mask_coverage <= 1.0:
        raise ValueError("min_mask_coverage must be between 0 and 1")
    filtered = []
    for detection in detections:
        box = detection.get("bbox")
        if box is None:
            continue
        coverage = grounding_box_mask_coverage(box, garment_mask)
        if coverage < min_mask_coverage:
            continue
        filtered.append({**detection, "garment_mask_coverage": coverage})
    return filtered
