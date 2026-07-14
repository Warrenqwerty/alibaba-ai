from __future__ import annotations

from typing import Any

import numpy as np


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
