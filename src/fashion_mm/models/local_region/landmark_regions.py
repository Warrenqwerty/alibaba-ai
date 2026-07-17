from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageDraw

from fashion_mm.data_loaders.deepfashion2_landmarks import FashionLandmark
from fashion_mm.data_loaders.deepfashion2_landmarks import parse_landmarks
from fashion_mm.models.local_region.proposal import LocalRegionProposal
from fashion_mm.models.local_region.proposal import propose_local_region


LANDMARK_REGION_GROUPS = {
    "neckline": (1, 2, 3, 4, 5, 6),
    "shoulder": (1, 2, 6, 7, 28, 29, 30),
    "hem": (15, 16, 18, 19, 20, 21),
}

# DeepFashion2 uses a different local landmark order for each garment category.
# These endpoint pairs follow the official category contours and isolate the
# sleeve terminal rather than the complete sleeve. Lower-body points 1-3 form
# the waistband contour for shorts, trousers, and skirts.
CATEGORY_LANDMARK_REGION_GROUPS = {
    1: {"left_cuff": (9, 10), "right_cuff": (22, 23)},
    2: {"left_cuff": (11, 12), "right_cuff": (28, 29)},
    3: {"left_cuff": (9, 10), "right_cuff": (22, 23)},
    4: {"left_cuff": (11, 12), "right_cuff": (28, 29)},
    7: {"waist": (1, 2, 3)},
    8: {"waist": (1, 2, 3)},
    9: {"waist": (1, 2, 3)},
    10: {"left_cuff": (9, 10), "right_cuff": (26, 27)},
    11: {"left_cuff": (11, 12), "right_cuff": (32, 33)},
}


def propose_region_from_landmarks(
    garment_mask: np.ndarray,
    garment_box: tuple[float, float, float, float] | list[float],
    raw_landmarks: Iterable[float | int],
    region: str,
    *,
    category_id: int | None = None,
    min_points: int = 2,
    padding_ratio: float = 0.08,
) -> LocalRegionProposal:
    """Create a local-region pseudo label from DeepFashion2 landmarks."""
    mask = np.asarray(garment_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"garment_mask must be 2D, got shape {mask.shape}")

    try:
        normalized_category_id = int(category_id) if category_id is not None else -1
    except (TypeError, ValueError):
        normalized_category_id = -1
    category_regions = CATEGORY_LANDMARK_REGION_GROUPS.get(
        normalized_category_id,
        {},
    )
    target_indices = category_regions.get(region, LANDMARK_REGION_GROUPS.get(region))
    if target_indices is None:
        return propose_local_region(mask, garment_box, region)

    landmarks = parse_landmarks(raw_landmarks)
    selected = [
        landmark
        for landmark in landmarks
        if landmark.index in target_indices and landmark.is_labeled
    ]
    if len(selected) < min_points:
        return propose_local_region(mask, garment_box, region)

    window = _landmark_window(
        selected,
        garment_box,
        mask.shape,
        padding_ratio=padding_ratio,
    )
    region_mask = mask & window
    box = _mask_to_box(region_mask)
    if box is None:
        return propose_local_region(mask, garment_box, region)

    return LocalRegionProposal(
        region=region,
        mask=region_mask,
        box=box,
        confidence=_landmark_confidence(region, selected),
        source="landmark_pseudo_label",
        status="ok",
    )


def _landmark_window(
    landmarks: list[FashionLandmark],
    garment_box: tuple[float, float, float, float] | list[float],
    image_shape: tuple[int, int],
    padding_ratio: float,
) -> np.ndarray:
    height, width = image_shape
    gx1, gy1, gx2, gy2 = [float(value) for value in garment_box]
    garment_width = max(gx2 - gx1, 1.0)
    garment_height = max(gy2 - gy1, 1.0)
    radius = max(3, int(round(min(garment_width, garment_height) * padding_ratio)))
    points = [
        (
            max(0, min(width - 1, int(round(landmark.x)))),
            max(0, min(height - 1, int(round(landmark.y)))),
        )
        for landmark in landmarks
    ]

    window = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(window)
    if len(points) >= 3:
        draw.polygon(_convex_hull(points), outline=1, fill=1)

    sorted_points = sorted(points)
    if len(sorted_points) >= 2:
        draw.line(sorted_points, fill=1, width=max(1, radius * 2))

    for x, y in points:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=1,
            fill=1,
        )
    return np.asarray(window, dtype=bool)


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(
        origin: tuple[int, int],
        point_a: tuple[int, int],
        point_b: tuple[int, int],
    ) -> int:
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (
            point_a[1] - origin[1]
        ) * (point_b[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[int, int]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _mask_to_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _landmark_confidence(region: str, landmarks: list[FashionLandmark]) -> float:
    visible_ratio = sum(landmark.is_visible for landmark in landmarks) / len(landmarks)
    base = {
        "neckline": 0.82,
        "hem": 0.78,
        "shoulder": 0.74,
        "left_cuff": 0.78,
        "right_cuff": 0.78,
        "waist": 0.8,
    }.get(region, 0.65)
    return round(base * (0.75 + 0.25 * visible_ratio), 4)
