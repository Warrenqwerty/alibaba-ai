from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from fashion_mm.utils.image_io import ImageInput, load_rgb_image


MaskInput = str | Path | Image.Image | np.ndarray


@dataclass(frozen=True)
class MaskedRegionCrop:
    image: Image.Image
    box: tuple[int, int, int, int]
    mask_area: int
    mask_coverage: float


def prepare_masked_region(
    image: ImageInput,
    mask: MaskInput,
    *,
    padding_fraction: float = 0.08,
    fill_rgb: tuple[int, int, int] = (124, 116, 104),
) -> tuple[Image.Image, MaskedRegionCrop]:
    """Apply a target mask, crop its padded bbox, and fill outside pixels."""
    if padding_fraction < 0.0:
        raise ValueError("padding_fraction cannot be negative.")
    rgb_image = load_rgb_image(image)
    mask_array = load_region_mask(mask, expected_size=rgb_image.size)
    ys, xs = np.nonzero(mask_array)
    if len(xs) == 0:
        raise ValueError("Target region mask is empty.")

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    region_size = max(x2 - x1, y2 - y1)
    padding = round(region_size * padding_fraction)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(rgb_image.width, x2 + padding)
    y2 = min(rgb_image.height, y2 + padding)

    image_array = np.asarray(rgb_image, dtype=np.uint8).copy()
    image_array[~mask_array] = np.asarray(fill_rgb, dtype=np.uint8)
    masked_image = Image.fromarray(image_array, mode="RGB")
    crop = masked_image.crop((x1, y1, x2, y2))
    mask_area = int(mask_array.sum())
    crop_area = max((x2 - x1) * (y2 - y1), 1)
    return rgb_image, MaskedRegionCrop(
        image=crop,
        box=(x1, y1, x2, y2),
        mask_area=mask_area,
        mask_coverage=mask_area / crop_area,
    )


def load_region_mask(mask: MaskInput, *, expected_size: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        mask_array = np.asarray(mask)
    elif isinstance(mask, Image.Image):
        mask_array = np.asarray(mask.convert("L"))
    else:
        path = Path(mask)
        if not path.is_file():
            raise FileNotFoundError(f"Region mask not found: {path}")
        with Image.open(path) as mask_image:
            mask_array = np.asarray(mask_image.convert("L"))

    if mask_array.ndim != 2:
        raise ValueError(f"Region mask must be 2D, got {mask_array.shape}")
    expected_shape = (expected_size[1], expected_size[0])
    if mask_array.shape != expected_shape:
        raise ValueError(
            f"Region mask shape {mask_array.shape} does not match image shape "
            f"{expected_shape}."
        )
    return np.asarray(mask_array > 0, dtype=bool)
