from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ImageInput = str | Path | Image.Image | np.ndarray


def load_rgb_image(image: ImageInput) -> Image.Image:
    """Load an image-like object and return a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim not in (2, 3):
            raise ValueError(f"Unsupported ndarray image shape: {image.shape}")
        array = image
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        if array.shape[2] == 4:
            return Image.fromarray(array).convert("RGBA").convert("RGB")
        if array.shape[2] == 3:
            return Image.fromarray(array).convert("RGB")
        raise ValueError(f"Unsupported ndarray channel count: {array.shape[2]}")

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save a boolean or 0/1 mask as an 8-bit grayscale PNG."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask_array.shape}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_array > 0).astype(np.uint8) * 255).save(output_path)


def pil_to_numpy(image: Image.Image) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Convert a PIL RGB image to a uint8 numpy array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)
