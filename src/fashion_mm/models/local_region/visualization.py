from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fashion_mm.models.local_region.predictor import LocalRegionResult


def draw_local_region_result(
    image_path: Path,
    result: LocalRegionResult,
    output_path: Path,
) -> None:
    """Draw selected garment and localized language-guided region."""
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    if result.selected_instance is not None:
        garment_layer = Image.new("RGBA", image.size, (0, 120, 255, 70))
        garment_mask = Image.fromarray(
            result.selected_instance.mask.astype(np.uint8) * 255
        )
        overlay = Image.composite(garment_layer, overlay, garment_mask)
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = result.selected_instance.box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255, 255), width=3)
        draw.text(
            (x1, max(0, y1 - 18)),
            f"Garment {result.selected_instance.label_name}",
            fill=(0, 80, 220, 255),
        )

    if result.proposal is not None and result.proposal.proposal.box is not None:
        proposal = result.proposal.proposal
        region_layer = Image.new("RGBA", image.size, (255, 100, 0, 130))
        region_mask = Image.fromarray(proposal.mask.astype(np.uint8) * 255)
        overlay = Image.composite(region_layer, overlay, region_mask)
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = proposal.box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 0, 255), width=3)
        draw.text(
            (x1, max(0, y1 - 18)),
            f"Region {proposal.region} {result.proposal.score:.2f}",
            fill=(220, 60, 0, 255),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)
