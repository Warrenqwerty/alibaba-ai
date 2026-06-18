from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fashion_mm.models.local_region import propose_region_from_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DeepFashion2 landmark-derived local region masks."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--item", default=None, help="Optional item key, e.g. item1.")
    parser.add_argument(
        "--region",
        default="neckline",
        choices=["neckline", "hem", "shoulder"],
    )
    return parser.parse_args()


def polygon_to_mask(
    segmentation: list[list[float]],
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) >= 6:
            points = [
                (float(polygon[index]), float(polygon[index + 1]))
                for index in range(0, len(polygon), 2)
            ]
            draw.polygon(points, outline=1, fill=1)
    return np.asarray(mask, dtype=bool)


def iter_items(annotation: dict[str, Any], item_key: str | None):
    for key in sorted(annotation):
        if not key.startswith("item"):
            continue
        if item_key is not None and key != item_key:
            continue
        yield key, annotation[key]


def draw_overlay(
    image_path: Path,
    item_key: str,
    item: dict[str, Any],
    region: str,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    garment_mask = polygon_to_mask(item.get("segmentation", []), image.size)
    proposal = propose_region_from_landmarks(
        garment_mask,
        item["bounding_box"],
        item.get("landmarks", []),
        region,
    )

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    garment_layer = Image.new("RGBA", image.size, (0, 120, 255, 65))
    garment_mask_image = Image.fromarray(garment_mask.astype(np.uint8) * 255)
    overlay = Image.composite(garment_layer, overlay, garment_mask_image)

    region_layer = Image.new("RGBA", image.size, (255, 100, 0, 140))
    region_mask_image = Image.fromarray(proposal.mask.astype(np.uint8) * 255)
    overlay = Image.composite(region_layer, overlay, region_mask_image)

    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = [float(value) for value in item["bounding_box"]]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255, 255), width=2)
    if proposal.box is not None:
        rx1, ry1, rx2, ry2 = proposal.box
        draw.rectangle([rx1, ry1, rx2, ry2], outline=(255, 80, 0, 255), width=3)
    label = f"{item_key} {item.get('category_name')} {region} {proposal.source}"
    draw.text((x1, max(0, y1 - 18)), label, fill=(255, 80, 0, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def visualize(
    image_path: Path,
    annotation_path: Path,
    output_path: Path,
    item_key: str | None,
    region: str,
) -> None:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    matches = list(iter_items(annotation, item_key))
    if not matches:
        raise ValueError(f"No matching item found: {item_key or 'all items'}")

    key, item = matches[0]
    draw_overlay(image_path, key, item, region, output_path)


def main() -> None:
    args = parse_args()
    visualize(
        image_path=Path(args.image),
        annotation_path=Path(args.annotation),
        output_path=Path(args.output),
        item_key=args.item,
        region=args.region,
    )


if __name__ == "__main__":
    main()
