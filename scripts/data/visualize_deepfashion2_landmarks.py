from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from fashion_mm.data_loaders.deepfashion2_landmarks import FashionLandmark
from fashion_mm.data_loaders.deepfashion2_landmarks import parse_landmarks


VISIBLE_COLOR = (0, 220, 80)
OCCLUDED_COLOR = (255, 170, 0)
UNLABELED_COLOR = (160, 160, 160)
BBOX_COLOR = (80, 160, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DeepFashion2 item landmarks with point indices."
    )
    parser.add_argument("--image", required=True, help="Path to a DeepFashion2 image.")
    parser.add_argument(
        "--annotation",
        required=True,
        help="Path to the matching DeepFashion2 annotation JSON.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--item",
        default=None,
        help="Optional item key, e.g. item1. Draw all items when omitted.",
    )
    parser.add_argument("--show-unlabeled", action="store_true")
    return parser.parse_args()


def iter_items(annotation: dict[str, Any], item_key: str | None):
    for key in sorted(annotation):
        if not key.startswith("item"):
            continue
        if item_key is not None and key != item_key:
            continue
        yield key, annotation[key]


def landmark_color(landmark: FashionLandmark) -> tuple[int, int, int]:
    if landmark.is_visible:
        return VISIBLE_COLOR
    if landmark.is_occluded:
        return OCCLUDED_COLOR
    return UNLABELED_COLOR


def draw_landmark(
    draw: ImageDraw.ImageDraw,
    landmark: FashionLandmark,
    font: ImageFont.ImageFont,
) -> None:
    color = landmark_color(landmark)
    radius = 4 if landmark.is_labeled else 3
    x = float(landmark.x)
    y = float(landmark.y)
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color,
        outline=(0, 0, 0),
        width=1,
    )
    draw.text((x + 5, y - 5), str(landmark.index), fill=color, font=font)


def draw_item(
    draw: ImageDraw.ImageDraw,
    item_key: str,
    item: dict[str, Any],
    font: ImageFont.ImageFont,
    show_unlabeled: bool,
) -> None:
    box = item.get("bounding_box")
    if box:
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=2)
        label = f"{item_key} {item.get('category_name', item.get('category_id'))}"
        draw.text((x1, max(0, y1 - 16)), label, fill=BBOX_COLOR, font=font)

    landmarks = parse_landmarks(item.get("landmarks", []))
    for landmark in landmarks:
        if not show_unlabeled and not landmark.is_labeled:
            continue
        draw_landmark(draw, landmark, font)


def visualize(
    image_path: Path,
    annotation_path: Path,
    output_path: Path,
    item_key: str | None,
    show_unlabeled: bool,
) -> None:
    image = Image.open(image_path).convert("RGB")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    drawn = 0
    for key, item in iter_items(annotation, item_key):
        draw_item(draw, key, item, font, show_unlabeled)
        drawn += 1
    if drawn == 0:
        raise ValueError(f"No matching item found: {item_key or 'all items'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    visualize(
        image_path=Path(args.image),
        annotation_path=Path(args.annotation),
        output_path=Path(args.output),
        item_key=args.item,
        show_unlabeled=args.show_unlabeled,
    )


if __name__ == "__main__":
    main()
