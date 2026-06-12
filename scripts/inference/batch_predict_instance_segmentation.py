from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.instance_segmentation.result import SegmentationResult
from fashion_mm.utils.config import load_config
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch fashion instance segmentation inference."
    )
    parser.add_argument("--image-dir", required=True, help="Directory of test images.")
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/test_predictions.json")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--vis-count", type=int, default=50)
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def collect_images(image_dir: Path, max_images: int | None) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")
    return image_paths


def draw_prediction(
    image_path: Path,
    result: SegmentationResult,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (0, 255, 80, 100),
        (255, 80, 0, 100),
        (80, 160, 255, 100),
        (255, 220, 0, 100),
        (220, 80, 255, 100),
    ]

    for index, instance in enumerate(result.instances):
        color = colors[index % len(colors)]
        mask_image = Image.fromarray(instance.mask.astype("uint8") * 255)
        color_layer = Image.new("RGBA", image.size, color)
        overlay = Image.composite(color_layer, overlay, mask_image)
        x1, y1, x2, y2 = instance.box
        draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=3)
        text = f"{instance.label_name} {instance.score:.2f}"
        draw.text((x1, max(0, y1 - 16)), text, fill=color[:3] + (255,))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def main() -> None:
    args = parse_args()
    image_paths = collect_images(Path(args.image_dir), args.max_images)
    config = load_config(args.model_config)
    predictor = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    predictions = []
    visualized = 0
    for index, image_path in enumerate(image_paths, start=1):
        result = predictor.predict(image_path)
        predictions.append(
            {
                "file_name": image_path.name,
                **result.to_dict(include_masks=args.include_masks),
            }
        )

        if args.vis_dir and visualized < args.vis_count:
            draw_prediction(
                image_path,
                result,
                Path(args.vis_dir) / f"{image_path.stem}_pred.jpg",
            )
            visualized += 1

        if index % args.log_interval == 0 or index == len(image_paths):
            LOGGER.info("predicted %s/%s images", index, len(image_paths))

    output = {
        "checkpoint": args.checkpoint,
        "image_dir": str(Path(args.image_dir)),
        "num_images": len(image_paths),
        "include_masks": bool(args.include_masks),
        "predictions": predictions,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {key: output[key] for key in output if key != "predictions"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
