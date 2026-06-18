from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import LocalRegionResult
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3.1.2 language-guided local region localization."
    )
    parser.add_argument("image", help="Path to an RGB fashion image.")
    parser.add_argument("query", help="Natural-language local-region query.")
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--vis-output", default=None)
    parser.add_argument("--include-mask", action="store_true")
    return parser.parse_args()


def draw_local_region_result(
    image_path: Path,
    result: LocalRegionResult,
    output_path: Path,
) -> None:
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

    if result.proposal is not None and result.proposal.box is not None:
        region_layer = Image.new("RGBA", image.size, (255, 100, 0, 130))
        region_mask = Image.fromarray(result.proposal.mask.astype(np.uint8) * 255)
        overlay = Image.composite(region_layer, overlay, region_mask)
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = result.proposal.box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 0, 255), width=3)
        draw.text(
            (x1, max(0, y1 - 18)),
            f"Region {result.proposal.region}",
            fill=(220, 60, 0, 255),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.model_config)
    predictor = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    segmentation = predictor.predict(args.image)
    result = localize_region_from_instances(segmentation, args.query)
    payload = {
        "image": str(Path(args.image)),
        "segmentation_inference_time_ms": segmentation.inference_time_ms,
        **result.to_dict(include_mask=args.include_mask),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)

    if args.vis_output:
        draw_local_region_result(Path(args.image), result, Path(args.vis_output))


if __name__ == "__main__":
    main()
