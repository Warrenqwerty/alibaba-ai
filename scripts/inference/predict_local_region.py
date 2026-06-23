from __future__ import annotations

import argparse
import json
from pathlib import Path

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.models.local_region.visualization import draw_local_region_result
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
