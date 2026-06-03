from __future__ import annotations

import argparse
import json
from pathlib import Path

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3.1.1 fashion segmentation.")
    parser.add_argument("image", help="Path to an RGB product image.")
    parser.add_argument("--config", default="configs/model/instance_segmentation.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    predictor = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    result = predictor.predict(args.image).to_dict(include_masks=args.include_masks)
    payload = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
