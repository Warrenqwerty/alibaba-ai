from __future__ import annotations

import argparse
import json
from pathlib import Path

from fashion_mm.models.attributes import FashionAttributePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3.1.3 fine-grained attributes on an image and target mask."
    )
    parser.add_argument("image", help="Path to the RGB product image.")
    parser.add_argument("--mask", required=True, help="Path to the target-region mask.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attributes", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FashionAttributePredictor(
        args.checkpoint,
        device=args.device,
        top_k=args.top_k,
        confidence_threshold=args.confidence_threshold,
    )
    result = predictor.predict(args.image, args.mask, attributes=args.attributes)
    payload = {
        "image": str(Path(args.image)),
        "mask": str(Path(args.mask)),
        **result.to_dict(),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
