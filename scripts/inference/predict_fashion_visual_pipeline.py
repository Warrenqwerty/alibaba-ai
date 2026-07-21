from __future__ import annotations

import argparse
import json
from pathlib import Path

from fashion_mm.models.attributes import FashionAttributePredictor
from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region.visualization import draw_local_region_result
from fashion_mm.pipelines import FashionVisualPipeline
from fashion_mm.utils.config import load_config
from fashion_mm.utils.image_io import save_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 3.1.1 -> 3.1.2 -> 3.1.3 fashion visual pipeline."
    )
    parser.add_argument("image", help="Path to an RGB fashion image.")
    parser.add_argument("query", help="Natural-language target-region query.")
    parser.add_argument(
        "--segmentation-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--segmentation-checkpoint", required=True)
    parser.add_argument("--attribute-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attributes", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--mask-output", default=None)
    parser.add_argument("--vis-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segmentation_config = load_config(args.segmentation_config)
    segmentation_predictor = FashionInstanceSegmentationPredictor(
        segmentation_config,
        checkpoint_path=args.segmentation_checkpoint,
        device=args.device,
    )
    attribute_predictor = FashionAttributePredictor(
        args.attribute_checkpoint,
        device=args.device,
    )
    pipeline = FashionVisualPipeline(segmentation_predictor, attribute_predictor)
    result = pipeline.predict(args.image, args.query, attributes=args.attributes)
    if args.mask_output and result.local_region.proposal is not None:
        save_mask(result.local_region.proposal.proposal.mask, args.mask_output)

    payload = {
        "image": str(Path(args.image)),
        "query": args.query,
        **result.to_dict(),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)

    if args.vis_output:
        draw_local_region_result(
            Path(args.image), result.local_region, Path(args.vis_output)
        )


if __name__ == "__main__":
    main()
