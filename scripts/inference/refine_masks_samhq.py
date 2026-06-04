from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine detected fashion masks with SAM-HQ box prompts."
    )
    parser.add_argument("image", help="Input RGB image.")
    parser.add_argument("--config", default="configs/model/instance_segmentation.yaml")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--sam-hq-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--output-json", default="outputs/samhq_refined_prediction.json")
    parser.add_argument("--output-vis", default="outputs/samhq_refined_prediction.png")
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--hq-token-only", action="store_true", default=True)
    return parser.parse_args()


def load_samhq_predictor(
    checkpoint_path: str | Path,
    model_type: str,
    device: torch.device,
) -> Any:
    """Load SAM-HQ predictor from the official SysCV/sam-hq package."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise ImportError(
            "SAM-HQ is not installed. Install it with:\n"
            "  cd /root/projects\n"
            "  git clone https://github.com/SysCV/sam-hq.git\n"
            "  cd sam-hq\n"
            "  pip install -e ."
        ) from error

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def refine_instances_with_samhq(
    image: Image.Image,
    detector_result: Any,
    sam_predictor: Any,
    hq_token_only: bool,
) -> list[dict[str, Any]]:
    """Use each detected box as a SAM-HQ prompt and return refined masks."""
    image_rgb = np.asarray(image.convert("RGB"))
    sam_predictor.set_image(image_rgb)
    refined_instances: list[dict[str, Any]] = []

    for instance in detector_result.instances:
        box = np.asarray(instance.box, dtype=np.float32)
        masks, sam_scores, _ = sam_predictor.predict(
            box=box,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
        refined_mask = np.asarray(masks[0], dtype=bool)
        sam_score = float(sam_scores[0]) if len(sam_scores) else 0.0
        refined_instances.append(
            {
                "box": [float(value) for value in instance.box],
                "label_id": int(instance.label_id),
                "label_name": instance.label_name,
                "detector_score": float(instance.score),
                "sam_hq_score": sam_score,
                "area": int(refined_mask.sum()),
                "mask": refined_mask,
            }
        )

    return refined_instances


def save_json(
    instances: list[dict[str, Any]],
    image_size: tuple[int, int],
    output_path: str | Path,
    include_masks: bool,
) -> None:
    serializable_instances = []
    for instance in instances:
        item = dict(instance)
        mask = item.pop("mask")
        if include_masks:
            item["mask"] = mask.astype(np.uint8).tolist()
        serializable_instances.append(item)

    payload = {
        "image_size": list(image_size),
        "instances": serializable_instances,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_visualization(
    image: Image.Image,
    instances: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    colors = [
        (0, 255, 80, 105),
        (255, 90, 0, 105),
        (80, 160, 255, 105),
        (255, 220, 0, 105),
        (220, 80, 255, 105),
    ]

    for index, instance in enumerate(instances):
        color = colors[index % len(colors)]
        mask_image = Image.fromarray(instance["mask"].astype(np.uint8) * 255, mode="L")
        color_layer = Image.new("RGBA", rgba.size, color)
        overlay = Image.composite(color_layer, overlay, mask_image)
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = instance["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=3)
        text = (
            f"{instance['label_name']} det={instance['detector_score']:.2f} "
            f"sam={instance['sam_hq_score']:.2f}"
        )
        draw.text((x1, max(0, y1 - 16)), text, fill=color[:3] + (255,))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(rgba, overlay).convert("RGB").save(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.score_threshold is not None:
        config["inference"]["score_threshold"] = args.score_threshold

    device = torch.device(args.device)
    image = Image.open(args.image).convert("RGB")

    detector = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=args.detector_checkpoint,
        device=device,
    )
    detector_result = detector.predict(image)
    sam_predictor = load_samhq_predictor(
        args.sam_hq_checkpoint,
        args.sam_model_type,
        device,
    )
    refined_instances = refine_instances_with_samhq(
        image,
        detector_result,
        sam_predictor,
        hq_token_only=args.hq_token_only,
    )

    save_json(
        refined_instances,
        image.size,
        args.output_json,
        include_masks=args.include_masks,
    )
    save_visualization(image, refined_instances, args.output_vis)
    print(f"saved json: {args.output_json}")
    print(f"saved visualization: {args.output_vis}")


if __name__ == "__main__":
    main()
