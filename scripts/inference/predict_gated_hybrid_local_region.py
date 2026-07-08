from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import LearnedRegionRanker
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.models.local_region.query import ParsedRegionQuery
from fashion_mm.models.local_region.query import parse_region_query
from fashion_mm.models.local_region.visualization import draw_local_region_result
from fashion_mm.utils.config import load_config
from scripts.eval.evaluate_gated_hybrid_manual_labels import (
    DEFAULT_GROUNDING_MODEL_NAME,
    DEFAULT_GROUNDING_REGIONS,
)
from scripts.eval.evaluate_pretrained_grounding_manual_labels import BACKEND_NAMES
from scripts.eval.evaluate_pretrained_grounding_manual_labels import HFZeroShotGrounder
from scripts.eval.evaluate_pretrained_grounding_manual_labels import build_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the explicit experimental gated 3.1.2 policy: pattern/pocket "
            "queries use pretrained grounding, all other queries use the "
            "heuristic local-region pipeline."
        )
    )
    parser.add_argument("image", help="Path to an RGB fashion image.")
    parser.add_argument("query", help="Natural-language local-region query.")
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="3.1.1 instance-segmentation checkpoint, required for heuristic-routed queries.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ranker-checkpoint",
        default=None,
        help="Optional experimental learned local-region ranker for heuristic route.",
    )
    parser.add_argument(
        "--grounding-regions",
        nargs="+",
        default=list(DEFAULT_GROUNDING_REGIONS),
        help="Parsed region names routed to pretrained grounding.",
    )
    parser.add_argument(
        "--grounding-model-name",
        default=DEFAULT_GROUNDING_MODEL_NAME,
        help="HuggingFace model name or local model directory.",
    )
    parser.add_argument(
        "--grounding-backend",
        choices=BACKEND_NAMES,
        default="auto",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("english", "chinese", "both"),
        default="english",
    )
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--output", default=None)
    parser.add_argument("--vis-output", default=None)
    parser.add_argument("--include-mask", action="store_true")
    return parser.parse_args()


def should_use_grounding_route(
    parsed_query: ParsedRegionQuery,
    grounding_regions: set[str],
) -> bool:
    """Return whether this query should use the experimental grounding route."""
    return parsed_query.region in grounding_regions


def grounding_payload(
    *,
    image_path: Path,
    query: str,
    parsed_query: ParsedRegionQuery,
    prediction: dict[str, Any],
    prompts: list[str],
    grounder_backend: str,
    grounding_model_name: str,
    latency_ms: float,
) -> dict[str, Any]:
    """Serialize a grounding-route prediction in the local-region output shape."""
    best = prediction.get("best")
    region = None
    if best is not None:
        canonical_region = canonical_grounding_region(parsed_query, best.get("prompt"))
        region = {
            "region": canonical_region,
            "raw_grounding_prompt": best["prompt"],
            "box": [float(value) for value in best["bbox"]],
            "confidence": float(best["score"]),
            "source": f"pretrained_grounding_{grounder_backend}",
            "status": "ok",
            "reason": "selected highest-scoring pretrained grounding detection",
            "match_score": float(best["score"]),
        }

    return {
        "image": str(image_path),
        "query": {
            "text": query,
            "region": parsed_query.region,
            "garment_hint": parsed_query.garment_hint,
            "is_supported_region": parsed_query.is_supported_region,
            "spatial_hints": list(parsed_query.spatial_hints),
            "attribute_hints": list(parsed_query.attribute_hints),
            "relation_hints": list(parsed_query.relation_hints),
        },
        "selected_instance": None,
        "region": region,
        "candidate_regions": [
            {
                "region": canonical_grounding_region(
                    parsed_query,
                    detection.get("prompt"),
                ),
                "raw_grounding_prompt": detection["prompt"],
                "box": [float(value) for value in detection["bbox"]],
                "confidence": float(detection["score"]),
                "source": f"pretrained_grounding_{grounder_backend}",
                "status": "ok",
                "reason": "pretrained grounding detection",
                "match_score": float(detection["score"]),
            }
            for detection in prediction.get("detections", [])[:5]
        ],
        "ranker_backend": f"gated_hybrid_grounding_{grounder_backend}",
        "status": prediction.get("status", "no_detection"),
        "reason": (
            "query region routed to pretrained grounding"
            if best is not None
            else "pretrained grounding returned no detection"
        ),
        "latency_ms": latency_ms,
        "gated_policy_route": "grounding",
        "grounding_model_name": grounding_model_name,
        "grounding_prompts": prompts,
    }


def canonical_grounding_region(
    parsed_query: ParsedRegionQuery,
    raw_prompt: str | None,
) -> str:
    """Map noisy grounding text labels to stable 3.1.2 region names."""
    if parsed_query.region == "pocket":
        if "left" in parsed_query.spatial_hints:
            return "left_pocket"
        if "right" in parsed_query.spatial_hints:
            return "right_pocket"
        return "pocket"
    if parsed_query.region == "cuff":
        if "left" in parsed_query.spatial_hints:
            return "left_cuff"
        if "right" in parsed_query.spatial_hints:
            return "right_cuff"
        return "cuff"
    if parsed_query.region:
        return parsed_query.region
    normalized = (raw_prompt or "").strip()
    return normalized or "grounding_region"


def run_grounding_route(args: argparse.Namespace, parsed_query: ParsedRegionQuery) -> dict[str, Any]:
    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    prompts = build_prompts(
        args.query,
        parsed_query.region,
        prompt_mode=args.prompt_mode,
    )
    grounder = HFZeroShotGrounder(
        args.grounding_model_name,
        backend=args.grounding_backend,
        device=args.device,
        score_threshold=args.score_threshold,
    )
    start = time.perf_counter()
    prediction = grounder.predict(image, prompts)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return grounding_payload(
        image_path=image_path,
        query=args.query,
        parsed_query=parsed_query,
        prediction=prediction,
        prompts=prompts,
        grounder_backend=grounder.backend,
        grounding_model_name=args.grounding_model_name,
        latency_ms=latency_ms,
    )


def run_heuristic_route(args: argparse.Namespace) -> dict[str, Any]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required when the query uses the heuristic route.")
    config = load_config(args.model_config)
    predictor = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    ranker = (
        LearnedRegionRanker(args.ranker_checkpoint, device=args.device)
        if args.ranker_checkpoint
        else None
    )
    segmentation = predictor.predict(args.image)
    result = localize_region_from_instances(segmentation, args.query, ranker=ranker)
    payload = {
        "image": str(Path(args.image)),
        "segmentation_inference_time_ms": segmentation.inference_time_ms,
        **result.to_dict(include_mask=args.include_mask),
        "gated_policy_route": "heuristic",
    }
    if args.vis_output:
        draw_local_region_result(Path(args.image), result, Path(args.vis_output))
    return payload


def draw_grounding_result(image_path: Path, payload: dict[str, Any], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    region = payload.get("region")
    if region is not None and region.get("box") is not None:
        x1, y1, x2, y2 = region["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 0), width=3)
        label = f'{region["region"]} {region["confidence"]:.2f}'
        draw.text((x1, max(0, y1 - 18)), label, fill=(220, 60, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    parsed_query = parse_region_query(args.query)
    grounding_regions = set(args.grounding_regions)
    if should_use_grounding_route(parsed_query, grounding_regions):
        payload = run_grounding_route(args, parsed_query)
        if args.vis_output:
            draw_grounding_result(Path(args.image), payload, Path(args.vis_output))
    else:
        payload = run_heuristic_route(args)

    payload["gated_policy"] = {
        "grounding_regions": sorted(grounding_regions),
        "selected_route": payload["gated_policy_route"],
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
