from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import LearnedRegionRanker
from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.utils.config import load_config
from scripts.eval.evaluate_local_region_manual_labels import load_manual_records
from scripts.eval.evaluate_local_region_manual_labels import summarize_records
from scripts.eval.evaluate_pretrained_grounding_manual_labels import BACKEND_NAMES
from scripts.eval.evaluate_pretrained_grounding_manual_labels import HFZeroShotGrounder
from scripts.eval.evaluate_pretrained_grounding_manual_labels import build_prompts


DEFAULT_GROUNDING_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
DEFAULT_GROUNDING_REGIONS = ("pattern", "pocket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an explicit experimental gated hybrid policy for 3.1.2: "
            "selected semantic regions use pretrained grounding, all other "
            "regions use the heuristic/local-region pipeline."
        )
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Manual JSONL with image, query_text, target_region, and target_bbox.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ranker-checkpoint",
        default=None,
        help="Optional local-region ranker checkpoint for non-grounding regions.",
    )
    parser.add_argument(
        "--grounding-regions",
        nargs="+",
        default=list(DEFAULT_GROUNDING_REGIONS),
        help="target_region values routed to pretrained grounding.",
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
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/local_region_manual_eval_gated_hybrid.json",
        help="Path to save summary and per-query records.",
    )
    return parser.parse_args()


def evaluate_gated_hybrid_records(
    manual_records: list[dict[str, Any]],
    *,
    model_config: str,
    checkpoint: str,
    device: str | None,
    ranker_checkpoint: str | None,
    grounding_regions: set[str],
    grounding_model_name: str,
    grounding_backend: str,
    prompt_mode: str,
    score_threshold: float,
) -> list[dict[str, Any]]:
    """Run the gated hybrid policy and compare selected boxes to manual labels."""
    config = load_config(model_config)
    predictor = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=checkpoint,
        device=device,
    )
    ranker = (
        LearnedRegionRanker(ranker_checkpoint, device=device)
        if ranker_checkpoint
        else None
    )
    grounder = (
        HFZeroShotGrounder(
            grounding_model_name,
            backend=grounding_backend,
            device=device,
            score_threshold=score_threshold,
        )
        if any(should_route_to_grounding(record, grounding_regions) for record in manual_records)
        else None
    )

    records: list[dict[str, Any]] = []
    image_cache: dict[str, Image.Image] = {}
    segmentation_cache: dict[str, Any] = {}
    for manual_record in manual_records:
        if should_route_to_grounding(manual_record, grounding_regions):
            if grounder is None:
                raise RuntimeError("Grounding route selected, but grounder was not initialized.")
            records.append(
                evaluate_grounding_record(
                    manual_record,
                    grounder=grounder,
                    image_cache=image_cache,
                    prompt_mode=prompt_mode,
                )
            )
            continue

        records.append(
            evaluate_heuristic_record(
                manual_record,
                predictor=predictor,
                ranker=ranker,
                segmentation_cache=segmentation_cache,
            )
        )
    return records


def should_route_to_grounding(
    manual_record: dict[str, Any],
    grounding_regions: set[str],
) -> bool:
    """Return whether a manual record should use pretrained grounding."""
    return str(manual_record.get("target_region") or "") in grounding_regions


def evaluate_grounding_record(
    manual_record: dict[str, Any],
    *,
    grounder: HFZeroShotGrounder,
    image_cache: dict[str, Image.Image],
    prompt_mode: str,
) -> dict[str, Any]:
    image_path = str(manual_record["image"])
    if image_path not in image_cache:
        image_cache[image_path] = Image.open(image_path).convert("RGB")
    prompts = build_prompts(
        manual_record["query_text"],
        manual_record.get("target_region"),
        prompt_mode=prompt_mode,
    )
    start = time.perf_counter()
    prediction = grounder.predict(image_cache[image_path], prompts)
    latency_ms = (time.perf_counter() - start) * 1000.0
    best = prediction["best"]
    predicted_box = tuple(best["bbox"]) if best is not None else None
    manual_iou = (
        box_iou(predicted_box, manual_record["target_bbox"])
        if predicted_box is not None
        else 0.0
    )
    return {
        "id": manual_record.get("id"),
        "image": image_path,
        "query_text": manual_record["query_text"],
        "target_region": manual_record.get("target_region"),
        "target_bbox": list(manual_record["target_bbox"]),
        "status": prediction["status"],
        "ranker_backend": f"gated_hybrid_grounding_{grounder.backend}",
        "selected_region": best["prompt"] if best is not None else None,
        "predicted_bbox": list(predicted_box) if predicted_box else None,
        "manual_bbox_iou": manual_iou,
        "gated_policy_route": "grounding",
        "local_region_latency_ms": latency_ms,
        "score": best["score"] if best is not None else None,
        "prompts": prompts,
        "detections": prediction["detections"][:5],
    }


def evaluate_heuristic_record(
    manual_record: dict[str, Any],
    *,
    predictor: FashionInstanceSegmentationPredictor,
    ranker: LearnedRegionRanker | None,
    segmentation_cache: dict[str, Any],
) -> dict[str, Any]:
    image_path = str(manual_record["image"])
    if image_path not in segmentation_cache:
        segmentation_cache[image_path] = predictor.predict(image_path)
    segmentation = segmentation_cache[image_path]
    result = localize_region_from_instances(
        segmentation,
        manual_record["query_text"],
        ranker=ranker,
    )
    predicted_box = (
        result.proposal.proposal.box
        if result.proposal is not None
        else None
    )
    manual_iou = (
        box_iou(predicted_box, manual_record["target_bbox"])
        if predicted_box is not None
        else 0.0
    )
    return {
        "id": manual_record.get("id"),
        "image": image_path,
        "query_text": manual_record["query_text"],
        "target_region": manual_record.get("target_region"),
        "target_bbox": list(manual_record["target_bbox"]),
        "status": result.status,
        "ranker_backend": f"gated_hybrid_{result.ranker_backend}",
        "selected_region": (
            result.proposal.proposal.region if result.proposal else None
        ),
        "predicted_bbox": list(predicted_box) if predicted_box else None,
        "manual_bbox_iou": manual_iou,
        "gated_policy_route": "heuristic",
        "segmentation_inference_time_ms": segmentation.inference_time_ms,
        "local_region_latency_ms": result.latency_ms,
    }


def main() -> None:
    args = parse_args()
    manual_records = load_manual_records(args.annotations, max_records=args.max_records)
    if not manual_records:
        raise ValueError("No labeled manual records found for gated hybrid eval.")

    grounding_regions = set(args.grounding_regions)
    records = evaluate_gated_hybrid_records(
        manual_records,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        device=args.device,
        ranker_checkpoint=args.ranker_checkpoint,
        grounding_regions=grounding_regions,
        grounding_model_name=args.grounding_model_name,
        grounding_backend=args.grounding_backend,
        prompt_mode=args.prompt_mode,
        score_threshold=args.score_threshold,
    )
    summary = {
        "annotations": str(Path(args.annotations)),
        "num_labeled_records": len(manual_records),
        "grounding_regions": sorted(grounding_regions),
        "grounding_model_name": args.grounding_model_name,
        "grounding_backend": args.grounding_backend,
        "prompt_mode": args.prompt_mode,
        "score_threshold": args.score_threshold,
        **summarize_records(records),
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
