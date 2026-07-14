from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from typing import Mapping

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import LearnedRegionRanker
from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import filter_grounding_detections_to_garment
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.models.local_region import parse_region_query
from fashion_mm.models.local_region import select_garment_instance
from fashion_mm.utils.config import load_config
from scripts.eval.evaluate_local_region_manual_labels import load_manual_records
from scripts.eval.evaluate_local_region_manual_labels import summarize_records
from scripts.eval.evaluate_pretrained_grounding_manual_labels import BACKEND_NAMES
from scripts.eval.evaluate_pretrained_grounding_manual_labels import HFZeroShotGrounder
from scripts.eval.evaluate_pretrained_grounding_manual_labels import PROMPT_PROFILES
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
        "--grounding-routes",
        nargs="*",
        default=None,
        metavar="REGION=MODEL_NAME",
        help=(
            "Optional fixed per-region model routes, e.g. "
            "pattern=IDEA-Research/grounding-dino-tiny "
            "pocket=IDEA-Research/grounding-dino-base. When supplied, this "
            "replaces --grounding-regions/--grounding-model-name."
        ),
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
    parser.add_argument("--prompt-profile", choices=PROMPT_PROFILES, default="ensemble")
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument(
        "--constrain-grounding-to-garment",
        action="store_true",
        help=(
            "Experimental: keep only GroundingDINO detections overlapping the "
            "3.1.1 selected garment mask; use heuristic fallback if none remain."
        ),
    )
    parser.add_argument(
        "--grounding-min-mask-coverage",
        type=float,
        default=0.2,
        help="Minimum detection-box area covered by the selected garment mask.",
    )
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
    grounding_routes: Mapping[str, str] | None = None,
    grounding_backend: str,
    prompt_mode: str,
    prompt_profile: str,
    score_threshold: float,
    constrain_grounding_to_garment: bool = False,
    grounding_min_mask_coverage: float = 0.2,
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
    resolved_grounding_routes = resolve_grounding_routes(
        grounding_regions=grounding_regions,
        grounding_model_name=grounding_model_name,
        grounding_routes=grounding_routes,
    )
    grounders = {
        model_name: HFZeroShotGrounder(
            model_name,
            backend=grounding_backend,
            device=device,
            score_threshold=score_threshold,
        )
        for model_name in sorted(set(resolved_grounding_routes.values()))
    }

    records: list[dict[str, Any]] = []
    image_cache: dict[str, Image.Image] = {}
    segmentation_cache: dict[str, Any] = {}
    for manual_record in manual_records:
        target_region = str(manual_record.get("target_region") or "")
        grounding_model = resolved_grounding_routes.get(target_region)
        if grounding_model is not None:
            grounder = grounders[grounding_model]
            segmentation = None
            if constrain_grounding_to_garment:
                image_path = str(manual_record["image"])
                if image_path not in segmentation_cache:
                    segmentation_cache[image_path] = predictor.predict(image_path)
                segmentation = segmentation_cache[image_path]
            grounding_record = evaluate_grounding_record(
                manual_record,
                grounder=grounder,
                image_cache=image_cache,
                prompt_mode=prompt_mode,
                prompt_profile=prompt_profile,
                segmentation=segmentation,
                grounding_min_mask_coverage=grounding_min_mask_coverage,
            )
            if (
                constrain_grounding_to_garment
                and grounding_record["status"] == "no_detection_in_selected_garment"
            ):
                fallback = evaluate_heuristic_record(
                    manual_record,
                    predictor=predictor,
                    ranker=ranker,
                    segmentation_cache=segmentation_cache,
                )
                fallback["gated_policy_route"] = "heuristic_fallback"
                fallback["ranker_backend"] = f"gated_hybrid_fallback_{fallback['ranker_backend']}"
                fallback["grounding_filter_status"] = grounding_record["status"]
                fallback["grounding_detections"] = grounding_record["detections"]
                fallback["grounding_selected_instance"] = grounding_record["selected_instance"]
                records.append(fallback)
            else:
                records.append(grounding_record)
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


def parse_grounding_routes(values: list[str] | None) -> dict[str, str] | None:
    """Parse explicit REGION=MODEL_NAME routes from command-line arguments."""
    if values is None:
        return None
    routes: dict[str, str] = {}
    for value in values:
        region, separator, model_name = value.partition("=")
        region = region.strip()
        model_name = model_name.strip()
        if not separator or not region or not model_name:
            raise ValueError(
                "Each --grounding-routes value must use REGION=MODEL_NAME; "
                f"got {value!r}."
            )
        if region in routes and routes[region] != model_name:
            raise ValueError(f"Conflicting models configured for region {region!r}.")
        routes[region] = model_name
    if not routes:
        raise ValueError("--grounding-routes requires at least one REGION=MODEL_NAME value.")
    return routes


def resolve_grounding_routes(
    *,
    grounding_regions: set[str],
    grounding_model_name: str,
    grounding_routes: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return fixed target-region to pretrained-model routing for one run."""
    if grounding_routes is not None:
        return dict(grounding_routes)
    return {region: grounding_model_name for region in grounding_regions}


def evaluate_grounding_record(
    manual_record: dict[str, Any],
    *,
    grounder: HFZeroShotGrounder,
    image_cache: dict[str, Image.Image],
    prompt_mode: str,
    prompt_profile: str,
    segmentation: Any | None = None,
    grounding_min_mask_coverage: float = 0.2,
) -> dict[str, Any]:
    image_path = str(manual_record["image"])
    if image_path not in image_cache:
        image_cache[image_path] = Image.open(image_path).convert("RGB")
    prompts = build_prompts(
        manual_record["query_text"],
        manual_record.get("target_region"),
        prompt_mode=prompt_mode,
        prompt_profile=prompt_profile,
    )
    start = time.perf_counter()
    prediction = grounder.predict(image_cache[image_path], prompts)
    latency_ms = (time.perf_counter() - start) * 1000.0
    selected_instance = None
    filter_status = "not_requested"
    if segmentation is not None:
        selected_instance = select_garment_instance(
            segmentation,
            parse_region_query(manual_record["query_text"]),
        )
        if selected_instance is None:
            filter_status = "no_selected_garment"
        else:
            filtered = filter_grounding_detections_to_garment(
                prediction["detections"],
                selected_instance.mask,
                min_mask_coverage=grounding_min_mask_coverage,
            )
            prediction = {
                **prediction,
                "status": "ok" if filtered else "no_detection_in_selected_garment",
                "detections": filtered,
                "best": filtered[0] if filtered else None,
            }
            filter_status = "accepted" if filtered else "no_detection_in_selected_garment"
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
        "grounding_model_name": grounder.model_name,
        "selected_region": best["prompt"] if best is not None else None,
        "predicted_bbox": list(predicted_box) if predicted_box else None,
        "manual_bbox_iou": manual_iou,
        "gated_policy_route": "grounding",
        "local_region_latency_ms": latency_ms,
        "score": best["score"] if best is not None else None,
        "prompts": prompts,
        "detections": prediction["detections"][:5],
        "grounding_filter_status": filter_status,
        "selected_instance": (
            selected_instance.to_dict(include_mask=False)
            if selected_instance is not None
            else None
        ),
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
    grounding_routes = parse_grounding_routes(args.grounding_routes)
    resolved_grounding_routes = resolve_grounding_routes(
        grounding_regions=grounding_regions,
        grounding_model_name=args.grounding_model_name,
        grounding_routes=grounding_routes,
    )
    records = evaluate_gated_hybrid_records(
        manual_records,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        device=args.device,
        ranker_checkpoint=args.ranker_checkpoint,
        grounding_regions=grounding_regions,
        grounding_model_name=args.grounding_model_name,
        grounding_routes=grounding_routes,
        grounding_backend=args.grounding_backend,
        prompt_mode=args.prompt_mode,
        prompt_profile=args.prompt_profile,
        score_threshold=args.score_threshold,
        constrain_grounding_to_garment=args.constrain_grounding_to_garment,
        grounding_min_mask_coverage=args.grounding_min_mask_coverage,
    )
    summary = {
        "annotations": str(Path(args.annotations)),
        "num_labeled_records": len(manual_records),
        "grounding_regions": sorted(resolved_grounding_routes),
        "grounding_model_name": (
            args.grounding_model_name if grounding_routes is None else None
        ),
        "grounding_routes": resolved_grounding_routes,
        "grounding_backend": args.grounding_backend,
        "prompt_mode": args.prompt_mode,
        "prompt_profile": args.prompt_profile,
        "score_threshold": args.score_threshold,
        "constrain_grounding_to_garment": args.constrain_grounding_to_garment,
        "grounding_min_mask_coverage": args.grounding_min_mask_coverage,
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
