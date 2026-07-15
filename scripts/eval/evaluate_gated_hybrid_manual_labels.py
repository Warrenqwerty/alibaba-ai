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
from fashion_mm.models.local_region import query_wearer_side
from fashion_mm.models.local_region import select_garment_instance
from fashion_mm.models.local_region import select_wearer_side_detection
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
        "--grounding-route-profiles",
        nargs="*",
        default=None,
        metavar="REGION=PROFILE",
        help=(
            "Optional per-region prompt-profile overrides for explicit routes, "
            "e.g. cuff=precise waist=ensemble. Unspecified regions use "
            "--prompt-profile."
        ),
    )
    parser.add_argument(
        "--grounding-route-thresholds",
        nargs="*",
        default=None,
        metavar="REGION=SCORE",
        help=(
            "Optional per-region detection-score thresholds for explicit routes, "
            "e.g. cuff=0.05 waist=0.05. Unspecified regions use "
            "--score-threshold."
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
    parser.add_argument(
        "--fallback-on-no-detection",
        action="store_true",
        help=(
            "Use the heuristic route when a selected grounding model returns "
            "no detection. Disabled by default to preserve historical results."
        ),
    )
    parser.add_argument(
        "--wearer-side-regions",
        nargs="*",
        default=[],
        help=(
            "Grounding routes that select a credible Top-K box on the query's "
            "garment/wearer side. Enable only for independently validated regions."
        ),
    )
    parser.add_argument(
        "--wearer-side-min-score-ratio",
        type=float,
        default=0.5,
        help="Minimum side-candidate score divided by the top detection score.",
    )
    parser.add_argument(
        "--record-heuristic-candidates-for-grounding",
        action="store_true",
        help=(
            "Diagnostic only: also run and save the heuristic candidate for "
            "grounding-routed records without changing the selected result."
        ),
    )
    parser.add_argument(
        "--diagnostic-grounding-routes",
        nargs="*",
        default=None,
        metavar="REGION=MODEL_NAME",
        help=(
            "Diagnostic-only grounding routes for heuristic-selected regions. "
            "Their Top-K detections are saved but never selected online."
        ),
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
    grounding_route_profiles: Mapping[str, str] | None = None,
    grounding_route_thresholds: Mapping[str, float] | None = None,
    grounding_backend: str,
    prompt_mode: str,
    prompt_profile: str,
    score_threshold: float,
    constrain_grounding_to_garment: bool = False,
    grounding_min_mask_coverage: float = 0.2,
    fallback_on_no_detection: bool = False,
    wearer_side_regions: set[str] | None = None,
    wearer_side_min_score_ratio: float = 0.5,
    record_heuristic_candidates_for_grounding: bool = False,
    diagnostic_grounding_routes: Mapping[str, str] | None = None,
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
    diagnostic_grounding_routes = dict(diagnostic_grounding_routes or {})
    duplicate_routes = {
        region
        for region, model_name in diagnostic_grounding_routes.items()
        if resolved_grounding_routes.get(region) == model_name
    }
    if duplicate_routes:
        raise ValueError(
            "A diagnostic route must use a different model from its selected "
            f"route; duplicates: {', '.join(sorted(duplicate_routes))}."
        )
    route_configs = {
        (
            model_name,
            resolve_score_threshold(
                region,
                default_threshold=score_threshold,
                route_thresholds=grounding_route_thresholds,
            ),
        )
        for region, model_name in resolved_grounding_routes.items()
    }
    route_configs.update(
        (model_name, score_threshold)
        for model_name in diagnostic_grounding_routes.values()
    )
    grounders = {
        (model_name, threshold): HFZeroShotGrounder(
            model_name,
            backend=grounding_backend,
            device=device,
            score_threshold=threshold,
        )
        for model_name, threshold in sorted(route_configs)
    }

    records: list[dict[str, Any]] = []
    image_cache: dict[str, Image.Image] = {}
    segmentation_cache: dict[str, Any] = {}
    for manual_record in manual_records:
        target_region = str(manual_record.get("target_region") or "")
        grounding_model = resolved_grounding_routes.get(target_region)
        if grounding_model is not None:
            route_score_threshold = resolve_score_threshold(
                target_region,
                default_threshold=score_threshold,
                route_thresholds=grounding_route_thresholds,
            )
            grounder = grounders[(grounding_model, route_score_threshold)]
            route_prompt_profile = resolve_prompt_profile(
                target_region,
                default_profile=prompt_profile,
                route_profiles=grounding_route_profiles,
            )
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
                prompt_profile=route_prompt_profile,
                segmentation=segmentation,
                grounding_min_mask_coverage=grounding_min_mask_coverage,
                apply_wearer_side_selection=(
                    target_region in (wearer_side_regions or set())
                ),
                wearer_side_min_score_ratio=wearer_side_min_score_ratio,
            )
            if record_heuristic_candidates_for_grounding:
                heuristic_candidate = evaluate_heuristic_record(
                    manual_record,
                    predictor=predictor,
                    ranker=ranker,
                    segmentation_cache=segmentation_cache,
                )
                grounding_record["heuristic_candidate"] = {
                    key: heuristic_candidate.get(key)
                    for key in (
                        "status",
                        "ranker_backend",
                        "selected_region",
                        "predicted_bbox",
                        "manual_bbox_iou",
                    )
                }
            diagnostic_model = diagnostic_grounding_routes.get(target_region)
            if diagnostic_model is not None:
                diagnostic_record = evaluate_grounding_record(
                    manual_record,
                    grounder=grounders[(diagnostic_model, score_threshold)],
                    image_cache=image_cache,
                    prompt_mode=prompt_mode,
                    prompt_profile=prompt_profile,
                )
                grounding_record["diagnostic_grounding_candidate"] = (
                    diagnostic_grounding_payload(diagnostic_record)
                )
            fallback_reason = grounding_fallback_reason(
                grounding_record,
                constrain_grounding_to_garment=constrain_grounding_to_garment,
                fallback_on_no_detection=fallback_on_no_detection,
            )
            if fallback_reason is not None:
                fallback = evaluate_heuristic_record(
                    manual_record,
                    predictor=predictor,
                    ranker=ranker,
                    segmentation_cache=segmentation_cache,
                )
                fallback["gated_policy_route"] = f"heuristic_fallback_{fallback_reason}"
                fallback["ranker_backend"] = f"gated_hybrid_fallback_{fallback['ranker_backend']}"
                fallback["grounding_filter_status"] = grounding_record["status"]
                fallback["grounding_detections"] = grounding_record["detections"]
                fallback["grounding_selected_instance"] = grounding_record.get("selected_instance")
                if "diagnostic_grounding_candidate" in grounding_record:
                    fallback["diagnostic_grounding_candidate"] = grounding_record[
                        "diagnostic_grounding_candidate"
                    ]
                records.append(fallback)
            else:
                records.append(grounding_record)
            continue

        heuristic_record = evaluate_heuristic_record(
            manual_record,
            predictor=predictor,
            ranker=ranker,
            segmentation_cache=segmentation_cache,
        )
        diagnostic_model = diagnostic_grounding_routes.get(target_region)
        if diagnostic_model is not None:
            diagnostic_record = evaluate_grounding_record(
                manual_record,
                grounder=grounders[(diagnostic_model, score_threshold)],
                image_cache=image_cache,
                prompt_mode=prompt_mode,
                prompt_profile=prompt_profile,
            )
            heuristic_record["diagnostic_grounding_candidate"] = (
                diagnostic_grounding_payload(diagnostic_record)
            )
        records.append(heuristic_record)
    return records


def diagnostic_grounding_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep candidate provenance without changing the selected online record."""
    return {
        key: record.get(key)
        for key in (
            "status",
            "grounding_model_name",
            "selected_region",
            "predicted_bbox",
            "manual_bbox_iou",
            "score",
            "prompt_profile",
            "prompts",
            "detections",
        )
    }


def should_route_to_grounding(
    manual_record: dict[str, Any],
    grounding_regions: set[str],
) -> bool:
    """Return whether a manual record should use pretrained grounding."""
    return str(manual_record.get("target_region") or "") in grounding_regions


def grounding_fallback_reason(
    grounding_record: Mapping[str, Any],
    *,
    constrain_grounding_to_garment: bool,
    fallback_on_no_detection: bool,
) -> str | None:
    """Return the explicit reason to replace a failed grounding result."""
    status = grounding_record.get("status")
    if (
        constrain_grounding_to_garment
        and status == "no_detection_in_selected_garment"
    ):
        return "garment_filter"
    if fallback_on_no_detection and status == "no_detection":
        return "no_detection"
    return None


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


def parse_grounding_route_profiles(values: list[str] | None) -> dict[str, str] | None:
    """Parse explicit REGION=PROMPT_PROFILE overrides for grounded routes."""
    if values is None:
        return None
    profiles: dict[str, str] = {}
    for value in values:
        region, separator, profile = value.partition("=")
        region = region.strip()
        profile = profile.strip()
        if not separator or not region or not profile:
            raise ValueError(
                "Each --grounding-route-profiles value must use REGION=PROFILE; "
                f"got {value!r}."
            )
        if profile not in PROMPT_PROFILES:
            raise ValueError(
                f"Unsupported prompt profile {profile!r}; "
                f"choose one of {', '.join(PROMPT_PROFILES)}."
            )
        if region in profiles and profiles[region] != profile:
            raise ValueError(f"Conflicting prompt profiles configured for region {region!r}.")
        profiles[region] = profile
    if not profiles:
        raise ValueError("--grounding-route-profiles requires at least one REGION=PROFILE value.")
    return profiles


def resolve_prompt_profile(
    region: str,
    *,
    default_profile: str,
    route_profiles: Mapping[str, str] | None,
) -> str:
    """Return a validated per-region override or the command-level default."""
    if route_profiles is None:
        return default_profile
    return route_profiles.get(region, default_profile)


def parse_grounding_route_thresholds(
    values: list[str] | None,
) -> dict[str, float] | None:
    """Parse explicit REGION=SCORE threshold overrides for grounded routes."""
    if values is None:
        return None
    thresholds: dict[str, float] = {}
    for value in values:
        region, separator, raw_threshold = value.partition("=")
        region = region.strip()
        raw_threshold = raw_threshold.strip()
        if not separator or not region or not raw_threshold:
            raise ValueError(
                "Each --grounding-route-thresholds value must use REGION=SCORE; "
                f"got {value!r}."
            )
        try:
            threshold = float(raw_threshold)
        except ValueError as error:
            raise ValueError(
                f"Invalid score threshold {raw_threshold!r} for region {region!r}."
            ) from error
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Score threshold for region {region!r} must be between 0 and 1."
            )
        if region in thresholds and thresholds[region] != threshold:
            raise ValueError(f"Conflicting score thresholds configured for region {region!r}.")
        thresholds[region] = threshold
    if not thresholds:
        raise ValueError("--grounding-route-thresholds requires at least one REGION=SCORE value.")
    return thresholds


def resolve_score_threshold(
    region: str,
    *,
    default_threshold: float,
    route_thresholds: Mapping[str, float] | None,
) -> float:
    """Return a per-region score threshold override or the command default."""
    if route_thresholds is None:
        return default_threshold
    return route_thresholds.get(region, default_threshold)


def evaluate_grounding_record(
    manual_record: dict[str, Any],
    *,
    grounder: HFZeroShotGrounder,
    image_cache: dict[str, Image.Image],
    prompt_mode: str,
    prompt_profile: str,
    segmentation: Any | None = None,
    grounding_min_mask_coverage: float = 0.2,
    apply_wearer_side_selection: bool = False,
    wearer_side_min_score_ratio: float = 0.5,
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
    wearer_side_selection_status = "not_requested"
    if apply_wearer_side_selection:
        selected, wearer_side_selection_status = select_wearer_side_detection(
            prediction["detections"],
            query_text=manual_record["query_text"],
            image_width=image_cache[image_path].width,
            min_score_ratio=wearer_side_min_score_ratio,
        )
        prediction = {**prediction, "best": selected}
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
        "grounding_score_threshold": grounder.score_threshold,
        "prompt_profile": prompt_profile,
        "prompts": prompts,
        "detections": prediction["detections"][:5],
        "grounding_filter_status": filter_status,
        "wearer_side_selection_status": wearer_side_selection_status,
        "wearer_side": query_wearer_side(manual_record["query_text"]),
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


def resolve_cli_grounding_policy(
    args: argparse.Namespace,
) -> tuple[
    set[str],
    dict[str, str] | None,
    dict[str, str] | None,
    dict[str, float] | None,
    dict[str, str],
]:
    """Parse and resolve all CLI route options before model initialization."""
    grounding_regions = set(args.grounding_regions)
    grounding_routes = parse_grounding_routes(args.grounding_routes)
    grounding_route_profiles = parse_grounding_route_profiles(
        args.grounding_route_profiles
    )
    grounding_route_thresholds = parse_grounding_route_thresholds(
        args.grounding_route_thresholds
    )
    resolved_grounding_routes = resolve_grounding_routes(
        grounding_regions=grounding_regions,
        grounding_model_name=args.grounding_model_name,
        grounding_routes=grounding_routes,
    )
    return (
        grounding_regions,
        grounding_routes,
        grounding_route_profiles,
        grounding_route_thresholds,
        resolved_grounding_routes,
    )


def main() -> None:
    args = parse_args()
    manual_records = load_manual_records(args.annotations, max_records=args.max_records)
    if not manual_records:
        raise ValueError("No labeled manual records found for gated hybrid eval.")

    (
        grounding_regions,
        grounding_routes,
        grounding_route_profiles,
        grounding_route_thresholds,
        resolved_grounding_routes,
    ) = resolve_cli_grounding_policy(args)
    diagnostic_grounding_routes = parse_grounding_routes(
        args.diagnostic_grounding_routes
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
        grounding_route_profiles=grounding_route_profiles,
        grounding_route_thresholds=grounding_route_thresholds,
        grounding_backend=args.grounding_backend,
        prompt_mode=args.prompt_mode,
        prompt_profile=args.prompt_profile,
        score_threshold=args.score_threshold,
        constrain_grounding_to_garment=args.constrain_grounding_to_garment,
        grounding_min_mask_coverage=args.grounding_min_mask_coverage,
        fallback_on_no_detection=args.fallback_on_no_detection,
        wearer_side_regions=set(args.wearer_side_regions),
        wearer_side_min_score_ratio=args.wearer_side_min_score_ratio,
        record_heuristic_candidates_for_grounding=(
            args.record_heuristic_candidates_for_grounding
        ),
        diagnostic_grounding_routes=diagnostic_grounding_routes,
    )
    summary = {
        "annotations": str(Path(args.annotations)),
        "num_labeled_records": len(manual_records),
        "grounding_regions": sorted(resolved_grounding_routes),
        "grounding_model_name": (
            args.grounding_model_name if grounding_routes is None else None
        ),
        "grounding_routes": resolved_grounding_routes,
        "grounding_route_profiles": grounding_route_profiles or {},
        "grounding_route_thresholds": grounding_route_thresholds or {},
        "grounding_backend": args.grounding_backend,
        "prompt_mode": args.prompt_mode,
        "prompt_profile": args.prompt_profile,
        "score_threshold": args.score_threshold,
        "constrain_grounding_to_garment": args.constrain_grounding_to_garment,
        "grounding_min_mask_coverage": args.grounding_min_mask_coverage,
        "fallback_on_no_detection": args.fallback_on_no_detection,
        "wearer_side_regions": sorted(set(args.wearer_side_regions)),
        "wearer_side_min_score_ratio": args.wearer_side_min_score_ratio,
        "record_heuristic_candidates_for_grounding": (
            args.record_heuristic_candidates_for_grounding
        ),
        "diagnostic_grounding_routes": diagnostic_grounding_routes or {},
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
