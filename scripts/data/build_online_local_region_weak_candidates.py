from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_gated_hybrid_manual_labels import (
    DEFAULT_GROUNDING_MODEL_NAME,
)
from scripts.eval.evaluate_gated_hybrid_manual_labels import (
    evaluate_gated_hybrid_records,
)
from scripts.eval.evaluate_gated_hybrid_manual_labels import (
    parse_grounding_route_profiles,
)
from scripts.eval.evaluate_gated_hybrid_manual_labels import (
    parse_grounding_route_thresholds,
)
from scripts.eval.evaluate_gated_hybrid_manual_labels import parse_grounding_routes
from scripts.eval.evaluate_gated_hybrid_manual_labels import resolve_grounding_routes
from scripts.eval.evaluate_local_region_manual_labels import summarize_records
from scripts.eval.evaluate_pretrained_grounding_manual_labels import BACKEND_NAMES
from scripts.eval.evaluate_pretrained_grounding_manual_labels import PROMPT_PROFILES


CANONICAL_REGIONS = {
    "left_cuff": "cuff",
    "right_cuff": "cuff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real online 3.1.2 candidate generators over independent "
            "DeepFashion2 landmark weak labels. Landmark boxes score candidates "
            "after inference and are never used to generate candidates."
        )
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--regions", nargs="+", default=["cuff", "waist"])
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-multi-item-images",
        action="store_true",
        help="Include images containing more than one annotated garment.",
    )
    parser.add_argument(
        "--grounding-routes",
        nargs="+",
        required=True,
        metavar="REGION=MODEL_NAME",
    )
    parser.add_argument(
        "--grounding-route-profiles",
        nargs="*",
        default=None,
        metavar="REGION=PROFILE",
    )
    parser.add_argument(
        "--grounding-route-thresholds",
        nargs="*",
        default=None,
        metavar="REGION=SCORE",
    )
    parser.add_argument(
        "--diagnostic-grounding-routes",
        nargs="*",
        default=None,
        metavar="REGION=MODEL_NAME",
    )
    parser.add_argument("--grounding-backend", choices=BACKEND_NAMES, default="auto")
    parser.add_argument(
        "--prompt-mode",
        choices=("english", "chinese", "both"),
        default="english",
    )
    parser.add_argument("--prompt-profile", choices=PROMPT_PROFILES, default="ensemble")
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--wearer-side-regions", nargs="*", default=[])
    parser.add_argument("--wearer-side-min-score-ratio", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def canonical_region(region: str) -> str:
    return CANONICAL_REGIONS.get(region, region)


def weak_group_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("image") or ""),
        str(record.get("item_key") or ""),
        str(record.get("region") or ""),
    )


def iter_weak_groups(path: str | Path) -> Iterator[list[dict[str, Any]]]:
    current_key: tuple[str, str, str] | None = None
    current_group: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(
                    f"line {line_number}: weak query record must be an object"
                )
            key = weak_group_key(record)
            if current_key is not None and key != current_key:
                yield current_group
                current_group = []
            current_key = key
            current_group.append(record)
    if current_group:
        yield current_group


def is_eligible_group(
    group: list[dict[str, Any]],
    *,
    regions: set[str],
    allow_multi_item_images: bool,
) -> bool:
    first = group[0]
    if first.get("source") != "landmark_pseudo_label":
        return False
    if canonical_region(str(first.get("region") or "")) not in regions:
        return False
    if not allow_multi_item_images and int(first.get("num_items_in_image") or 1) != 1:
        return False
    return first.get("region_box") is not None and bool(first.get("image"))


def validate_weak_group(group: list[dict[str, Any]]) -> None:
    """Reject mixed or malformed groups before any model inference runs."""
    first = group[0]
    fields = ("image", "item_key", "region", "region_box", "source")
    for record in group[1:]:
        for field in fields:
            if record.get(field) != first.get(field):
                raise ValueError(
                    f"Weak query group mixes {field!r} values for "
                    f"{weak_group_key(first)}"
                )


def sample_weak_query_records(
    path: str | Path,
    *,
    regions: set[str],
    max_records: int | None,
    seed: int,
    allow_multi_item_images: bool,
) -> tuple[list[dict[str, Any]], int]:
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when supplied")
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    num_eligible_groups = 0
    for group in iter_weak_groups(path):
        validate_weak_group(group)
        if not is_eligible_group(
            group,
            regions=regions,
            allow_multi_item_images=allow_multi_item_images,
        ):
            continue
        chosen = dict(rng.choice(group))
        num_eligible_groups += 1
        if max_records is None or len(reservoir) < max_records:
            reservoir.append(chosen)
            continue
        replacement = rng.randrange(num_eligible_groups)
        if replacement < max_records:
            reservoir[replacement] = chosen
    reservoir.sort(
        key=lambda record: (
            str(record.get("image") or ""),
            str(record.get("item_key") or ""),
            str(record.get("region") or ""),
        )
    )
    return reservoir, num_eligible_groups


def weak_query_to_eval_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    raw_region = str(record["region"])
    return {
        "id": f"weak-online-{index:07d}",
        "image": str(record["image"]),
        "query_text": str(record["query"]),
        "target_region": canonical_region(raw_region),
        "target_bbox": [float(value) for value in record["region_box"]],
        "label_status": "labeled",
        "supervision_type": "landmark_pseudo_label",
        "weak_region_variant": raw_region,
        "weak_label_source": record.get("source"),
        "weak_label_confidence": record.get("confidence"),
        "annotation": record.get("annotation"),
        "item_key": record.get("item_key"),
        "category_id": record.get("category_id"),
        "category_name": record.get("category_name"),
        "num_items_in_image": record.get("num_items_in_image"),
    }


def merge_weak_metadata(
    evaluated: list[dict[str, Any]],
    weak_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(evaluated) != len(weak_records):
        raise ValueError("Online evaluation did not preserve the weak-record count")
    merged = []
    for prediction, weak_record in zip(evaluated, weak_records, strict=True):
        merged.append(
            {
                **weak_record,
                **prediction,
                "weak_bbox_iou": prediction.get("manual_bbox_iou"),
                "evaluation_target": "landmark_pseudo_label_only",
            }
        )
    return merged


def main() -> None:
    args = parse_args()
    regions = set(args.regions)
    sampled, num_eligible_groups = sample_weak_query_records(
        args.queries,
        regions=regions,
        max_records=args.max_records,
        seed=args.seed,
        allow_multi_item_images=args.allow_multi_item_images,
    )
    if not sampled:
        raise ValueError(
            "No landmark-only weak query records matched the requested regions"
        )
    weak_records = [
        weak_query_to_eval_record(record, index) for index, record in enumerate(sampled)
    ]
    grounding_routes = parse_grounding_routes(args.grounding_routes)
    route_profiles = parse_grounding_route_profiles(args.grounding_route_profiles)
    route_thresholds = parse_grounding_route_thresholds(args.grounding_route_thresholds)
    diagnostic_routes = parse_grounding_routes(args.diagnostic_grounding_routes)
    resolved_routes = resolve_grounding_routes(
        grounding_regions=regions,
        grounding_model_name=DEFAULT_GROUNDING_MODEL_NAME,
        grounding_routes=grounding_routes,
    )
    missing_routes = regions - set(resolved_routes)
    if missing_routes:
        raise ValueError(
            "Every requested weak-training region needs an online grounding "
            f"route; missing: {', '.join(sorted(missing_routes))}"
        )
    evaluated = evaluate_gated_hybrid_records(
        weak_records,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        device=args.device,
        ranker_checkpoint=None,
        grounding_regions=regions,
        grounding_model_name=DEFAULT_GROUNDING_MODEL_NAME,
        grounding_routes=grounding_routes,
        grounding_route_profiles=route_profiles,
        grounding_route_thresholds=route_thresholds,
        grounding_backend=args.grounding_backend,
        prompt_mode=args.prompt_mode,
        prompt_profile=args.prompt_profile,
        score_threshold=args.score_threshold,
        fallback_on_no_detection=True,
        wearer_side_regions=set(args.wearer_side_regions),
        wearer_side_min_score_ratio=args.wearer_side_min_score_ratio,
        record_heuristic_candidates_for_grounding=True,
        diagnostic_grounding_routes=diagnostic_routes,
    )
    records = merge_weak_metadata(evaluated, weak_records)
    summary = {
        "queries": str(Path(args.queries)),
        "supervision_type": "landmark_pseudo_label_only",
        "candidate_generation_uses_target_bbox": False,
        "sampling_unit": "image_item_region_one_query_template",
        "seed": args.seed,
        "num_eligible_groups": num_eligible_groups,
        "num_sampled_records": len(records),
        "num_unique_images": len({record["image"] for record in records}),
        "regions": sorted(regions),
        "grounding_routes": resolved_routes,
        "diagnostic_grounding_routes": diagnostic_routes or {},
        "weak_region_variant_counts": dict(
            Counter(record["weak_region_variant"] for record in records)
        ),
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
