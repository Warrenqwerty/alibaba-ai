from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.compare_local_region_manual_evals import (
    load_eval,
    record_key,
    summarize_records,
)


DEFAULT_THRESHOLDS = (0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
DEFAULT_GROUNDING_REGIONS = ("pattern", "pocket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline confidence-gate analysis for the 3.1.2 gated hybrid. "
            "It reuses completed manual-eval JSONs and never reruns a model."
        )
    )
    parser.add_argument(
        "--gated-eval-json",
        required=True,
        help="Completed evaluate_gated_hybrid_manual_labels.py JSON.",
    )
    parser.add_argument(
        "--heuristic-eval-json",
        required=True,
        help="Completed heuristic-only manual evaluation JSON on the same labels.",
    )
    parser.add_argument(
        "--grounding-regions",
        nargs="+",
        default=list(DEFAULT_GROUNDING_REGIONS),
        help="Only these regions may fall back from grounding to heuristic.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Grounding confidence thresholds to evaluate.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.3,
        help="Fraction of image groups held out for threshold selection validation.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        default="outputs/local_region_gated_confidence_analysis.json",
        help="Output JSON containing calibration and image-held-out summaries.",
    )
    return parser.parse_args()


def common_records(
    gated_records: list[dict[str, Any]],
    heuristic_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    """Join two manual evaluations using their stable manual-record keys."""
    gated_by_key = _records_by_key(gated_records)
    heuristic_by_key = _records_by_key(heuristic_records)
    keys = sorted(set(gated_by_key) & set(heuristic_by_key))
    if not keys:
        raise ValueError("The two evaluations do not contain any common manual records")
    return gated_by_key, heuristic_by_key, keys


def _records_by_key(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key in keyed:
            raise ValueError(f"Duplicate manual-eval record key: {key}")
        keyed[key] = record
    return keyed


def image_group_split(
    keys: list[str],
    records: dict[str, dict[str, Any]],
    *,
    grounding_regions: set[str],
    holdout_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split by image, stratified by target region, to limit visual leakage."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    images_by_region: dict[str, set[str]] = defaultdict(set)
    for key in keys:
        record = records[key]
        region = str(record.get("target_region") or "unknown")
        if region in grounding_regions:
            images_by_region[region].add(str(record.get("image") or ""))

    rng = random.Random(seed)
    holdout_images: set[str] = set()
    for images in images_by_region.values():
        ordered = sorted(images)
        rng.shuffle(ordered)
        if len(ordered) < 2:
            continue
        num_holdout = max(1, round(len(ordered) * holdout_fraction))
        num_holdout = min(num_holdout, len(ordered) - 1)
        holdout_images.update(ordered[:num_holdout])

    calibration_keys = {
        key for key in keys if str(records[key].get("image") or "") not in holdout_images
    }
    holdout_keys = set(keys) - calibration_keys
    if not holdout_keys or not calibration_keys:
        raise ValueError("Could not create non-empty calibration and holdout image groups")
    return calibration_keys, holdout_keys


def confidence_gated_records(
    keys: list[str] | set[str],
    *,
    gated_by_key: dict[str, dict[str, Any]],
    heuristic_by_key: dict[str, dict[str, Any]],
    grounding_regions: set[str],
    confidence_threshold: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Simulate confidence fallback without rerunning model inference."""
    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for key in sorted(keys):
        gated = gated_by_key[key]
        region = str(gated.get("target_region") or "unknown")
        score = gated.get("score")
        use_grounding = (
            region in grounding_regions
            and gated.get("gated_policy_route") == "grounding"
            and isinstance(score, int | float)
            and float(score) >= confidence_threshold
        )
        if use_grounding:
            record = dict(gated)
            record["confidence_gate_source"] = "grounding"
            source_counts["grounding"] += 1
        elif region in grounding_regions:
            record = dict(heuristic_by_key[key])
            record["confidence_gate_source"] = "heuristic_fallback"
            record["grounding_score"] = score
            source_counts["heuristic_fallback"] += 1
        else:
            record = dict(gated)
            record["confidence_gate_source"] = "unchanged_gated_policy"
            source_counts["unchanged_gated_policy"] += 1
        records.append(record)
    return records, source_counts


def semantic_records(
    records: list[dict[str, Any]],
    grounding_regions: set[str],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if str(record.get("target_region") or "unknown") in grounding_regions
    ]


def threshold_result(
    threshold: float,
    keys: list[str] | set[str],
    *,
    gated_by_key: dict[str, dict[str, Any]],
    heuristic_by_key: dict[str, dict[str, Any]],
    grounding_regions: set[str],
) -> dict[str, Any]:
    records, source_counts = confidence_gated_records(
        keys,
        gated_by_key=gated_by_key,
        heuristic_by_key=heuristic_by_key,
        grounding_regions=grounding_regions,
        confidence_threshold=threshold,
    )
    semantic = semantic_records(records, grounding_regions)
    return {
        "confidence_threshold": threshold,
        "num_records": len(records),
        "num_semantic_records": len(semantic),
        "source_counts": dict(sorted(source_counts.items())),
        "summary": summarize_records(records),
        "semantic_summary": summarize_records(semantic),
    }


def choose_best_threshold(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose by calibration semantic IoU, then Hit@0.3/0.5, then lower threshold."""
    if not results:
        raise ValueError("At least one threshold result is required")
    return max(
        results,
        key=lambda result: (
            result["semantic_summary"]["avg_manual_bbox_iou"],
            result["semantic_summary"]["manual_hit_at"]["0.3"],
            result["semantic_summary"]["manual_hit_at"]["0.5"],
            -result["confidence_threshold"],
        ),
    )


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.thresholds))
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("All thresholds must be between 0 and 1")
    grounding_regions = set(args.grounding_regions)
    gated_eval = load_eval(args.gated_eval_json, name="gated")
    heuristic_eval = load_eval(args.heuristic_eval_json, name="heuristic")
    gated_by_key, heuristic_by_key, keys = common_records(
        gated_eval["records"],
        heuristic_eval["records"],
    )
    calibration_keys, holdout_keys = image_group_split(
        keys,
        gated_by_key,
        grounding_regions=grounding_regions,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    calibration = [
        threshold_result(
            threshold,
            calibration_keys,
            gated_by_key=gated_by_key,
            heuristic_by_key=heuristic_by_key,
            grounding_regions=grounding_regions,
        )
        for threshold in thresholds
    ]
    selected = choose_best_threshold(calibration)
    selected_threshold = selected["confidence_threshold"]
    holdout_results = [
        threshold_result(
            threshold,
            holdout_keys,
            gated_by_key=gated_by_key,
            heuristic_by_key=heuristic_by_key,
            grounding_regions=grounding_regions,
        )
        for threshold in thresholds
    ]
    holdout = next(
        result
        for result in holdout_results
        if result["confidence_threshold"] == selected_threshold
    )
    output = {
        "gated_eval_json": str(Path(args.gated_eval_json)),
        "heuristic_eval_json": str(Path(args.heuristic_eval_json)),
        "grounding_regions": sorted(grounding_regions),
        "thresholds": thresholds,
        "split": {
            "unit": "image",
            "seed": args.seed,
            "holdout_fraction": args.holdout_fraction,
            "num_common_records": len(keys),
            "num_calibration_records": len(calibration_keys),
            "num_holdout_records": len(holdout_keys),
        },
        "calibration_results": calibration,
        "selected_threshold": selected_threshold,
        "selected_calibration_result": selected,
        "holdout_results": holdout_results,
        "selected_threshold_holdout_result": holdout,
        "interpretation": (
            "Exploratory offline threshold analysis. Change the online policy only "
            "if the selected threshold improves the image-held-out semantic result "
            "and the full manual benchmark is rerun afterwards."
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
