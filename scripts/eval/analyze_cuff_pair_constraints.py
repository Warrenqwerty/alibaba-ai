from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import desired_image_side
from fashion_mm.models.local_region import query_wearer_side
from scripts.eval.cross_validate_grounding_candidate_selector import (
    online_garment_instance,
)
from scripts.eval.cross_validate_grounding_candidate_selector import selector_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether garment-side and paired-cuff constraints can "
            "recover existing candidate-selection failures."
        )
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--hit-threshold", type=float, default=0.3)
    parser.add_argument("--pair-max-iou", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def record_wearer_side(record: dict[str, Any]) -> str | None:
    side = query_wearer_side(str(record.get("query_text") or ""))
    if side is not None:
        return side
    variant = str(record.get("weak_region_variant") or "")
    if variant.startswith("left_"):
        return "left"
    if variant.startswith("right_"):
        return "right"
    return None


def candidate_matches_wearer_side(
    record: dict[str, Any],
    box: list[float] | tuple[float, ...],
) -> bool | None:
    wearer_side = record_wearer_side(record)
    instance = online_garment_instance(record)
    if wearer_side is None or instance is None:
        return None
    garment_box = instance["box"]
    garment_center_x = (float(garment_box[0]) + float(garment_box[2])) * 0.5
    center_x = (float(box[0]) + float(box[2])) * 0.5
    image_side = "left" if center_x < garment_center_x else "right"
    return image_side == desired_image_side(wearer_side)


def candidate_iou(
    record: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    return box_iou(candidate["bbox"], record["target_bbox"])


def hit_candidate_count(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    hit_threshold: float,
) -> int:
    return sum(
        candidate_iou(record, candidate) >= hit_threshold
        for candidate in candidates
    )


def paired_cuff_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("image") or ""), str(record.get("item_key") or "")


def side_distinct_pair_oracle(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    *,
    hit_threshold: float,
    pair_max_iou: float,
) -> tuple[bool, bool]:
    left_candidates = [
        candidate
        for candidate in selector_candidates(left_record)
        if candidate_matches_wearer_side(left_record, candidate["bbox"]) is True
    ]
    right_candidates = [
        candidate
        for candidate in selector_candidates(right_record)
        if candidate_matches_wearer_side(right_record, candidate["bbox"]) is True
    ]
    best_hits = 0
    for left_candidate in left_candidates:
        for right_candidate in right_candidates:
            if box_iou(left_candidate["bbox"], right_candidate["bbox"]) >= pair_max_iou:
                continue
            hits = int(
                candidate_iou(left_record, left_candidate) >= hit_threshold
            ) + int(candidate_iou(right_record, right_candidate) >= hit_threshold)
            best_hits = max(best_hits, hits)
    return best_hits >= 1, best_hits == 2


def analyze_cuff_constraints(
    records: list[dict[str, Any]],
    *,
    hit_threshold: float,
    pair_max_iou: float,
) -> dict[str, Any]:
    if not 0.0 <= hit_threshold <= 1.0:
        raise ValueError("hit_threshold must be between 0 and 1")
    if not 0.0 <= pair_max_iou <= 1.0:
        raise ValueError("pair_max_iou must be between 0 and 1")

    cuff_records = [
        record
        for record in records
        if str(record.get("target_region") or "") == "cuff"
    ]
    selected = defaultdict(int)
    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in cuff_records:
        selected["num_records"] += 1
        predicted_box = record.get("predicted_bbox")
        if predicted_box is None:
            selected["records_without_prediction"] += 1
            continue
        selected_iou = box_iou(predicted_box, record["target_bbox"])
        selected_hit = selected_iou >= hit_threshold
        selected["selected_hits"] += int(selected_hit)

        side_match = candidate_matches_wearer_side(record, predicted_box)
        if side_match is None:
            selected["records_without_side_or_garment_geometry"] += 1
        elif side_match:
            selected["selected_side_matches"] += 1
            selected["selected_side_match_hits"] += int(selected_hit)
        else:
            selected["selected_side_mismatches"] += 1
            selected["selected_side_mismatch_hits"] += int(selected_hit)

        candidates = selector_candidates(record)
        compatible = [
            candidate
            for candidate in candidates
            if candidate_matches_wearer_side(record, candidate["bbox"]) is True
        ]
        if compatible:
            selected["records_with_side_compatible_candidates"] += 1
        full_has_hit = hit_candidate_count(
            record,
            candidates,
            hit_threshold=hit_threshold,
        ) > 0
        compatible_has_hit = hit_candidate_count(
            record,
            compatible,
            hit_threshold=hit_threshold,
        ) > 0
        selected["full_candidate_oracle_hits"] += int(full_has_hit)
        selected["side_compatible_oracle_hits"] += int(compatible_has_hit)
        if not selected_hit and compatible_has_hit:
            selected["selected_misses_recoverable_on_compatible_side"] += 1
            if side_match is False:
                selected["wrong_side_misses_recoverable_on_compatible_side"] += 1
        if selected_hit and side_match is False:
            selected["selected_hits_on_incompatible_side"] += 1

        side = record_wearer_side(record)
        if side in {"left", "right"}:
            pairs[paired_cuff_key(record)][side] = record

    pair_counts = defaultdict(int)
    for pair in pairs.values():
        if set(pair) != {"left", "right"}:
            continue
        pair_counts["num_complete_pairs"] += 1
        left_record = pair["left"]
        right_record = pair["right"]
        left_box = left_record.get("predicted_bbox")
        right_box = right_record.get("predicted_bbox")
        if left_box is None or right_box is None:
            pair_counts["pairs_without_both_predictions"] += 1
            continue
        left_hit = box_iou(left_box, left_record["target_bbox"]) >= hit_threshold
        right_hit = box_iou(right_box, right_record["target_bbox"]) >= hit_threshold
        pair_counts["selected_pairs_with_any_hit"] += int(left_hit or right_hit)
        pair_counts["selected_pairs_with_both_hits"] += int(left_hit and right_hit)
        pair_counts["selected_pair_box_collisions"] += int(
            box_iou(left_box, right_box) >= pair_max_iou
        )
        both_side_matches = (
            candidate_matches_wearer_side(left_record, left_box) is True
            and candidate_matches_wearer_side(right_record, right_box) is True
        )
        pair_counts["selected_pairs_with_both_side_matches"] += int(
            both_side_matches
        )
        oracle_any, oracle_both = side_distinct_pair_oracle(
            left_record,
            right_record,
            hit_threshold=hit_threshold,
            pair_max_iou=pair_max_iou,
        )
        pair_counts["side_distinct_oracle_pairs_with_any_hit"] += int(oracle_any)
        pair_counts["side_distinct_oracle_pairs_with_both_hits"] += int(oracle_both)

    return {
        "hit_threshold": hit_threshold,
        "pair_max_iou": pair_max_iou,
        "diagnostic_only": True,
        "target_bbox_usage": "metrics_and_oracle_only",
        "selected_and_side_candidate_summary": dict(selected),
        "paired_cuff_summary": dict(pair_counts),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {args.eval_json}")
    result = {
        "eval_json": str(Path(args.eval_json)),
        **analyze_cuff_constraints(
            records,
            hit_threshold=args.hit_threshold,
            pair_max_iou=args.pair_max_iou,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
