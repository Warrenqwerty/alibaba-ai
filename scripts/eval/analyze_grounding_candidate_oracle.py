from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.local_region import box_iou
from scripts.eval.evaluate_local_region_manual_labels import summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the manual-label oracle ceiling of grounding detections "
            "already saved in a gated evaluation JSON."
        )
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--hit-threshold", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def grounding_detections(record: dict[str, Any]) -> list[dict[str, Any]]:
    detections = record.get("detections")
    if isinstance(detections, list):
        return detections
    fallback = record.get("grounding_detections")
    return fallback if isinstance(fallback, list) else []


def manual_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        {
            **detection,
            "candidate_source": "grounding",
            "candidate_rank": rank,
        }
        for rank, detection in enumerate(grounding_detections(record), start=1)
        if detection.get("bbox") is not None
    ]
    heuristic = record.get("heuristic_candidate")
    if isinstance(heuristic, dict) and heuristic.get("predicted_bbox") is not None:
        candidates.append(
            {
                "bbox": heuristic["predicted_bbox"],
                "prompt": heuristic.get("selected_region"),
                "score": None,
                "candidate_source": "heuristic",
                "candidate_rank": None,
            }
        )
    return candidates


def best_manual_candidate(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, float]:
    scored = [
        (candidate, box_iou(candidate["bbox"], record["target_bbox"]))
        for candidate in manual_candidates(record)
    ]
    if not scored:
        return None, 0.0
    return max(scored, key=lambda item: item[1])


def build_candidate_oracle(
    records: list[dict[str, Any]],
    *,
    regions: set[str],
    hit_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oracle_records = []
    diagnostics: dict[str, dict[str, int]] = {
        region: Counter() for region in sorted(regions)
    }
    oracle_rank_counts: Counter[str] = Counter()
    oracle_source_counts: Counter[str] = Counter()

    for record in records:
        updated = dict(record)
        region = str(record.get("target_region") or "")
        if region not in regions:
            oracle_records.append(updated)
            continue

        region_stats = diagnostics[region]
        region_stats["num_records"] += 1
        selected_iou = float(record.get("manual_bbox_iou") or 0.0)
        if selected_iou >= hit_threshold:
            region_stats["selected_hits"] += 1

        if grounding_detections(record):
            region_stats["records_with_grounding_candidates"] += 1
        heuristic = record.get("heuristic_candidate")
        if isinstance(heuristic, dict) and heuristic.get("predicted_bbox") is not None:
            region_stats["records_with_heuristic_candidate"] += 1
        candidate, candidate_iou = best_manual_candidate(record)
        if candidate is not None:
            region_stats["records_with_candidates"] += 1
        oracle_iou = max(selected_iou, candidate_iou)
        if oracle_iou >= hit_threshold:
            region_stats["oracle_hits"] += 1
            if selected_iou < hit_threshold:
                region_stats["recoverable_failures"] += 1
        if candidate is not None and candidate_iou > selected_iou:
            candidate_source = str(candidate["candidate_source"])
            oracle_source_counts[f"{candidate_source}_candidate"] += 1
            region_stats[f"oracle_selected_{candidate_source}"] += 1
            rank = candidate.get("candidate_rank")
            if rank is not None:
                oracle_rank_counts[str(rank)] += 1
            updated.update(
                {
                    "predicted_bbox": [float(value) for value in candidate["bbox"]],
                    "manual_bbox_iou": candidate_iou,
                    "selected_region": candidate.get("prompt"),
                    "score": candidate.get("score"),
                    "candidate_oracle_rank": rank,
                    "candidate_oracle_source": candidate_source,
                }
            )
        else:
            oracle_source_counts["current_selection"] += 1
        oracle_records.append(updated)

    return oracle_records, {
        "by_region": {
            region: dict(counts) for region, counts in diagnostics.items()
        },
        "oracle_rank_counts": dict(oracle_rank_counts),
        "oracle_source_counts": dict(oracle_source_counts),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {args.eval_json}")
    oracle_records, diagnostics = build_candidate_oracle(
        records,
        regions=set(args.regions),
        hit_threshold=args.hit_threshold,
    )
    result = {
        "eval_json": str(Path(args.eval_json)),
        "regions": args.regions,
        "hit_threshold": args.hit_threshold,
        "baseline_summary": summarize_records(records),
        "candidate_oracle_summary": summarize_records(oracle_records),
        **diagnostics,
        "records": oracle_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
