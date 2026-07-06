from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MANUAL_IOU_THRESHOLDS = (0.3, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more 3.1.2 manual-eval JSON files and estimate a "
            "simple per-region hybrid oracle."
        )
    )
    parser.add_argument(
        "--eval-json",
        nargs="+",
        required=True,
        help="Manual-eval JSON files containing a top-level records list.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Optional short names for the eval files, in the same order.",
    )
    parser.add_argument(
        "--output",
        default="outputs/local_region_manual_eval_comparison.json",
        help="Path to save comparison summary.",
    )
    return parser.parse_args()


def load_eval(path: str | Path, name: str | None = None) -> dict[str, Any]:
    eval_path = Path(path)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{eval_path} must contain top-level records list")
    return {
        "name": name or eval_path.stem,
        "path": str(eval_path),
        "summary": {key: value for key, value in payload.items() if key != "records"},
        "records": records,
    }


def compare_evals(evals: list[dict[str, Any]]) -> dict[str, Any]:
    keyed_records = {evaluation["name"]: records_by_key(evaluation["records"]) for evaluation in evals}
    common_keys = sorted(set.intersection(*(set(records) for records in keyed_records.values())))
    if not common_keys:
        raise ValueError("No common records found across eval JSON files")

    per_eval = {
        evaluation["name"]: summarize_records(
            [keyed_records[evaluation["name"]][key] for key in common_keys]
        )
        for evaluation in evals
    }
    per_region = compare_by_region(evals, keyed_records, common_keys)
    region_policy = {
        region: region_summary["best_eval"]
        for region, region_summary in per_region.items()
    }
    hybrid_records = build_region_hybrid_records(keyed_records, common_keys, region_policy)

    return {
        "evals": [
            {
                "name": evaluation["name"],
                "path": evaluation["path"],
            }
            for evaluation in evals
        ],
        "num_common_records": len(common_keys),
        "per_eval": per_eval,
        "per_region": per_region,
        "region_policy": region_policy,
        "region_hybrid_oracle": summarize_records(hybrid_records),
        "records": hybrid_records,
    }


def records_by_key(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key in keyed:
            raise ValueError(f"Duplicate manual-eval record key: {key}")
        keyed[key] = record
    return keyed


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("image", "")),
        str(record.get("query_text", "")),
        str(record.get("target_region", "")),
    )


def compare_by_region(
    evals: list[dict[str, Any]],
    keyed_records: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
    common_keys: list[tuple[str, str, str]],
) -> dict[str, Any]:
    keys_by_region: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in common_keys:
        keys_by_region[key[2] or "unknown"].append(key)

    per_region = {}
    for region, region_keys in sorted(keys_by_region.items()):
        eval_summaries = {
            evaluation["name"]: summarize_records(
                [keyed_records[evaluation["name"]][key] for key in region_keys]
            )
            for evaluation in evals
        }
        best_eval = max(
            eval_summaries,
            key=lambda name: (
                eval_summaries[name]["avg_manual_bbox_iou"],
                eval_summaries[name]["manual_hit_at"]["0.3"],
                eval_summaries[name]["manual_hit_at"]["0.5"],
            ),
        )
        per_region[region] = {
            "num_records": len(region_keys),
            "best_eval": best_eval,
            "evals": eval_summaries,
        }
    return per_region


def build_region_hybrid_records(
    keyed_records: dict[str, dict[tuple[str, str, str], dict[str, Any]]],
    common_keys: list[tuple[str, str, str]],
    region_policy: dict[str, str],
) -> list[dict[str, Any]]:
    records = []
    for key in common_keys:
        region = key[2] or "unknown"
        source_name = region_policy[region]
        record = dict(keyed_records[source_name][key])
        record["hybrid_source_eval"] = source_name
        records.append(record)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [manual_iou(record) for record in records]
    return {
        "num_records": len(records),
        "avg_manual_bbox_iou": mean(ious) if ious else 0.0,
        "manual_hit_at": {
            str(threshold): hit_rate(ious, threshold)
            for threshold in MANUAL_IOU_THRESHOLDS
        },
    }


def manual_iou(record: dict[str, Any]) -> float:
    value = record.get("manual_bbox_iou")
    return float(value) if value is not None else 0.0


def hit_rate(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value >= threshold for value in values) / len(values)


def main() -> None:
    args = parse_args()
    if args.names is not None and len(args.names) != len(args.eval_json):
        raise ValueError("--names must have the same length as --eval-json")
    names = args.names or [None] * len(args.eval_json)
    evals = [load_eval(path, name=name) for path, name in zip(args.eval_json, names, strict=True)]
    comparison = compare_evals(evals)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in comparison.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
