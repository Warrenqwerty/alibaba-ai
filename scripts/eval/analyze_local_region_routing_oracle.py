from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.compare_local_region_manual_evals import record_key
from scripts.eval.compare_local_region_manual_evals import summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the per-record routing upper bound between two completed "
            "manual local-region evaluations. This is offline analysis only."
        )
    )
    parser.add_argument("--baseline-eval-json", required=True)
    parser.add_argument("--candidate-eval-json", required=True)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Optional target_region filter, e.g. pattern pocket.",
    )
    parser.add_argument(
        "--output",
        default="outputs/local_region_routing_oracle.json",
    )
    return parser.parse_args()


def load_eval_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {path}")
    return records


def records_by_key(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key in keyed:
            raise ValueError(f"Duplicate manual-eval record key: {key}")
        keyed[key] = record
    return keyed


def manual_iou(record: dict[str, Any]) -> float:
    value = record.get("manual_bbox_iou")
    return float(value) if value is not None else 0.0


def build_routing_oracle(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    regions: set[str] | None = None,
) -> dict[str, Any]:
    """Build a per-record best-of-two upper bound from completed evaluations."""
    baseline_by_key = records_by_key(baseline_records)
    candidate_by_key = records_by_key(candidate_records)
    common_keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    if regions is not None:
        common_keys = [
            key
            for key in common_keys
            if str(candidate_by_key[key].get("target_region") or "unknown") in regions
        ]
    if not common_keys:
        raise ValueError("No common records remain after applying the region filter")

    baseline = [baseline_by_key[key] for key in common_keys]
    candidate = [candidate_by_key[key] for key in common_keys]
    oracle_records = []
    source_counts: Counter[str] = Counter()
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in common_keys:
        baseline_record = baseline_by_key[key]
        candidate_record = candidate_by_key[key]
        baseline_iou = manual_iou(baseline_record)
        candidate_iou = manual_iou(candidate_record)
        source = "candidate" if candidate_iou > baseline_iou else "baseline"
        selected = dict(candidate_record if source == "candidate" else baseline_record)
        selected["routing_oracle_source"] = source
        selected["baseline_manual_bbox_iou"] = baseline_iou
        selected["candidate_manual_bbox_iou"] = candidate_iou
        selected["routing_oracle_iou_gain"] = candidate_iou - baseline_iou
        oracle_records.append(selected)
        source_counts[source] += 1
        by_region[str(selected.get("target_region") or "unknown")].append(selected)

    return {
        "num_common_records": len(common_keys),
        "regions": sorted(regions) if regions else None,
        "baseline_summary": summarize_records(baseline),
        "candidate_summary": summarize_records(candidate),
        "per_record_oracle": {
            "summary": summarize_records(oracle_records),
            "source_counts": dict(sorted(source_counts.items())),
            "by_region": {
                region: {
                    "summary": summarize_records(records),
                    "source_counts": dict(
                        sorted(Counter(record["routing_oracle_source"] for record in records).items())
                    ),
                }
                for region, records in sorted(by_region.items())
            },
        },
        "records": oracle_records,
    }


def main() -> None:
    args = parse_args()
    result = build_routing_oracle(
        load_eval_records(args.baseline_eval_json),
        load_eval_records(args.candidate_eval_json),
        regions=set(args.regions) if args.regions else None,
    )
    output = {
        "baseline_eval_json": str(Path(args.baseline_eval_json)),
        "candidate_eval_json": str(Path(args.candidate_eval_json)),
        **result,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
