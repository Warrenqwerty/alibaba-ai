from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TARGET_REGIONS = ("pattern", "neckline", "hem", "shoulder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible qualitative-demo manifest from a completed "
            "gated-hybrid manual evaluation JSON. Only records meeting the "
            "manual IoU threshold are selected."
        )
    )
    parser.add_argument(
        "--eval-json",
        required=True,
        help="Output JSON from evaluate_gated_hybrid_manual_labels.py.",
    )
    parser.add_argument(
        "--target-regions",
        nargs="+",
        default=list(DEFAULT_TARGET_REGIONS),
        help="Regions to show, in manifest order.",
    )
    parser.add_argument(
        "--per-region",
        type=int,
        default=2,
        help="Maximum successful records selected for each target region.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.3,
        help="Minimum manual bbox IoU required for a qualitative demo record.",
    )
    parser.add_argument(
        "--require-full-quota",
        action="store_true",
        help="Fail unless every requested region supplies --per-region records.",
    )
    parser.add_argument(
        "--output",
        default="outputs/local_region_gated_demo_manifest.jsonl",
        help="Output JSONL accepted by evaluate_gated_hybrid_queries.py --manifest.",
    )
    return parser.parse_args()


def load_evaluation_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate per-query records from a gated manual evaluation."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{source} must contain a top-level records list")
    return [record for record in records if isinstance(record, dict)]


def select_demo_records(
    records: list[dict[str, Any]],
    *,
    target_regions: list[str],
    per_region: int,
    min_iou: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Choose highest-IoU successful records independently for each region."""
    if per_region <= 0:
        raise ValueError("per_region must be positive")
    if not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be between 0 and 1")

    selected: list[dict[str, Any]] = []
    available_counts: dict[str, int] = {}
    seen_keys: set[tuple[str, str, str]] = set()
    for target_region in target_regions:
        candidates = [
            record
            for record in records
            if record.get("target_region") == target_region
            and record.get("status") == "ok"
            and record.get("predicted_bbox") is not None
            and float(record.get("manual_bbox_iou") or 0.0) >= min_iou
            and isinstance(record.get("image"), str)
            and isinstance(record.get("query_text"), str)
        ]
        candidates.sort(
            key=lambda record: (
                -float(record.get("manual_bbox_iou") or 0.0),
                str(record.get("id") or ""),
            )
        )
        available_counts[target_region] = len(candidates)
        region_count = 0
        for record in candidates:
            key = (
                str(record["image"]),
                str(record["query_text"]),
                target_region,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(record)
            region_count += 1
            if region_count >= per_region:
                break
    return selected, available_counts


def manifest_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Keep provenance so a qualitative demo cannot be mistaken for a benchmark."""
    target_region = str(record["target_region"])
    return {
        "id": f"manual_demo_{target_region}_{index:03d}",
        "image": str(record["image"]),
        "query_text": str(record["query_text"]),
        "target_region": target_region,
        "reference_bbox": list(record["target_bbox"]),
        "selection_source": "gated_hybrid_manual_evaluation",
        "selection_manual_bbox_iou": float(record["manual_bbox_iou"]),
        "gated_policy_route": record.get("gated_policy_route"),
        "note": "Qualitative example selected by manual-benchmark IoU; not an aggregate metric.",
    }


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summary(
    *,
    eval_json: str | Path,
    target_regions: list[str],
    selected: list[dict[str, Any]],
    available_counts: dict[str, int],
    min_iou: float,
    output: Path,
) -> dict[str, Any]:
    selected_by_region = Counter(record["target_region"] for record in selected)
    route_counts = Counter(record.get("gated_policy_route") for record in selected)
    return {
        "eval_json": str(Path(eval_json)),
        "output": str(output),
        "target_regions": target_regions,
        "min_iou": min_iou,
        "num_records": len(selected),
        "available_qualified_records_by_region": available_counts,
        "selected_records_by_region": {
            region: selected_by_region.get(region, 0) for region in target_regions
        },
        "gated_policy_route_counts": dict(sorted(route_counts.items())),
        "selection_note": (
            "Qualitative records are selected from the completed manual benchmark "
            "by IoU and must not be reported as aggregate performance."
        ),
    }


def main() -> None:
    args = parse_args()
    records = load_evaluation_records(args.eval_json)
    selected, available_counts = select_demo_records(
        records,
        target_regions=list(args.target_regions),
        per_region=args.per_region,
        min_iou=args.min_iou,
    )
    selected_counts = Counter(record["target_region"] for record in selected)
    if args.require_full_quota:
        missing = [
            region
            for region in args.target_regions
            if selected_counts.get(region, 0) < args.per_region
        ]
        if missing:
            raise ValueError(
                "Not enough qualified records for: " + ", ".join(missing)
            )
    if not selected:
        raise ValueError("No records met the requested qualitative-demo threshold")

    output_path = Path(args.output)
    write_jsonl(
        [manifest_record(record, index) for index, record in enumerate(selected)],
        output_path,
    )
    print(
        json.dumps(
            summary(
                eval_json=args.eval_json,
                target_regions=list(args.target_regions),
                selected=selected,
                available_counts=available_counts,
                min_iou=args.min_iou,
                output=output_path,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
