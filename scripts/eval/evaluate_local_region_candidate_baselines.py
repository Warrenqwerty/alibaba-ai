from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fashion_mm.data_loaders import LocalRegionCandidateRecord
from fashion_mm.data_loaders import iter_local_region_candidate_records


SelectionFn = Callable[[list[LocalRegionCandidateRecord]], LocalRegionCandidateRecord]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate non-CLIP baselines on local-region candidate JSONL."
    )
    parser.add_argument("--candidates", required=True, help="Candidate JSONL path.")
    parser.add_argument("--max-groups", type=int, default=2000)
    parser.add_argument("--skip-groups", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_candidate_baselines(
        args.candidates,
        max_groups=args.max_groups,
        skip_groups=args.skip_groups,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def evaluate_candidate_baselines(
    candidates_path: str | Path,
    *,
    max_groups: int | None,
    skip_groups: int,
) -> dict[str, Any]:
    groups = iter_candidate_groups(
        candidates_path,
        max_groups=max_groups,
        skip_groups=skip_groups,
    )
    evaluators: dict[str, SelectionFn] = {
        "oracle_best_iou": select_oracle_best_iou,
        "target_region_name": select_target_region_name,
        "whole_garment": lambda group: select_first_region(group, "whole_garment"),
        "upper": lambda group: select_first_region(group, "upper"),
        "lower": lambda group: select_first_region(group, "lower"),
        "center": lambda group: select_first_region(group, "center"),
    }
    summary = {
        name: _empty_summary()
        for name in evaluators
    }
    num_groups = 0
    target_region_counts: Counter[str] = Counter()
    for group in groups:
        num_groups += 1
        target_region_counts[group[0].target_region] += 1
        for name, selector in evaluators.items():
            selected = selector(group)
            _update_summary(summary[name], selected)

    return {
        "candidates": str(candidates_path),
        "num_groups": num_groups,
        "target_region_counts": dict(target_region_counts),
        "baselines": {
            name: _finalize_summary(values)
            for name, values in summary.items()
        },
    }


def iter_candidate_groups(
    jsonl_path: str | Path,
    *,
    max_groups: int | None = None,
    skip_groups: int = 0,
) -> Iterator[list[LocalRegionCandidateRecord]]:
    """Stream adjacent candidate rows grouped by one query target."""
    current_key: tuple[Any, ...] | None = None
    current_group: list[LocalRegionCandidateRecord] = []
    seen_groups = 0
    yielded_groups = 0

    for record in iter_local_region_candidate_records(jsonl_path):
        key = _group_key(record)
        if current_key is None:
            current_key = key
        if key != current_key:
            if seen_groups >= skip_groups:
                yield current_group
                yielded_groups += 1
                if max_groups is not None and yielded_groups >= max_groups:
                    return
            seen_groups += 1
            current_key = key
            current_group = []
        current_group.append(record)

    if current_group and seen_groups >= skip_groups:
        yield current_group


def select_oracle_best_iou(
    group: list[LocalRegionCandidateRecord],
) -> LocalRegionCandidateRecord:
    return max(group, key=lambda record: record.iou)


def select_target_region_name(
    group: list[LocalRegionCandidateRecord],
) -> LocalRegionCandidateRecord:
    target_region = group[0].target_region
    for record in group:
        if record.candidate_region == target_region:
            return record
    return select_oracle_best_iou(group)


def select_first_region(
    group: list[LocalRegionCandidateRecord],
    region: str,
) -> LocalRegionCandidateRecord:
    for record in group:
        if record.candidate_region == region:
            return record
    return group[0]


def _empty_summary() -> dict[str, Any]:
    return {
        "num_records": 0,
        "iou_sum": 0.0,
        "weak_hit_at": {"0.3": 0, "0.5": 0},
        "selected_region_counts": Counter(),
        "by_region": {},
    }


def _update_summary(
    summary: dict[str, Any],
    selected: LocalRegionCandidateRecord,
) -> None:
    target_region = selected.target_region
    iou = selected.iou
    summary["num_records"] += 1
    summary["iou_sum"] += iou
    summary["selected_region_counts"][selected.candidate_region] += 1
    for threshold in summary["weak_hit_at"]:
        summary["weak_hit_at"][threshold] += int(iou >= float(threshold))

    region_summary = summary["by_region"].setdefault(target_region, _empty_summary())
    region_summary["num_records"] += 1
    region_summary["iou_sum"] += iou
    region_summary["selected_region_counts"][selected.candidate_region] += 1
    for threshold in region_summary["weak_hit_at"]:
        region_summary["weak_hit_at"][threshold] += int(iou >= float(threshold))


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    num_records = int(summary["num_records"])
    return {
        "num_records": num_records,
        "avg_top1_iou": summary["iou_sum"] / max(num_records, 1),
        "weak_hit_at": {
            threshold: count / max(num_records, 1)
            for threshold, count in summary["weak_hit_at"].items()
        },
        "selected_region_counts": dict(summary["selected_region_counts"]),
        "by_region": {
            region: _finalize_summary(values)
            for region, values in summary["by_region"].items()
        },
    }


def _group_key(record: LocalRegionCandidateRecord) -> tuple[Any, ...]:
    return (
        record.image,
        record.annotation,
        record.item_key,
        record.query,
        record.target_region,
        record.target_region_box,
        record.garment_box,
    )


if __name__ == "__main__":
    main()
