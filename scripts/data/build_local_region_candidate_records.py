#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fashion_mm.data_loaders.local_region_queries import LocalRegionQueryRecord
from fashion_mm.data_loaders.local_region_queries import iter_local_region_query_records
from fashion_mm.models.local_region.learned_ranker import box_iou
from fashion_mm.models.local_region.learned_ranker import candidate_boxes_from_garment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand weak local-region query records into candidate box records "
            "for vision-language region ranker training."
        )
    )
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="Input weak local-region query JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output candidate-level JSONL path.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional maximum number of query records to process.",
    )
    parser.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="Number of input query records to skip before exporting.",
    )
    parser.add_argument(
        "--positive-iou-threshold",
        type=float,
        default=0.5,
        help="Candidate IoU threshold used to set label=1.",
    )
    return parser.parse_args()


def record_to_candidate_payloads(
    record: LocalRegionQueryRecord,
    *,
    positive_iou_threshold: float,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for candidate in candidate_boxes_from_garment(record.garment_box):
        iou = box_iou(candidate.box, record.region_box)
        payloads.append(
            {
                "image": str(record.image),
                "annotation": str(record.annotation),
                "item_key": record.item_key,
                "query": record.query,
                "target_region": record.region,
                "target_region_box": list(record.region_box),
                "garment_box": list(record.garment_box),
                "candidate_region": candidate.region,
                "candidate_box": list(candidate.box),
                "iou": iou,
                "label": int(iou >= positive_iou_threshold),
                "weak_label_source": record.source,
                "weak_label_confidence": record.confidence,
                "category_id": record.category_id,
                "category_name": record.category_name,
            }
        )
    return payloads


def build_candidate_records(
    records_path: Path,
    output_path: Path,
    *,
    max_records: int | None,
    skip_records: int,
    positive_iou_threshold: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_region_counts: Counter[str] = Counter()
    target_region_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    query_records = 0
    candidate_records = 0
    iou_sum = 0.0

    with output_path.open("w", encoding="utf-8") as file:
        for record in iter_local_region_query_records(
            records_path,
            max_records=max_records,
            skip_records=skip_records,
        ):
            query_records += 1
            target_region_counts[record.region] += 1
            source_counts[record.source] += 1
            for payload in record_to_candidate_payloads(
                record,
                positive_iou_threshold=positive_iou_threshold,
            ):
                candidate_records += 1
                candidate_region_counts[str(payload["candidate_region"])] += 1
                label_counts[int(payload["label"])] += 1
                iou_sum += float(payload["iou"])
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return {
        "input": str(records_path),
        "output": str(output_path),
        "num_query_records": query_records,
        "num_candidate_records": candidate_records,
        "positive_iou_threshold": positive_iou_threshold,
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "target_region_counts": dict(target_region_counts),
        "candidate_region_counts": dict(candidate_region_counts),
        "weak_label_source_counts": dict(source_counts),
        "avg_candidate_iou": (
            iou_sum / candidate_records if candidate_records > 0 else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    summary = build_candidate_records(
        args.records,
        args.output,
        max_records=args.max_records,
        skip_records=args.skip_records,
        positive_iou_threshold=args.positive_iou_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
