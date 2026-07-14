from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.data.merge_local_region_manual_eval_labels import record_key


AUDIT_INSTRUCTION = (
    "Review the existing box without predictions or landmarks. Keep a tight box only "
    "when the query identifies one visible garment part. For left/right, use garment/wearer "
    "left/right. Skip when the target is absent, occluded, or ambiguous across garments."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a low-IoU manual-label audit manifest from an evaluation JSON."
    )
    parser.add_argument("--annotations", required=True, help="Current labeled manual JSONL.")
    parser.add_argument("--eval-json", required=True, help="Manual evaluation JSON with records.")
    parser.add_argument("--output", required=True, help="Audit manifest JSONL to review.")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Optional target-region filter, e.g. cuff pocket zipper waist.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def load_eval_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {path}")
    return records


def build_audit_records(
    annotations: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    iou_threshold: float,
    regions: set[str] | None = None,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create review-ready copies of manual labels with low evaluation IoU."""
    annotation_by_key = {record_key(record): record for record in annotations}
    selected = []
    missing_annotation_keys = []

    for eval_record in eval_records:
        iou = eval_record.get("manual_bbox_iou")
        if iou is None or float(iou) >= iou_threshold:
            continue
        region = str(eval_record.get("target_region") or "unknown")
        if regions is not None and region not in regions:
            continue
        key = record_key(eval_record)
        annotation = annotation_by_key.get(key)
        if annotation is None:
            missing_annotation_keys.append(key)
            continue

        original_bbox = annotation.get("target_bbox")
        audit_record = {
            **annotation,
            "audit_original_target_bbox": list(original_bbox) if original_bbox else None,
            "audit_previous_manual_bbox_iou": float(iou),
            "audit_instruction": AUDIT_INSTRUCTION,
            "label_status": "unlabeled",
        }
        selected.append(audit_record)

    selected.sort(
        key=lambda record: (
            float(record.get("audit_previous_manual_bbox_iou") or 0.0),
            str(record.get("target_region") or ""),
            str(record.get("id") or ""),
        )
    )
    if max_records is not None:
        selected = selected[:max_records]

    summary = {
        "num_annotations": len(annotations),
        "num_eval_records": len(eval_records),
        "iou_threshold": iou_threshold,
        "regions": sorted(regions) if regions else None,
        "num_audit_records": len(selected),
        "num_missing_annotation_keys": len(missing_annotation_keys),
        "audit_instruction": AUDIT_INSTRUCTION,
    }
    return selected, summary


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    annotations = load_jsonl(args.annotations)
    eval_records = load_eval_records(args.eval_json)
    records, summary = build_audit_records(
        annotations,
        eval_records,
        iou_threshold=args.iou_threshold,
        regions=set(args.regions) if args.regions else None,
        max_records=args.max_records,
    )
    write_jsonl(records, args.output)
    summary["annotations"] = str(Path(args.annotations))
    summary["eval_json"] = str(Path(args.eval_json))
    summary["output"] = str(Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
