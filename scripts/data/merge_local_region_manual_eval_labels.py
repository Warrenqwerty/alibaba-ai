from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge labeled local-region manual-eval JSONL files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input labeled JSONL files. Later files win duplicate keys.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output merged labeled JSONL.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Keep skipped records in the output. Unlabeled records are always dropped.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def merge_labeled_records(
    input_paths: list[str | Path],
    *,
    include_skipped: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge manual labels, dropping unlabeled rows and deduplicating records."""
    merged_by_key: dict[str, dict[str, Any]] = {}
    input_counts: dict[str, int] = {}
    status_counts: Counter[str] = Counter()
    duplicate_count = 0

    for input_path in input_paths:
        path = Path(input_path)
        records = load_jsonl(path)
        input_counts[str(path)] = len(records)
        for record in records:
            status = str(record.get("label_status", "unlabeled"))
            status_counts[status] += 1
            if status == "skip" and include_skipped:
                pass
            elif status != "labeled" or not record.get("target_bbox"):
                continue

            key = record_key(record)
            duplicate_count += int(key in merged_by_key)
            merged_by_key[key] = {**record, "merge_source": str(path)}

    merged = sorted(
        merged_by_key.values(),
        key=lambda record: (
            str(record.get("image", "")),
            str(record.get("target_region", "")),
            str(record.get("query_text", "")),
            str(record.get("id", "")),
        ),
    )
    summary = {
        "inputs": [str(Path(path)) for path in input_paths],
        "input_record_counts": input_counts,
        "input_label_status_counts": dict(status_counts),
        "num_merged_records": len(merged),
        "num_duplicate_keys_replaced": duplicate_count,
        "dedupe_key": "id when present, otherwise image/query_text/target_region",
    }
    return merged, summary


def record_key(record: dict[str, Any]) -> str:
    """Return the merge key for one manual label record."""
    record_id = record.get("id")
    if isinstance(record_id, str) and record_id:
        return f"id:{record_id}"
    return "fallback:" + "\t".join(
        (
            str(record.get("image", "")),
            str(record.get("query_text", "")),
            str(record.get("target_region", "")),
        )
    )


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    merged, summary = merge_labeled_records(
        args.inputs,
        include_skipped=args.include_skipped,
    )
    write_jsonl(merged, args.output)
    summary["output"] = str(Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
