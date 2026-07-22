from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fashion_mm.data_loaders.fashionai_attributes import (
    deduplicate_fashionai_records,
)
from fashion_mm.data_loaders.fashionai_attributes import FashionAIAttributeRecord
from fashion_mm.data_loaders.fashionai_attributes import infer_fashionai_schema
from fashion_mm.data_loaders.fashionai_attributes import read_fashionai_annotations
from fashion_mm.data_loaders.fashionai_attributes import stratified_split_records
from fashion_mm.utils.config import load_yaml


def prepare_fashionai_round1_splits(
    *,
    source_a_root: Path,
    source_b_root: Path,
    answer_a: Path,
    answer_b: Path,
    output_dir: Path,
    split_fractions: dict[str, float],
    seed: int,
    label_map: Path | None,
    validate_images: bool,
) -> dict[str, Any]:
    """Merge labeled Round1 A/B sources and write leak-free split manifests."""
    records_a = read_fashionai_annotations(
        answer_a,
        image_root=source_a_root,
        validate_images=validate_images,
        source_name="round1_test_a",
    )
    records_b = read_fashionai_annotations(
        answer_b,
        image_root=source_b_root,
        validate_images=validate_images,
        source_name="round1_test_b",
    )
    combined_records, duplicate_count = deduplicate_fashionai_records(
        [*records_a, *records_b]
    )
    value_names = _load_value_names(label_map)
    schema = infer_fashionai_schema(combined_records, value_names=value_names)
    splits = stratified_split_records(
        combined_records,
        split_fractions=split_fractions,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_files = {}
    for split_name, split_records in splits.items():
        split_path = output_dir / f"{split_name}.csv"
        _write_records(split_path, split_records)
        split_files[split_name] = str(split_path)

    split_image_ids = {
        name: {record.split_key for record in split_records}
        for name, split_records in splits.items()
    }
    split_names = list(splits)
    overlaps = {
        f"{left}_{right}": len(split_image_ids[left] & split_image_ids[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    payload = {
        "dataset": "FashionAI Round1 labeled test A+B",
        "supervision": "human_answer_csv",
        "seed": seed,
        "split_strategy": "image_grouped_stratified_attribute_and_y_class",
        "split_fractions": split_fractions,
        "source_roots": {
            "round1_test_a": str(source_a_root),
            "round1_test_b": str(source_b_root),
        },
        "answer_files": {
            "round1_test_a": str(answer_a),
            "round1_test_b": str(answer_b),
        },
        "source_record_counts": {
            "round1_test_a": len(records_a),
            "round1_test_b": len(records_b),
        },
        "num_records_before_deduplication": len(records_a) + len(records_b),
        "num_duplicate_records": duplicate_count,
        "num_unique_records": len(combined_records),
        "split_files": split_files,
        "split_overlap_counts": overlaps,
        "splits": {
            name: _summarize_records(split_records)
            for name, split_records in splits.items()
        },
        "stratification_audit": _summarize_strata_balance(
            combined_records,
            splits,
            split_fractions,
        ),
        "schema": schema.to_dict(),
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload["summary_file"] = str(summary_path)
    return payload


def _write_records(
    path: Path,
    records: list[FashionAIAttributeRecord],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_path", "attribute_name", "label"))
        for record in records:
            writer.writerow(
                (
                    str(record.image_path.resolve()),
                    record.attribute_name,
                    record.label,
                )
            )


def _summarize_records(
    records: list[FashionAIAttributeRecord],
) -> dict[str, Any]:
    class_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        class_counts[record.attribute_name][record.target_index] += 1
    return {
        "num_records": len(records),
        "num_unique_images": len({record.split_key for record in records}),
        "source_counts": dict(
            sorted(Counter(record.source_name for record in records).items())
        ),
        "attribute_counts": dict(
            sorted(Counter(record.attribute_name for record in records).items())
        ),
        "strict_class_counts": {
            attribute_name: {
                str(class_index): count for class_index, count in sorted(counts.items())
            }
            for attribute_name, counts in sorted(class_counts.items())
        },
        "num_ambiguous_records": sum(
            bool(record.probable_indices) for record in records
        ),
    }


def _summarize_strata_balance(
    records: list[FashionAIAttributeRecord],
    splits: dict[str, list[FashionAIAttributeRecord]],
    split_fractions: dict[str, float],
) -> dict[str, Any]:
    """Summarize split balance for each `(attribute_name, strict y class)` stratum."""
    total_counts = Counter(_stratum_key(record) for record in records)
    split_counts = {
        split_name: Counter(_stratum_key(record) for record in split_records)
        for split_name, split_records in splits.items()
    }

    strata = {}
    max_fraction_error = 0.0
    for stratum in sorted(total_counts):
        total = total_counts[stratum]
        counts = {
            split_name: split_counts[split_name].get(stratum, 0)
            for split_name in splits
        }
        fractions = {
            split_name: counts[split_name] / total
            for split_name in splits
        }
        fraction_errors = {
            split_name: fractions[split_name] - split_fractions[split_name]
            for split_name in splits
        }
        max_fraction_error = max(
            max_fraction_error,
            *(abs(value) for value in fraction_errors.values()),
        )
        strata[stratum] = {
            "total": total,
            "counts": counts,
            "fractions": {
                split_name: round(value, 6)
                for split_name, value in fractions.items()
            },
            "fraction_errors": {
                split_name: round(value, 6)
                for split_name, value in fraction_errors.items()
            },
        }

    return {
        "stratification_key": "attribute_name + strict_y_class",
        "num_strata": len(strata),
        "max_absolute_fraction_error": round(max_fraction_error, 6),
        "strata": strata,
    }


def _stratum_key(record: FashionAIAttributeRecord) -> str:
    return f"{record.attribute_name}::{record.target_index}"


def _load_value_names(path: Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    payload = load_yaml(path)
    raw_attributes = payload.get("attributes", payload)
    value_names = {}
    for attribute_name, raw in raw_attributes.items():
        values = raw.get("values") if isinstance(raw, dict) else raw
        if not isinstance(values, list):
            raise ValueError(
                f"Label map for {attribute_name!r} must be a list or contain values."
            )
        value_names[str(attribute_name)] = [str(value) for value in values]
    return value_names
