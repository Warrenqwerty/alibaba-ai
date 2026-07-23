from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
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
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


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
    return prepare_fashionai_source_splits(
        sources=(
            ("round1_test_a", source_a_root, answer_a),
            ("round1_test_b", source_b_root, answer_b),
        ),
        dataset_name="FashionAI Round1 labeled test A+B",
        supervision="human_answer_csv",
        output_dir=output_dir,
        split_fractions=split_fractions,
        seed=seed,
        label_map=label_map,
        validate_images=validate_images,
        content_hash_images=False,
    )


def discover_fashionai_training_sources(
    root: str | Path,
) -> list[tuple[str, Path, Path]]:
    """Discover extracted FashionAI training roots with `Annotations/label.csv`."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"FashionAI root not found: {root_path}")

    annotation_files = sorted(
        path
        for path in root_path.rglob("label.csv")
        if path.parent.name.lower() == "annotations"
        and not any(part.startswith("._") for part in path.parts)
    )
    sources = []
    used_names: set[str] = set()
    for index, annotation_file in enumerate(annotation_files, start=1):
        source_root = annotation_file.parent.parent
        try:
            relative_root = source_root.relative_to(root_path).as_posix()
        except ValueError:
            relative_root = source_root.name
        normalized_name = relative_root.replace("/", "::").strip(":")
        source_name = normalized_name or f"training_source_{index:02d}"
        if source_name in used_names:
            source_name = f"{source_name}_{index:02d}"
        used_names.add(source_name)
        sources.append((source_name, source_root, annotation_file))
    return sources


def prepare_fashionai_source_splits(
    *,
    sources: Sequence[tuple[str, Path, Path]],
    dataset_name: str,
    supervision: str,
    output_dir: Path,
    split_fractions: dict[str, float],
    seed: int,
    label_map: Path | None,
    validate_images: bool,
    content_hash_images: bool,
) -> dict[str, Any]:
    """Merge labeled FashionAI sources and write content-safe stratified splits."""
    if not sources:
        raise ValueError("At least one FashionAI source is required.")
    source_names = [name for name, _, _ in sources]
    if len(source_names) != len(set(source_names)):
        raise ValueError("FashionAI source names must be unique.")

    records_by_source: dict[str, list[FashionAIAttributeRecord]] = {}
    all_records: list[FashionAIAttributeRecord] = []
    content_hash_cache: dict[Path, str] = {}
    for source_name, source_root, annotation_file in sources:
        LOGGER.info(
            "Reading FashionAI source %s from %s",
            source_name,
            annotation_file,
        )
        records = read_fashionai_annotations(
            annotation_file,
            image_root=source_root,
            validate_images=validate_images,
            source_name=source_name,
        )
        if content_hash_images:
            LOGGER.info(
                "Hashing %s referenced images for source %s",
                len(records),
                source_name,
            )
            records = _replace_image_ids_with_content_hashes(
                records,
                cache=content_hash_cache,
            )
        records_by_source[source_name] = records
        all_records.extend(records)

    combined_records, duplicate_count = deduplicate_fashionai_records(all_records)
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
        "dataset": dataset_name,
        "supervision": supervision,
        "seed": seed,
        "split_strategy": "image_grouped_stratified_attribute_and_y_class",
        "image_identity_strategy": (
            "sha256_file_content"
            if content_hash_images
            else "normalized_relative_image_path"
        ),
        "split_fractions": split_fractions,
        "source_roots": {
            name: str(source_root) for name, source_root, _ in sources
        },
        "answer_files": {
            name: str(annotation_file) for name, _, annotation_file in sources
        },
        "source_record_counts": {
            name: len(records_by_source[name]) for name in source_names
        },
        "num_records_before_deduplication": len(all_records),
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


def _replace_image_ids_with_content_hashes(
    records: Sequence[FashionAIAttributeRecord],
    *,
    cache: dict[Path, str],
) -> list[FashionAIAttributeRecord]:
    hashed_records = []
    for index, record in enumerate(records, start=1):
        image_path = record.image_path.resolve()
        digest = cache.get(image_path)
        if digest is None:
            hasher = hashlib.sha256()
            with image_path.open("rb") as image_file:
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            cache[image_path] = digest
        hashed_records.append(replace(record, image_id=f"sha256:{digest}"))
        if index % 10_000 == 0:
            LOGGER.info("Hashed %s/%s FashionAI records", index, len(records))
    return hashed_records


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
