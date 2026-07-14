from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from fashion_mm.models.local_region import parse_region_query


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEEPFASHION2_CATEGORY_NAMES = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outerwear",
    4: "long sleeve outerwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}
DEFAULT_MANUAL_QUERIES = [
    "这件衣服的领口",
    "衣服下方的下摆",
    "这件衣服的肩部",
    "左边的袖口",
    "右边的袖口",
    "右侧的口袋",
    "衣服上的拉链",
    "这件衣服上的碎花图案",
]
CLASS_AWARE_QUERY_TEMPLATES = {
    1: ("这件上衣的领口", "这件上衣的肩部", "这件上衣左侧的袖口", "这件上衣右侧的袖口", "这件上衣的下摆", "这件上衣上的图案"),
    2: ("这件上衣的领口", "这件上衣的肩部", "这件上衣左侧的袖口", "这件上衣右侧的袖口", "这件上衣的下摆", "这件上衣上的图案"),
    3: ("这件外套的领口", "这件外套的肩部", "这件外套左侧的袖口", "这件外套右侧的袖口", "这件外套的下摆", "这件外套上的拉链", "这件外套右侧的口袋", "这件外套上的图案"),
    4: ("这件外套的领口", "这件外套的肩部", "这件外套左侧的袖口", "这件外套右侧的袖口", "这件外套的下摆", "这件外套上的拉链", "这件外套右侧的口袋", "这件外套上的图案"),
    5: ("这件上衣的领口", "这件上衣的肩部", "这件上衣的下摆", "这件上衣上的图案"),
    6: ("这件上衣的领口", "这件上衣的肩部", "这件上衣的下摆", "这件上衣上的图案"),
    7: ("这条裤子的腰部", "这条裤子的裤脚", "这条裤子右侧的口袋", "这条裤子上的拉链", "这条裤子上的图案"),
    8: ("这条裤子的腰部", "这条裤子的裤脚", "这条裤子右侧的口袋", "这条裤子上的拉链", "这条裤子上的图案"),
    9: ("这条裙子的腰部", "这条裙子的裙摆", "这条裙子上的图案"),
    10: ("这件连衣裙的领口", "这件连衣裙的肩部", "这件连衣裙左侧的袖口", "这件连衣裙右侧的袖口", "这件连衣裙的腰部", "这件连衣裙的裙摆", "这件连衣裙上的图案"),
    11: ("这件连衣裙的领口", "这件连衣裙的肩部", "这件连衣裙左侧的袖口", "这件连衣裙右侧的袖口", "这件连衣裙的腰部", "这件连衣裙的裙摆", "这件连衣裙上的图案"),
    12: ("这件连衣裙的领口", "这件连衣裙的肩部", "这件连衣裙的腰部", "这件连衣裙的裙摆", "这件连衣裙上的图案"),
    13: ("这件连衣裙的领口", "这件连衣裙的肩部", "这件连衣裙的腰部", "这件连衣裙的裙摆", "这件连衣裙上的图案"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small image-query JSONL manifest for manual 3.1.2 "
            "local-region bbox annotation."
        )
    )
    parser.add_argument("--image-dir", required=True, help="Directory of images to sample.")
    parser.add_argument(
        "--anno-dir",
        default=None,
        help=(
            "Optional DeepFashion2 annotation directory. When provided and "
            "--queries is omitted, class-aware query templates are used."
        ),
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help=(
            "Natural-language queries to annotate for sampled images. If omitted "
            "with --anno-dir, queries are chosen from the garment category."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum number of images to include before query expansion.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=300,
        help="Maximum number of image-query annotation records to write.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order before sampling.",
    )
    parser.add_argument(
        "--target-regions",
        nargs="+",
        default=None,
        help=(
            "Optional target_region filter after query generation, e.g. "
            "pattern zipper pocket."
        ),
    )
    parser.add_argument(
        "--exclude-existing",
        nargs="*",
        default=None,
        help=(
            "Optional existing manual JSONL files. Records with the same "
            "image/query_text/target_region key are skipped."
        ),
    )
    parser.add_argument(
        "--balance-target-regions",
        action="store_true",
        help=(
            "When --target-regions is set, select records in round-robin region "
            "order instead of taking the first max-records records."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        default="outputs/local_region_manual_eval_manifest.jsonl",
        help="Output JSONL path. Fill target_bbox manually after generation.",
    )
    return parser.parse_args()


def collect_images(
    image_dir: Path,
    max_images: int | None,
    *,
    shuffle: bool = False,
    seed: int = 2026,
) -> list[Path]:
    """Collect visible image files for manual annotation."""
    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(image_paths)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    return image_paths


def build_manifest_records(
    image_paths: list[Path],
    queries: list[str],
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Create JSONL-ready manual annotation records with empty target boxes."""
    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        width, height = Image.open(image_path).size
        for query in queries:
            parsed = parse_region_query(query)
            records.append(
                {
                    "id": f"{image_path.stem}__{len(records):06d}",
                    "image": str(image_path),
                    "query_text": query,
                    "target_region": parsed.region,
                    "image_width": width,
                    "image_height": height,
                    "target_bbox": None,
                    "bbox_format": "xyxy",
                    "label_status": "unlabeled",
                    "notes": "",
                }
            )
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def build_class_aware_manifest_records(
    image_paths: list[Path],
    anno_dir: str | Path,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Create manual records using category-compatible query templates."""
    records: list[dict[str, Any]] = []
    annotation_dir = Path(anno_dir)
    for image_path in image_paths:
        annotation_path = annotation_dir / f"{image_path.stem}.json"
        if not annotation_path.exists():
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        width, height = Image.open(image_path).size
        for item_key, item in iter_annotation_items(annotation):
            category_id = int(item.get("category_id", 0))
            for query in queries_for_category(category_id):
                parsed = parse_region_query(query)
                records.append(
                    {
                        "id": f"{image_path.stem}_{item_key}__{len(records):06d}",
                        "image": str(image_path),
                        "annotation": str(annotation_path),
                        "source_item_key": item_key,
                        "category_id": category_id,
                        "category_name": DEEPFASHION2_CATEGORY_NAMES.get(
                            category_id,
                            "unknown",
                        ),
                        "query_text": query,
                        "target_region": parsed.region,
                        "image_width": width,
                        "image_height": height,
                        "target_bbox": None,
                        "bbox_format": "xyxy",
                        "label_status": "unlabeled",
                        "notes": "",
                    }
                )
                if max_records is not None and len(records) >= max_records:
                    return records
    return records


def filter_records_by_target_regions(
    records: list[dict[str, Any]],
    target_regions: set[str] | None,
) -> list[dict[str, Any]]:
    """Keep only records whose parsed target_region is in target_regions."""
    if not target_regions:
        return records
    return [
        record
        for record in records
        if str(record.get("target_region") or "") in target_regions
    ]


def load_existing_record_keys(paths: list[str] | None) -> set[tuple[str, str, str]]:
    """Load existing manual annotation keys to avoid duplicate labeling."""
    keys: set[tuple[str, str, str]] = set()
    for path_value in paths or []:
        path = Path(path_value)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                keys.add(manual_record_key(record))
    return keys


def filter_existing_records(
    records: list[dict[str, Any]],
    existing_keys: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Remove records already present in existing manual annotation JSONL."""
    if not existing_keys:
        return records
    return [
        record
        for record in records
        if manual_record_key(record) not in existing_keys
    ]


def manual_record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """Stable key for manual annotation deduplication."""
    return (
        str(record.get("image") or ""),
        str(record.get("query_text") or ""),
        str(record.get("target_region") or ""),
    )


def limit_records(
    records: list[dict[str, Any]],
    max_records: int | None,
    *,
    balance_target_regions: bool = False,
    region_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Limit records, optionally balancing by target_region."""
    if max_records is None or len(records) <= max_records:
        return records
    if not balance_target_regions:
        return records[:max_records]
    return balanced_region_records(
        records,
        max_records=max_records,
        region_order=region_order,
    )


def balanced_region_records(
    records: list[dict[str, Any]],
    *,
    max_records: int,
    region_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Round-robin records across target regions."""
    records_by_region: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        region = str(record.get("target_region") or "unknown")
        records_by_region.setdefault(region, []).append(record)
    ordered_regions = [
        region
        for region in (region_order or sorted(records_by_region))
        if region in records_by_region
    ]
    if not ordered_regions:
        return []

    selected: list[dict[str, Any]] = []
    offsets = {region: 0 for region in ordered_regions}
    while len(selected) < max_records:
        made_progress = False
        for region in ordered_regions:
            offset = offsets[region]
            region_records = records_by_region[region]
            if offset >= len(region_records):
                continue
            selected.append(region_records[offset])
            offsets[region] += 1
            made_progress = True
            if len(selected) >= max_records:
                break
        if not made_progress:
            break
    return selected


def target_region_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count generated records by target_region."""
    counts: dict[str, int] = {}
    for record in records:
        region = str(record.get("target_region") or "unknown")
        counts[region] = counts.get(region, 0) + 1
    return dict(sorted(counts.items()))


def iter_annotation_items(annotation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return DeepFashion2 item entries in deterministic order."""
    return [
        (key, annotation[key])
        for key in sorted(annotation)
        if key.startswith("item") and isinstance(annotation[key], dict)
    ]


def queries_for_category(category_id: int) -> tuple[str, ...]:
    """Return garment-category-compatible manual eval queries."""
    return CLASS_AWARE_QUERY_TEMPLATES.get(category_id, tuple(DEFAULT_MANUAL_QUERIES))


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    image_paths = collect_images(
        Path(args.image_dir),
        args.max_images,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    if args.anno_dir is not None and args.queries is None:
        records = build_class_aware_manifest_records(
            image_paths,
            args.anno_dir,
            max_records=None,
        )
        query_mode = "class_aware"
        queries = "category-dependent"
    else:
        queries = list(args.queries or DEFAULT_MANUAL_QUERIES)
        records = build_manifest_records(
            image_paths,
            queries,
            max_records=None,
        )
        query_mode = "fixed"
    target_regions = set(args.target_regions or [])
    records = filter_records_by_target_regions(records, target_regions or None)
    existing_keys = load_existing_record_keys(args.exclude_existing)
    records = filter_existing_records(records, existing_keys)
    records = limit_records(
        records,
        args.max_records,
        balance_target_regions=args.balance_target_regions,
        region_order=args.target_regions,
    )
    if not records:
        raise ValueError("No manual annotation records generated")
    output_path = Path(args.output)
    write_jsonl(records, output_path)
    summary = {
        "image_dir": str(Path(args.image_dir)),
        "anno_dir": str(Path(args.anno_dir)) if args.anno_dir else None,
        "output": str(output_path),
        "num_images": len(image_paths),
        "num_records": len(records),
        "query_mode": query_mode,
        "queries": queries,
        "target_regions": args.target_regions,
        "target_region_counts": target_region_counts(records),
        "num_excluded_existing_keys": len(existing_keys),
        "annotation_instruction": (
            "Fill target_bbox as [x1, y1, x2, y2] in image pixels and set "
            "label_status to labeled. Do not use landmarks while labeling."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
