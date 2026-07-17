from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fashion_mm.models.local_region import propose_region_from_landmarks


QUERY_TEMPLATES = {
    "neckline": ("这件衣服的领口", "衣领位置", "领口的设计"),
    "hem": ("衣服下方的下摆", "下摆位置", "衣摆的设计"),
    "shoulder": ("这件衣服的肩部", "肩线位置", "肩部设计"),
    "left_cuff": ("左边的袖口", "衣服左侧的袖口", "左侧袖口位置"),
    "right_cuff": ("右边的袖口", "衣服右侧的袖口", "右侧袖口位置"),
    "waist": ("这条下装的腰部", "腰线位置", "腰头位置"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weak DeepFashion2 query-region records for 3.1.2."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--anno-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["neckline", "hem", "shoulder"],
        choices=sorted(QUERY_TEMPLATES),
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--landmark-only",
        action="store_true",
        help="Skip records that would fall back to rule-generated targets.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Optional directory for pseudo-label bbox spot-check images.",
    )
    parser.add_argument("--vis-count", type=int, default=30)
    return parser.parse_args()


def collect_annotations(
    anno_dir: Path,
    max_images: int | None,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> list[Path]:
    """Collect visible DeepFashion2 annotations in deterministic order."""
    annotation_paths = [
        path
        for path in sorted(anno_dir.glob("*.json"))
        if path.is_file() and not path.name.startswith(".")
    ]
    if shuffle:
        random.Random(seed).shuffle(annotation_paths)
    return annotation_paths[:max_images] if max_images is not None else annotation_paths


def image_path_for_annotation(
    image_dir: Path,
    annotation_path: Path,
    annotation: dict[str, Any],
) -> Path:
    """Resolve a DeepFashion2 image path from annotation metadata."""
    candidates = []
    source = annotation.get("source")
    if isinstance(source, str):
        candidates.append(image_dir / source)
    candidates.append(image_dir / f"{annotation_path.stem}.jpg")
    candidates.append(image_dir / f"{annotation_path.stem}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for annotation: {annotation_path}")


def polygon_to_mask(
    segmentation: list[list[float]],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Rasterize DeepFashion2 polygon segmentation to a boolean mask."""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        points = [
            (float(polygon[index]), float(polygon[index + 1]))
            for index in range(0, len(polygon), 2)
        ]
        draw.polygon(points, outline=1, fill=1)
    return np.asarray(mask, dtype=bool)


def iter_items(annotation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return DeepFashion2 item entries in deterministic order."""
    return [
        (key, annotation[key])
        for key in sorted(annotation)
        if key.startswith("item") and isinstance(annotation[key], dict)
    ]


def build_records_for_annotation(
    image_path: Path,
    annotation_path: Path,
    annotation: dict[str, Any],
    regions: list[str],
    *,
    landmark_only: bool = False,
) -> list[dict[str, Any]]:
    """Build weak query-region records for one DeepFashion2 annotation."""
    with Image.open(image_path) as image:
        image_size = image.size
    records: list[dict[str, Any]] = []
    items = iter_items(annotation)
    for item_key, item in items:
        garment_mask = polygon_to_mask(item.get("segmentation", []), image_size)
        if garment_mask.sum() == 0 or "bounding_box" not in item:
            continue

        for region in regions:
            proposal = propose_region_from_landmarks(
                garment_mask,
                item["bounding_box"],
                item.get("landmarks", []),
                region,
                category_id=item.get("category_id"),
            )
            if proposal.status != "ok" or proposal.box is None:
                continue
            if landmark_only and proposal.source != "landmark_pseudo_label":
                continue

            for query in QUERY_TEMPLATES[region]:
                records.append(
                    {
                        "image": str(image_path),
                        "annotation": str(annotation_path),
                        "item_key": item_key,
                        "category_id": item.get("category_id"),
                        "category_name": item.get("category_name"),
                        "num_items_in_image": len(items),
                        "query": query,
                        "region": region,
                        "garment_box": [float(value) for value in item["bounding_box"]],
                        "region_box": [float(value) for value in proposal.box],
                        "source": proposal.source,
                        "confidence": proposal.confidence,
                    }
                )
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exported weak query-region records."""
    return {
        "num_records": len(records),
        "region_counts": dict(Counter(record["region"] for record in records)),
        "source_counts": dict(Counter(record["source"] for record in records)),
    }


def export_record_visualizations(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    max_records: int,
) -> int:
    """Draw one pseudo-label box per image/item/region for visual auditing."""
    if max_records < 0:
        raise ValueError("max_records cannot be negative")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str]] = set()
    exported = 0
    for record in records:
        key = (
            str(record["image"]),
            str(record["item_key"]),
            str(record["region"]),
        )
        if key in seen:
            continue
        seen.add(key)
        if exported >= max_records:
            break
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(record["garment_box"], outline="#2563eb", width=3)
        draw.rectangle(record["region_box"], outline="#16a34a", width=4)
        label = (
            f"{record['region']} | item={record['item_key']} | "
            f"category={record.get('category_id')}"
        )
        draw.text((6, 6), label, fill="#16a34a", stroke_width=2, stroke_fill="white")
        stem = Path(record["image"]).stem
        output_path = destination / (
            f"{exported:03d}_{stem}_{record['item_key']}_{record['region']}.jpg"
        )
        image.save(output_path, quality=92)
        exported += 1
    return exported


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    anno_dir = Path(args.anno_dir)
    output_path = Path(args.output)
    annotation_paths = collect_annotations(
        anno_dir,
        args.max_images,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not annotation_paths:
        raise ValueError(f"No annotations found in {anno_dir}")

    records: list[dict[str, Any]] = []
    for annotation_path in annotation_paths:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = image_path_for_annotation(image_dir, annotation_path, annotation)
        records.extend(
            build_records_for_annotation(
                image_path,
                annotation_path,
                annotation,
                args.regions,
                landmark_only=args.landmark_only,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    num_visualizations = 0
    if args.vis_dir:
        num_visualizations = export_record_visualizations(
            records,
            args.vis_dir,
            max_records=args.vis_count,
        )

    summary = {
        "num_annotations": len(annotation_paths),
        "output": str(output_path),
        "landmark_only": args.landmark_only,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "num_unique_images": len({record["image"] for record in records}),
        "num_visualizations": num_visualizations,
        "visualization_dir": str(Path(args.vis_dir)) if args.vis_dir else None,
        **summarize_records(records),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
