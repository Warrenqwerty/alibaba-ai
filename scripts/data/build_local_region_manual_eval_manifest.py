from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from fashion_mm.models.local_region import parse_region_query


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small image-query JSONL manifest for manual 3.1.2 "
            "local-region bbox annotation."
        )
    )
    parser.add_argument("--image-dir", required=True, help="Directory of images to sample.")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_MANUAL_QUERIES,
        help="Natural-language queries to annotate for sampled images.",
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

    records = build_manifest_records(
        image_paths,
        list(args.queries),
        max_records=args.max_records,
    )
    output_path = Path(args.output)
    write_jsonl(records, output_path)
    summary = {
        "image_dir": str(Path(args.image_dir)),
        "output": str(output_path),
        "num_images": len(image_paths),
        "num_records": len(records),
        "queries": list(args.queries),
        "annotation_instruction": (
            "Fill target_bbox as [x1, y1, x2, y2] in image pixels and set "
            "label_status to labeled. Do not use landmarks while labeling."
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
