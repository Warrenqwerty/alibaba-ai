from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fashion_mm.data_loaders import discover_fashionai_csvs
from fashion_mm.data_loaders import infer_fashionai_schema
from fashion_mm.data_loaders import read_fashionai_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect downloaded FashionAI CSVs before 3.1.3 training."
    )
    parser.add_argument(
        "--root",
        default="/root/autodl-tmp/datasets/FashionAI",
        help="Downloaded FashionAI root used for CSV discovery and image paths.",
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=None,
        help="Optional explicit annotation CSV files.",
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--validate-images", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    csv_paths = (
        [Path(path) for path in args.annotations]
        if args.annotations
        else discover_fashionai_csvs(root)
    )
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under FashionAI root: {root}")

    if not args.annotations:
        payload = {
            "fashionai_root": str(root),
            "mode": "csv_discovery",
            "num_csv_files": len(csv_paths),
            "csv_files": [str(path) for path in csv_paths],
            "next_step": "Re-run with --annotations using the training label CSV files.",
        }
    else:
        records = read_fashionai_annotations(
            csv_paths,
            image_root=args.image_root or root,
            validate_images=args.validate_images,
            skip_invalid=args.skip_invalid,
            max_records=args.max_records,
        )
        schema = infer_fashionai_schema(records)
        attribute_counts = Counter(record.attribute_name for record in records)
        payload = {
            "fashionai_root": str(root),
            "mode": "annotation_schema",
            "annotation_files": [str(path) for path in csv_paths],
            "num_records": len(records),
            "num_unique_images": len({str(record.image_path) for record in records}),
            "num_attribute_heads": len(schema.definitions),
            "num_total_values": sum(
                definition.num_classes for definition in schema.definitions
            ),
            "num_ambiguous_records": sum(
                bool(record.probable_indices) for record in records
            ),
            "attribute_counts": dict(sorted(attribute_counts.items())),
            "schema": schema.to_dict(),
        }

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
