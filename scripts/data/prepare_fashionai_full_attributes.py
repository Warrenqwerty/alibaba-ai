from __future__ import annotations

import argparse
import json
from pathlib import Path

from fashion_mm.data_loaders import discover_fashionai_training_sources
from fashion_mm.data_loaders import prepare_fashionai_source_splits


DEFAULT_ROOT = Path("/root/autodl-tmp/datasets/FashionAI")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/outputs/fashionai_full_stratified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover extracted FashionAI training releases, deduplicate images "
            "by content, and create deterministic stratified splits."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("configs/dataset/fashionai_round1_label_map.yaml"),
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print discovered training annotation files without preparing splits.",
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip image existence checks. Not recommended for final preparation.",
    )
    parser.add_argument(
        "--skip-content-hash",
        action="store_true",
        help="Use relative paths instead of content hashes. Not recommended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = discover_fashionai_training_sources(args.root)
    archives = sorted(
        path
        for path in args.root.rglob("*.zip")
        if "attribute" in path.name.lower()
        and not path.name.startswith("._")
    )
    source_payload = [
        {
            "name": name,
            "root": str(source_root),
            "annotations": str(annotation_file),
        }
        for name, source_root, annotation_file in sources
    ]
    if args.list_sources:
        print(
            json.dumps(
                {
                    "root": str(args.root),
                    "sources": source_payload,
                    "archives": [str(path) for path in archives],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not sources:
        raise FileNotFoundError(
            f"No extracted Annotations/label.csv files found under {args.root}. "
            "Extract fashionAI_attributes_train1.zip and train2.zip first."
        )

    payload = prepare_fashionai_source_splits(
        sources=sources,
        dataset_name="FashionAI full extracted training releases",
        supervision="training_label_csv",
        output_dir=args.output_dir,
        split_fractions={
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        seed=args.seed,
        label_map=args.label_map,
        validate_images=not args.skip_image_validation,
        content_hash_images=not args.skip_content_hash,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
