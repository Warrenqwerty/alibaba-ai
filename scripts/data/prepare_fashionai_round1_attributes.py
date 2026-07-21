from __future__ import annotations

import argparse
import json
from pathlib import Path

from fashion_mm.data_loaders import prepare_fashionai_round1_splits


DEFAULT_ROOT = Path("/root/autodl-tmp/datasets/FashionAI")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/outputs/fashionai_round1_stratified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the labeled FashionAI Round1 test A/B files, remove their "
            "overlap, and create deterministic stratified train/val/test splits."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-a-root", type=Path, default=None)
    parser.add_argument("--source-b-root", type=Path, default=None)
    parser.add_argument("--answer-a", type=Path, default=None)
    parser.add_argument("--answer-b", type=Path, default=None)
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
        "--skip-image-validation",
        action="store_true",
        help="Skip image existence checks. This is not recommended for the final split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_a_root = args.source_a_root or (
        args.root / "round1_fashionAI_attributes_test_a"
    )
    source_b_root = args.source_b_root or (
        args.root / "round1_fashionAI_attributes_test_b"
    )
    answer_a = args.answer_a or (
        source_a_root / "Tests" / "round1_fashionAI_attributes_answer_a.csv"
    )
    answer_b = args.answer_b or (
        source_b_root / "Tests" / "round1_fashionAI_attributes_answer_b.csv"
    )
    payload = prepare_fashionai_round1_splits(
        source_a_root=source_a_root,
        source_b_root=source_b_root,
        answer_a=answer_a,
        answer_b=answer_b,
        output_dir=args.output_dir,
        split_fractions={
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        seed=args.seed,
        label_map=args.label_map,
        validate_images=not args.skip_image_validation,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
