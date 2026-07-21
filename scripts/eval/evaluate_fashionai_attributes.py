from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

import torch
from torch.utils.data import DataLoader

from fashion_mm.data_loaders import build_fashionai_transform
from fashion_mm.data_loaders import collate_fashionai_attributes
from fashion_mm.data_loaders import FashionAIAttributeDataset
from fashion_mm.data_loaders import FashionAIAttributeSchema
from fashion_mm.data_loaders import read_fashionai_annotations
from fashion_mm.models.attributes import FashionAttributeClassifier
from fashion_mm.models.attributes import run_attribute_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a 3.1.3 checkpoint on a held-out FashionAI split."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the 3.1.3 evaluation. Use --device cpu only "
            "for an explicit smoke test."
        )

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    schema = FashionAIAttributeSchema.from_dict(checkpoint["schema"])
    model_config = checkpoint.get("model_config", {})
    model = FashionAttributeClassifier(
        schema,
        backbone_name=str(model_config.get("backbone", "mobilenet_v3_small")),
        pretrained=False,
        dropout=float(model_config.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    records = read_fashionai_annotations(
        args.annotations,
        image_root=args.image_root,
        validate_images=True,
        max_records=args.max_records,
        source_name="held_out_test",
    )
    _validate_records_against_schema(records, schema)
    image_size = int(model_config.get("image_size", 224))
    dataset = FashionAIAttributeDataset(
        records,
        build_fashionai_transform(image_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fashionai_attributes,
        pin_memory=device.type == "cuda",
    )
    metrics = run_attribute_epoch(model, loader, device=device)
    payload = {
        "annotations": str(Path(args.annotations)),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "split_role": "held_out_test",
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    output_path.write_text(serialized, encoding="utf-8")
    print(serialized)


def _validate_records_against_schema(records, schema) -> None:
    for record in records:
        definition = schema.definition(record.attribute_name)
        if record.num_classes != definition.num_classes:
            raise ValueError(
                f"Test record {record.image_path} has {record.num_classes} classes "
                f"for {record.attribute_name}, expected {definition.num_classes}."
            )


if __name__ == "__main__":
    main()
