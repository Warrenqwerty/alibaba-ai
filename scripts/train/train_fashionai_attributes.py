from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import torch
from torch.utils.data import DataLoader

from fashion_mm.data_loaders import build_fashionai_transform
from fashion_mm.data_loaders import collate_fashionai_attributes
from fashion_mm.data_loaders import FashionAIAttributeDataset
from fashion_mm.data_loaders import infer_fashionai_schema
from fashion_mm.data_loaders import read_fashionai_annotations
from fashion_mm.data_loaders import split_records_by_image
from fashion_mm.models.attributes import build_attribute_optimizer
from fashion_mm.models.attributes import FashionAttributeClassifier
from fashion_mm.models.attributes import run_attribute_epoch
from fashion_mm.utils.config import load_config
from fashion_mm.utils.config import load_yaml
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 3.1.3 multi-head FashionAI attribute classifier."
    )
    parser.add_argument("--annotations", nargs="+", default=None)
    parser.add_argument("--validation-annotations", nargs="+", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--paths-config", default="configs/paths.autodl.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset/fashionai.yaml")
    parser.add_argument(
        "--model-config", default="configs/model/fashionai_attributes.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.paths_config, args.dataset_config, args.model_config)
    seed = int(config["training"].get("seed", 42))
    _seed_everything(seed)
    device = torch.device(args.device or config["inference"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for 3.1.3 training. Use --device cpu only for an "
            "explicit smoke test."
        )

    dataset_config = config["fashionai_attributes"]
    annotation_paths = args.annotations or dataset_config.get("train_annotations")
    if not annotation_paths:
        raise ValueError(
            "No FashionAI training annotations configured. Run "
            "scripts/data/inspect_fashionai_attributes.py first, then pass "
            "--annotations CSV [CSV ...]."
        )
    image_root = Path(args.image_root or config["fashionai"]["root"])
    records = read_fashionai_annotations(
        annotation_paths,
        image_root=image_root,
        validate_images=bool(dataset_config.get("validate_images", True)),
        skip_invalid=bool(dataset_config.get("skip_invalid_annotations", False)),
        max_records=args.max_records,
    )
    value_names = _load_value_names(args.label_map or dataset_config.get("label_map"))
    schema = infer_fashionai_schema(records, value_names=value_names)

    validation_paths = args.validation_annotations or dataset_config.get(
        "validation_annotations"
    )
    if validation_paths:
        train_records = records
        validation_records = read_fashionai_annotations(
            validation_paths,
            image_root=image_root,
            validate_images=bool(dataset_config.get("validate_images", True)),
            skip_invalid=bool(dataset_config.get("skip_invalid_annotations", False)),
        )
        validation_schema = infer_fashionai_schema(
            [*train_records, *validation_records], value_names=value_names
        )
        if validation_schema != schema:
            schema = validation_schema
    else:
        train_records, validation_records = split_records_by_image(
            records,
            validation_fraction=float(dataset_config.get("validation_fraction", 0.1)),
            seed=seed,
        )

    image_size = int(config["model"].get("image_size", 224))
    input_mode = str(config["model"].get("input_mode", "crop"))
    train_dataset = FashionAIAttributeDataset(
        train_records,
        build_fashionai_transform(
            image_size,
            train=True,
            input_mode=input_mode,
        ),
    )
    validation_dataset = (
        FashionAIAttributeDataset(
            validation_records,
            build_fashionai_transform(
                image_size,
                train=False,
                input_mode=input_mode,
            ),
        )
        if validation_records
        else None
    )
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "collate_fn": collate_fashionai_attributes,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
        if validation_dataset is not None
        else None
    )

    model = FashionAttributeClassifier(
        schema,
        backbone_name=str(config["model"].get("backbone", "mobilenet_v3_small")),
        pretrained=bool(config["model"].get("pretrained", True)),
        dropout=float(config["model"].get("dropout", 0.2)),
    ).to(device)
    learning_rate = float(config["training"]["learning_rate"])
    backbone_learning_rate = float(
        config["training"].get("backbone_learning_rate", learning_rate)
    )
    optimizer = build_attribute_optimizer(
        model,
        learning_rate=learning_rate,
        backbone_learning_rate=backbone_learning_rate,
        weight_decay=float(config["training"].get("weight_decay", 0.0001)),
    )
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        resume_schema = checkpoint.get("schema")
        if resume_schema is not None and resume_schema != schema.to_dict():
            raise ValueError(
                "Resume checkpoint schema does not match current CSV schema."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        LOGGER.info("Resumed 3.1.3 checkpoint: %s", args.resume)

    use_amp = (
        bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    output_dir = Path(
        args.output_dir or Path(config["checkpoint_root"]) / "fashionai_attributes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    LOGGER.info(
        "FashionAI records: train=%s validation=%s heads=%s values=%s "
        "input_mode=%s backbone_lr=%g head_lr=%g",
        len(train_records),
        len(validation_records),
        len(schema.definitions),
        sum(definition.num_classes for definition in schema.definitions),
        input_mode,
        backbone_learning_rate,
        learning_rate,
    )

    for epoch in range(start_epoch, int(config["training"]["num_epochs"])):
        train_metrics = run_attribute_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
            log_interval=int(config["training"].get("log_interval", 100)),
        )
        validation_metrics = (
            run_attribute_epoch(model, validation_loader, device=device)
            if validation_loader is not None
            else train_metrics
        )
        LOGGER.info(
            "epoch=%s train_loss=%.4f train_acc=%.4f val_acc=%.4f val_acceptable=%.4f",
            epoch + 1,
            train_metrics["loss"],
            train_metrics["strict_accuracy"],
            validation_metrics["strict_accuracy"],
            validation_metrics["acceptable_accuracy"],
        )

        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "schema": schema.to_dict(),
            "model_config": {
                "backbone": model.backbone_name,
                "image_size": image_size,
                "input_mode": input_mode,
                "dropout": float(config["model"].get("dropout", 0.2)),
                "top_k": int(config["inference"].get("top_k", 3)),
                "confidence_threshold": float(
                    config["inference"].get("confidence_threshold", 0.0)
                ),
                "mask_padding_fraction": float(
                    config["inference"].get("mask_padding_fraction", 0.08)
                ),
            },
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "config": config,
        }
        epoch_path = output_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(checkpoint_payload, epoch_path)
        if validation_metrics["strict_accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["strict_accuracy"]
            torch.save(checkpoint_payload, output_dir / "best.pt")
        LOGGER.info("Saved 3.1.3 checkpoint: %s", epoch_path)


def _load_value_names(path: str | Path | None) -> dict[str, list[str]] | None:
    if not path:
        return None
    payload = load_yaml(path)
    raw_attributes = payload.get("attributes", payload)
    value_names: dict[str, list[str]] = {}
    for attribute_name, raw in raw_attributes.items():
        values = raw.get("values") if isinstance(raw, dict) else raw
        if not isinstance(values, list):
            raise ValueError(
                f"Label map for {attribute_name!r} must be a list or contain values."
            )
        value_names[str(attribute_name)] = [str(value) for value in values]
    return value_names


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
