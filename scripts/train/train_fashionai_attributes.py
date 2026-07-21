from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fashion_mm.data_loaders import build_fashionai_transform
from fashion_mm.data_loaders import collate_fashionai_attributes
from fashion_mm.data_loaders import FashionAIAttributeDataset
from fashion_mm.data_loaders import infer_fashionai_schema
from fashion_mm.data_loaders import read_fashionai_annotations
from fashion_mm.data_loaders import split_records_by_image
from fashion_mm.models.attributes import FashionAttributeClassifier
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

    validation_paths = (
        args.validation_annotations or dataset_config.get("validation_annotations")
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
    train_dataset = FashionAIAttributeDataset(
        train_records, build_fashionai_transform(image_size, train=True)
    )
    validation_dataset = (
        FashionAIAttributeDataset(
            validation_records, build_fashionai_transform(image_size, train=False)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0001)),
    )
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        resume_schema = checkpoint.get("schema")
        if resume_schema is not None and resume_schema != schema.to_dict():
            raise ValueError("Resume checkpoint schema does not match current CSV schema.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        LOGGER.info("Resumed 3.1.3 checkpoint: %s", args.resume)

    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    output_dir = Path(
        args.output_dir
        or Path(config["checkpoint_root"]) / "fashionai_attributes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    LOGGER.info(
        "FashionAI records: train=%s validation=%s heads=%s values=%s",
        len(train_records),
        len(validation_records),
        len(schema.definitions),
        sum(definition.num_classes for definition in schema.definitions),
    )

    for epoch in range(start_epoch, int(config["training"]["num_epochs"])):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
            log_interval=int(config["training"].get("log_interval", 100)),
        )
        validation_metrics = (
            run_epoch(model, validation_loader, device=device)
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


def run_epoch(
    model: FashionAttributeClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    label_smoothing: float = 0.0,
    log_interval: int = 0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_records = 0
    strict_correct = 0
    acceptable_correct = 0
    by_attribute: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "strict_correct": 0, "acceptable_correct": 0}
    )

    for step, batch in enumerate(loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["target_indices"].to(device, non_blocking=True)
        attribute_names = batch["attribute_names"]
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=scaler is not None and scaler.is_enabled(),
            ):
                features = model.encode(images)
                loss_sum = torch.zeros((), device=device)
                batch_predictions = torch.empty_like(targets)
                for attribute_name in sorted(set(attribute_names)):
                    indices = [
                        index
                        for index, name in enumerate(attribute_names)
                        if name == attribute_name
                    ]
                    index_tensor = torch.tensor(indices, device=device)
                    logits = model.classify(features.index_select(0, index_tensor), attribute_name)
                    group_targets = targets.index_select(0, index_tensor)
                    loss_sum = loss_sum + F.cross_entropy(
                        logits,
                        group_targets,
                        reduction="sum",
                        label_smoothing=label_smoothing,
                    )
                    batch_predictions.index_copy_(0, index_tensor, logits.argmax(dim=-1))
                loss = loss_sum / len(attribute_names)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        predictions = batch_predictions.detach().cpu().tolist()
        target_values = targets.detach().cpu().tolist()
        total_loss += float(loss.detach().cpu()) * len(attribute_names)
        total_records += len(attribute_names)
        for index, (attribute_name, prediction, target) in enumerate(
            zip(attribute_names, predictions, target_values, strict=True)
        ):
            strict_hit = prediction == target
            acceptable_hit = prediction in batch["acceptable_indices"][index]
            strict_correct += int(strict_hit)
            acceptable_correct += int(acceptable_hit)
            metrics = by_attribute[attribute_name]
            metrics["records"] += 1
            metrics["strict_correct"] += int(strict_hit)
            metrics["acceptable_correct"] += int(acceptable_hit)

        if training and log_interval and step % log_interval == 0:
            LOGGER.info(
                "batch=%s loss=%.4f strict_accuracy=%.4f",
                step,
                total_loss / max(total_records, 1),
                strict_correct / max(total_records, 1),
            )

    return {
        "num_records": total_records,
        "loss": total_loss / max(total_records, 1),
        "strict_accuracy": strict_correct / max(total_records, 1),
        "acceptable_accuracy": acceptable_correct / max(total_records, 1),
        "by_attribute": {
            name: {
                **counts,
                "strict_accuracy": counts["strict_correct"] / counts["records"],
                "acceptable_accuracy": counts["acceptable_correct"] / counts["records"],
            }
            for name, counts in sorted(by_attribute.items())
        },
    }


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
