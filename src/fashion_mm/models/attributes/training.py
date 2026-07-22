from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fashion_mm.models.attributes.model import FashionAttributeClassifier
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


def build_attribute_optimizer(
    model: FashionAttributeClassifier,
    *,
    learning_rate: float,
    weight_decay: float,
    backbone_learning_rate: float | None = None,
) -> torch.optim.AdamW:
    """Build AdamW with an optional lower rate for the pretrained backbone."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if backbone_learning_rate is not None and backbone_learning_rate <= 0.0:
        raise ValueError("backbone_learning_rate must be positive when provided.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative.")

    effective_backbone_rate = (
        backbone_learning_rate
        if backbone_learning_rate is not None
        else learning_rate
    )
    if effective_backbone_rate == learning_rate:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(
        [
            {
                "name": "backbone",
                "params": model.backbone.parameters(),
                "lr": effective_backbone_rate,
            },
            {
                "name": "heads",
                "params": model.heads.parameters(),
                "lr": learning_rate,
            },
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_attribute_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_config: Mapping[str, Any] | None,
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build an optional epoch-level learning-rate scheduler."""
    if scheduler_config is None:
        return None
    if not isinstance(scheduler_config, Mapping):
        raise ValueError("scheduler configuration must be a mapping.")
    if not scheduler_config:
        return None
    name = str(scheduler_config.get("name", "none")).strip().lower()
    if name in {"", "none"}:
        return None
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive when using a scheduler.")
    if name == "cosine":
        minimum_rate = float(scheduler_config.get("min_learning_rate", 0.0))
        if minimum_rate < 0.0:
            raise ValueError("min_learning_rate cannot be negative.")
        initial_rates = [float(group["lr"]) for group in optimizer.param_groups]
        if minimum_rate >= min(initial_rates):
            raise ValueError(
                "min_learning_rate must be below every optimizer learning rate."
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=minimum_rate,
        )
    raise ValueError(f"Unsupported attribute scheduler: {name!r}.")


def run_attribute_epoch(
    model: FashionAttributeClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    label_smoothing: float = 0.0,
    log_interval: int = 0,
) -> dict[str, Any]:
    """Run one heterogeneous-head train or evaluation epoch."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_records = 0
    strict_correct = 0
    acceptable_correct = 0
    model_time_seconds = 0.0
    by_attribute: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "strict_correct": 0, "acceptable_correct": 0}
    )

    for step, batch in enumerate(loader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["target_indices"].to(device, non_blocking=True)
        attribute_names = batch["attribute_names"]
        if training:
            optimizer.zero_grad(set_to_none=True)
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = perf_counter()

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
                    logits = model.classify(
                        features.index_select(0, index_tensor), attribute_name
                    )
                    group_targets = targets.index_select(0, index_tensor)
                    loss_sum = loss_sum + F.cross_entropy(
                        logits,
                        group_targets,
                        reduction="sum",
                        label_smoothing=label_smoothing,
                    )
                    batch_predictions.index_copy_(
                        0, index_tensor, logits.argmax(dim=-1)
                    )
                loss = loss_sum / len(attribute_names)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        if not training:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_time_seconds += perf_counter() - started_at

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

    payload = {
        "num_records": total_records,
        "loss": total_loss / max(total_records, 1),
        "strict_accuracy": strict_correct / max(total_records, 1),
        "acceptable_accuracy": acceptable_correct / max(total_records, 1),
        "by_attribute": {
            name: {
                **counts,
                "strict_accuracy": counts["strict_correct"] / counts["records"],
                "acceptable_accuracy": (
                    counts["acceptable_correct"] / counts["records"]
                ),
            }
            for name, counts in sorted(by_attribute.items())
        },
    }
    if not training:
        payload["model_inference_time_ms"] = model_time_seconds * 1000.0
        payload["avg_model_inference_time_ms"] = (
            model_time_seconds * 1000.0 / max(total_records, 1)
        )
    return payload
