from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fashion_mm.models.attributes.model import FashionAttributeClassifier
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


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
