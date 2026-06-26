from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fashion_mm.data_loaders import LocalRegionCandidateRecord
from fashion_mm.data_loaders import iter_local_region_candidate_records
from fashion_mm.models.local_region import build_candidate_record_feature
from fashion_mm.models.local_region import CandidateListwiseScorer
from fashion_mm.models.local_region import parse_region_query
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a listwise 3.1.2 candidate ranker from weak IoU labels."
    )
    parser.add_argument("--candidates", required=True, help="Candidate JSONL path.")
    parser.add_argument(
        "--output",
        default="outputs/local_region_candidate_ranker.pt",
        help="Checkpoint path.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-buckets", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--loss",
        choices=("hard", "soft"),
        default="soft",
        help="Use hard best-IoU class targets or soft IoU distributions.",
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=0.08,
        help="Temperature for soft IoU target distributions when --loss=soft.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-groups", type=int, default=50000)
    parser.add_argument("--val-groups", type=int, default=2000)
    parser.add_argument(
        "--val-offset",
        type=int,
        default=0,
        help="Number of candidate groups to skip before reading validation groups.",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--metrics-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CandidateListwiseScorer(
        num_buckets=args.num_buckets,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    validation_groups = list(
        iter_candidate_groups(
            args.candidates,
            max_groups=args.val_groups,
            skip_groups=args.val_offset,
        )
    )
    LOGGER.info("Loaded validation groups: %s", len(validation_groups))

    for epoch in range(args.num_epochs):
        model.train()
        batch_features: list[torch.Tensor] = []
        batch_targets: list[torch.Tensor | int] = []
        total_loss = 0.0
        total_batches = 0
        train_stream_limit = _train_stream_limit(
            args.max_groups,
            args.val_groups,
            args.val_offset,
        )
        for group_index, group in enumerate(
            iter_candidate_groups(args.candidates, max_groups=train_stream_limit)
        ):
            if args.val_offset <= group_index < args.val_offset + args.val_groups:
                continue
            features, target = build_group_training_example(
                group,
                args.num_buckets,
                loss_mode=args.loss,
                softmax_temperature=args.softmax_temperature,
            )
            batch_features.append(features)
            batch_targets.append(target)
            if len(batch_features) >= args.batch_size:
                loss = _train_batch(
                    model,
                    optimizer,
                    batch_features,
                    batch_targets,
                    device,
                    loss_mode=args.loss,
                )
                total_loss += loss
                total_batches += 1
                batch_features.clear()
                batch_targets.clear()
                if total_batches % args.log_interval == 0:
                    LOGGER.info(
                        "epoch=%s batch=%s loss=%.4f",
                        epoch + 1,
                        total_batches,
                        total_loss / total_batches,
                    )

        if batch_features:
            loss = _train_batch(
                model,
                optimizer,
                batch_features,
                batch_targets,
                device,
                loss_mode=args.loss,
            )
            total_loss += loss
            total_batches += 1

        metrics = evaluate_ranker(
            model,
            validation_groups,
            args.num_buckets,
            device,
        )
        LOGGER.info(
            "epoch=%s loss=%.4f val_top1_iou=%.4f",
            epoch + 1,
            total_loss / max(total_batches, 1),
            metrics["avg_top1_iou"],
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_buckets": args.num_buckets,
        "hidden_dim": args.hidden_dim,
        "loss": args.loss,
        "softmax_temperature": args.softmax_temperature,
        "validation_metrics": evaluate_ranker(
            model,
            validation_groups,
            args.num_buckets,
            device,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    LOGGER.info("Saved candidate local-region ranker checkpoint: %s", output_path)
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(checkpoint["validation_metrics"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def iter_candidate_groups(
    jsonl_path: str | Path,
    *,
    max_groups: int | None = None,
    skip_groups: int = 0,
) -> Iterator[list[LocalRegionCandidateRecord]]:
    """Stream adjacent candidate rows grouped by one query target."""
    current_key: tuple[Any, ...] | None = None
    current_group: list[LocalRegionCandidateRecord] = []
    seen_groups = 0
    yielded_groups = 0

    for record in iter_local_region_candidate_records(jsonl_path):
        key = _group_key(record)
        if current_key is None:
            current_key = key
        if key != current_key:
            if seen_groups >= skip_groups:
                yield current_group
                yielded_groups += 1
                if max_groups is not None and yielded_groups >= max_groups:
                    return
            seen_groups += 1
            current_key = key
            current_group = []
        current_group.append(record)

    if current_group and seen_groups >= skip_groups:
        yield current_group


def build_group_training_example(
    group: list[LocalRegionCandidateRecord],
    num_buckets: int,
    *,
    loss_mode: str = "hard",
    softmax_temperature: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor | int]:
    """Build one listwise training example and best-IoU class index."""
    if loss_mode == "soft":
        return build_group_soft_training_example(
            group,
            num_buckets,
            softmax_temperature=softmax_temperature,
        )
    features = build_group_features(group, num_buckets)
    target_index = max(range(len(group)), key=lambda index: group[index].iou)
    return features, int(target_index)


def build_group_soft_training_example(
    group: list[LocalRegionCandidateRecord],
    num_buckets: int,
    *,
    softmax_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one listwise example with a soft IoU target distribution."""
    features = build_group_features(group, num_buckets)
    ious = torch.tensor([record.iou for record in group], dtype=torch.float32)
    target_distribution = torch.softmax(
        ious / max(float(softmax_temperature), 1e-6),
        dim=0,
    )
    return features, target_distribution


def build_group_features(
    group: list[LocalRegionCandidateRecord],
    num_buckets: int,
) -> torch.Tensor:
    parsed_region = parse_region_query(group[0].query).region
    return torch.stack(
        [
            build_candidate_record_feature(
                record.query,
                record.candidate_region,
                record.garment_box,
                record.candidate_box,
                parsed_region,
                num_buckets=num_buckets,
            )
            for record in group
        ]
    )


def evaluate_ranker(
    model: CandidateListwiseScorer,
    groups: list[list[LocalRegionCandidateRecord]],
    num_buckets: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    summary = _empty_summary()
    oracle_summary = _empty_summary()
    with torch.no_grad():
        for group in groups:
            features = build_group_features(group, num_buckets).to(device)
            scores = model(features)
            best_index = int(torch.argmax(scores).detach().cpu())
            oracle_index = max(range(len(group)), key=lambda index: group[index].iou)
            _update_summary(summary, group[best_index])
            _update_summary(oracle_summary, group[oracle_index])
    metrics = _finalize_summary(summary)
    metrics["oracle_best_iou"] = _finalize_summary(oracle_summary)
    return metrics


def _train_batch(
    model: CandidateListwiseScorer,
    optimizer: torch.optim.Optimizer,
    features: list[torch.Tensor],
    targets: list[torch.Tensor | int],
    device: torch.device,
    *,
    loss_mode: str,
) -> float:
    feature_tensor = torch.stack(features).to(device)
    batch_size, num_candidates, feature_dim = feature_tensor.shape
    logits = model(feature_tensor.reshape(batch_size * num_candidates, feature_dim))
    logits = logits.reshape(batch_size, num_candidates)
    if loss_mode == "soft":
        target_tensor = torch.stack(
            [target for target in targets if isinstance(target, torch.Tensor)]
        ).to(device)
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(target_tensor * log_probs).sum(dim=1).mean()
    else:
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        loss = nn.functional.cross_entropy(logits, target_tensor)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _empty_summary() -> dict[str, Any]:
    return {
        "num_records": 0,
        "iou_sum": 0.0,
        "weak_hit_at": {"0.3": 0, "0.5": 0},
        "selected_region_counts": Counter(),
        "by_region": {},
    }


def _update_summary(
    summary: dict[str, Any],
    selected: LocalRegionCandidateRecord,
) -> None:
    target_region = selected.target_region
    iou = selected.iou
    summary["num_records"] += 1
    summary["iou_sum"] += iou
    summary["selected_region_counts"][selected.candidate_region] += 1
    for threshold in summary["weak_hit_at"]:
        summary["weak_hit_at"][threshold] += int(iou >= float(threshold))

    region_summary = summary["by_region"].setdefault(target_region, _empty_summary())
    region_summary["num_records"] += 1
    region_summary["iou_sum"] += iou
    region_summary["selected_region_counts"][selected.candidate_region] += 1
    for threshold in region_summary["weak_hit_at"]:
        region_summary["weak_hit_at"][threshold] += int(iou >= float(threshold))


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    num_records = int(summary["num_records"])
    return {
        "num_records": num_records,
        "avg_top1_iou": summary["iou_sum"] / max(num_records, 1),
        "weak_hit_at": {
            threshold: count / max(num_records, 1)
            for threshold, count in summary["weak_hit_at"].items()
        },
        "selected_region_counts": dict(summary["selected_region_counts"]),
        "by_region": {
            region: _finalize_summary(values)
            for region, values in summary["by_region"].items()
        },
    }


def _group_key(record: LocalRegionCandidateRecord) -> tuple[Any, ...]:
    return (
        record.image,
        record.annotation,
        record.item_key,
        record.query,
        record.target_region,
        record.target_region_box,
        record.garment_box,
    )


def _train_stream_limit(max_groups: int, val_groups: int, val_offset: int) -> int:
    if val_offset < max_groups:
        return max_groups + val_groups
    return max_groups


if __name__ == "__main__":
    main()
