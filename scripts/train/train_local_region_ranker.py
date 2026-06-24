from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from fashion_mm.data_loaders import iter_local_region_query_records
from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import build_pair_feature
from fashion_mm.models.local_region import candidate_boxes_from_garment
from fashion_mm.models.local_region import HashingTextRegionScorer
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight 3.1.2 text-region ranker."
    )
    parser.add_argument("--records", required=True, help="Weak query JSONL path.")
    parser.add_argument(
        "--output",
        default="outputs/local_region_ranker.pt",
        help="Checkpoint path.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-buckets", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-records", type=int, default=50000)
    parser.add_argument("--val-records", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HashingTextRegionScorer(
        num_buckets=args.num_buckets,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    validation_records = list(
        iter_local_region_query_records(args.records, max_records=args.val_records)
    )
    LOGGER.info("Loaded validation records: %s", len(validation_records))

    for epoch in range(args.num_epochs):
        model.train()
        batch_features: list[torch.Tensor] = []
        batch_targets: list[float] = []
        total_loss = 0.0
        total_batches = 0
        train_limit = args.max_records + args.val_records

        for index, record in enumerate(
            iter_local_region_query_records(args.records, max_records=train_limit)
        ):
            if index < args.val_records:
                continue
            for feature, target in build_training_examples(record, args.num_buckets):
                batch_features.append(feature)
                batch_targets.append(target)
            if len(batch_features) >= args.batch_size:
                loss = _train_batch(
                    model,
                    optimizer,
                    loss_fn,
                    batch_features,
                    batch_targets,
                    device,
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
                loss_fn,
                batch_features,
                batch_targets,
                device,
            )
            total_loss += loss
            total_batches += 1

        metrics = evaluate_ranker(model, validation_records, args.num_buckets, device)
        LOGGER.info(
            "epoch=%s loss=%.4f val_top1_iou=%.4f",
            epoch + 1,
            total_loss / max(total_batches, 1),
            metrics["top1_iou"],
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_buckets": args.num_buckets,
        "hidden_dim": args.hidden_dim,
        "validation_metrics": evaluate_ranker(
            model,
            validation_records,
            args.num_buckets,
            device,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    LOGGER.info("Saved local-region ranker checkpoint: %s", output_path)


def build_training_examples(record, num_buckets: int):
    """Build candidate features and soft IoU labels for one weak record."""
    candidates = candidate_boxes_from_garment(record.garment_box)
    for candidate in candidates:
        yield (
            build_pair_feature(
                record.query,
                candidate,
                record.garment_box,
                num_buckets=num_buckets,
            ),
            box_iou(candidate.box, record.region_box),
        )


def evaluate_ranker(
    model: HashingTextRegionScorer,
    records,
    num_buckets: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate top-1 selected candidate IoU against weak target boxes."""
    model.eval()
    ious: list[float] = []
    with torch.no_grad():
        for record in records:
            candidates = candidate_boxes_from_garment(record.garment_box)
            features = torch.stack(
                [
                    build_pair_feature(
                        record.query,
                        candidate,
                        record.garment_box,
                        num_buckets=num_buckets,
                    )
                    for candidate in candidates
                ]
            ).to(device)
            scores = model(features)
            best_index = int(torch.argmax(scores).detach().cpu())
            ious.append(box_iou(candidates[best_index].box, record.region_box))
    return {"top1_iou": sum(ious) / max(len(ious), 1)}


def _train_batch(
    model: HashingTextRegionScorer,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    features: list[torch.Tensor],
    targets: list[float],
    device: torch.device,
) -> float:
    feature_tensor = torch.stack(features).to(device)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    logits = model(feature_tensor)
    loss = loss_fn(logits, target_tensor)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


if __name__ == "__main__":
    main()
