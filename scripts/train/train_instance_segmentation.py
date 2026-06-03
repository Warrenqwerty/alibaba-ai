from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fashion_mm.data_loaders import DeepFashion2Dataset
from fashion_mm.models.instance_segmentation import build_mask_rcnn
from fashion_mm.utils.config import load_config
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3.1.1 fashion segmentation.")
    parser.add_argument("--model-config", default="configs/model/instance_segmentation.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.paths_config, args.model_config)
    device = torch.device(config["inference"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable; training on CPU for debugging only.")
        device = torch.device("cpu")

    train_dataset = DeepFashion2Dataset(
        config["deepfashion2"]["train_image_dir"],
        config["deepfashion2"]["train_anno_dir"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_mask_rcnn(
        num_classes=int(config["model"]["num_classes"]),
        pretrained=bool(config["model"].get("pretrained", True)),
    ).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        LOGGER.info("Resumed checkpoint: %s", args.resume)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0001)),
    )

    output_dir = Path(args.output_dir or config["checkpoint_root"]) / "instance_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()

    for epoch in range(int(config["training"]["num_epochs"])):
        total_loss = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            losses = model(images, targets)
            loss = sum(loss_value for loss_value in losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            if step % int(config["training"].get("log_interval", 50)) == 0:
                LOGGER.info(
                    "epoch=%s step=%s loss=%.4f",
                    epoch + 1,
                    step,
                    total_loss / step,
                )

        checkpoint_path = output_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            checkpoint_path,
        )
        LOGGER.info("Saved checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
