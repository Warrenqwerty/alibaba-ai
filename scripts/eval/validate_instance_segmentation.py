from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from fashion_mm.data_loaders.deepfashion2 import DeepFashion2Dataset
from fashion_mm.models.instance_segmentation import build_mask_rcnn
from fashion_mm.utils.config import load_config
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 3.1.1 segmentation.")
    parser.add_argument("--model-config", default="configs/model/instance_segmentation.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--output", default="outputs/validation_metrics.json")
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--vis-count", type=int, default=10)
    return parser.parse_args()


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, gt).sum() / union)


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = build_mask_rcnn(
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    return model


def best_iou_for_gt(
    gt_mask: np.ndarray,
    gt_label: int,
    pred_masks: np.ndarray,
    pred_labels: np.ndarray,
) -> float:
    best_iou = 0.0
    for pred_mask, pred_label in zip(pred_masks, pred_labels):
        if int(pred_label) != int(gt_label):
            continue
        best_iou = max(best_iou, mask_iou(pred_mask, gt_mask))
    return best_iou


def draw_visualization(
    image_path: Path,
    pred_masks: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    categories: dict[int, str],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (0, 255, 80, 100),
        (255, 80, 0, 100),
        (80, 160, 255, 100),
        (255, 220, 0, 100),
        (220, 80, 255, 100),
    ]

    for index, (mask, box, label, score) in enumerate(
        zip(pred_masks, pred_boxes, pred_labels, pred_scores)
    ):
        color = colors[index % len(colors)]
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        color_layer = Image.new("RGBA", image.size, color)
        overlay = Image.composite(color_layer, overlay, mask_image)
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=3)
        text = f"{categories.get(int(label), 'unknown')} {float(score):.2f}"
        draw.text((x1, max(0, y1 - 16)), text, fill=color[:3] + (255,))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.paths_config, args.model_config)
    score_threshold = float(
        args.score_threshold
        if args.score_threshold is not None
        else config["inference"]["score_threshold"]
    )
    mask_threshold = float(
        args.mask_threshold
        if args.mask_threshold is not None
        else config["inference"]["mask_threshold"]
    )
    device = torch.device(args.device or config["inference"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    dataset = DeepFashion2Dataset(
        config["deepfashion2"]["val_image_dir"],
        config["deepfashion2"]["val_anno_dir"],
    )
    model = load_model(config, args.checkpoint, device)
    categories = {int(key): value for key, value in config["categories"].items()}
    max_images = min(args.max_images, len(dataset)) if args.max_images else len(dataset)

    gt_ious: list[float] = []
    pred_counts: list[int] = []
    score_values: list[float] = []
    visualized = 0

    for index in range(max_images):
        image_tensor, target = dataset[index]
        prediction = model([image_tensor.to(device)])[0]

        scores = prediction["scores"].detach().cpu().numpy()
        keep = scores >= score_threshold
        scores = scores[keep]
        pred_labels = prediction["labels"].detach().cpu().numpy()[keep]
        pred_boxes = prediction["boxes"].detach().cpu().numpy()[keep]
        pred_masks = (
            prediction["masks"].detach().cpu().numpy()[keep, 0] >= mask_threshold
        )

        gt_masks = target["masks"].numpy().astype(bool)
        gt_labels = target["labels"].numpy()
        for gt_mask, gt_label in zip(gt_masks, gt_labels):
            gt_ious.append(best_iou_for_gt(gt_mask, gt_label, pred_masks, pred_labels))

        pred_counts.append(int(len(scores)))
        score_values.extend(float(score) for score in scores.tolist())

        if args.vis_dir and visualized < args.vis_count:
            annotation_path = dataset.annotation_paths[index]
            image_path = dataset.image_dir / f"{annotation_path.stem}.jpg"
            draw_visualization(
                image_path,
                pred_masks,
                pred_boxes,
                pred_labels,
                scores,
                categories,
                Path(args.vis_dir) / f"{annotation_path.stem}_pred.jpg",
            )
            visualized += 1

        if (index + 1) % 20 == 0:
            LOGGER.info("validated %s/%s images", index + 1, max_images)

    metrics = {
        "checkpoint": args.checkpoint,
        "max_images": max_images,
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "gt_instances": len(gt_ious),
        "mean_best_mask_iou": mean(gt_ious) if gt_ious else 0.0,
        "recall_iou_50": mean([iou >= 0.5 for iou in gt_ious]) if gt_ious else 0.0,
        "recall_iou_75": mean([iou >= 0.75 for iou in gt_ious]) if gt_ious else 0.0,
        "avg_predictions_per_image": mean(pred_counts) if pred_counts else 0.0,
        "avg_prediction_score": mean(score_values) if score_values else 0.0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
