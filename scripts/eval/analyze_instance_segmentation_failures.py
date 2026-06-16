from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from fashion_mm.data_loaders.deepfashion2 import DeepFashion2Dataset
from fashion_mm.models.instance_segmentation import build_mask_rcnn
from fashion_mm.utils.config import load_config
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze validation failure cases for fashion segmentation."
    )
    parser.add_argument("--model-config", default="configs/model/instance_segmentation.yaml")
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--low-iou-threshold", type=float, default=0.75)
    parser.add_argument("--miss-iou-threshold", type=float, default=0.3)
    parser.add_argument("--output", default="outputs/failure_analysis/failure_cases.json")
    parser.add_argument("--vis-dir", default="outputs/failure_analysis/visualizations")
    parser.add_argument("--vis-per-reason", type=int, default=30)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, gt).sum() / union)


def load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device):
    model = build_mask_rcnn(
        num_classes=int(config["model"]["num_classes"]),
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    return model


def best_match(
    gt_mask: np.ndarray,
    pred_masks: np.ndarray,
    pred_labels: np.ndarray,
    label_id: int | None = None,
) -> tuple[int | None, float]:
    best_index = None
    best_iou = 0.0
    for index, (pred_mask, pred_label) in enumerate(zip(pred_masks, pred_labels)):
        if label_id is not None and int(pred_label) != int(label_id):
            continue
        iou = mask_iou(pred_mask, gt_mask)
        if iou > best_iou:
            best_index = index
            best_iou = iou
    return best_index, best_iou


def classify_failure_reason(
    gt_label_name: str,
    best_any_label_name: str | None,
    best_any_iou: float,
    best_same_iou: float,
    low_iou_threshold: float,
    miss_iou_threshold: float,
) -> str | None:
    if best_same_iou >= low_iou_threshold:
        return None

    if best_any_iou < miss_iou_threshold:
        return f"missed_{gt_label_name}"

    if best_any_label_name and best_any_label_name != gt_label_name:
        return f"{gt_label_name}_confused_as_{best_any_label_name}"

    return f"low_iou_{gt_label_name}"


def draw_failure_case(
    image_path: Path,
    gt_mask: np.ndarray,
    gt_box: list[float],
    gt_label_name: str,
    pred_mask: np.ndarray | None,
    pred_box: list[float] | None,
    pred_label_name: str | None,
    pred_score: float | None,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    gt_layer = Image.new("RGBA", image.size, (0, 120, 255, 90))
    gt_mask_image = Image.fromarray(gt_mask.astype(np.uint8) * 255)
    overlay = Image.composite(gt_layer, overlay, gt_mask_image)
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(gt_box, outline=(0, 120, 255, 255), width=3)
    draw.text(
        (gt_box[0], max(0, gt_box[1] - 18)),
        f"GT {gt_label_name}",
        fill=(0, 80, 220, 255),
    )

    if pred_mask is not None and pred_box is not None:
        pred_layer = Image.new("RGBA", image.size, (255, 80, 0, 90))
        pred_mask_image = Image.fromarray(pred_mask.astype(np.uint8) * 255)
        overlay = Image.composite(pred_layer, overlay, pred_mask_image)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(pred_box, outline=(255, 80, 0, 255), width=3)
        score_text = f"{pred_score:.2f}" if pred_score is not None else "n/a"
        pred_text = f"Pred {pred_label_name} {score_text}"
        draw.text(
            (pred_box[0], max(0, pred_box[1] - 18)),
            pred_text,
            fill=(220, 60, 0, 255),
        )

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

    cases: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    reason_visualized: Counter[str] = Counter()
    same_class_ious: defaultdict[str, list[float]] = defaultdict(list)
    confusion_pairs: Counter[str] = Counter()

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
        gt_boxes = target["boxes"].numpy()
        annotation_path = dataset.annotation_paths[index]
        image_path = dataset.image_dir / f"{annotation_path.stem}.jpg"

        for gt_instance_index, (gt_mask, gt_label, gt_box) in enumerate(
            zip(gt_masks, gt_labels, gt_boxes)
        ):
            gt_label_id = int(gt_label)
            gt_label_name = categories.get(gt_label_id, str(gt_label_id))
            class_counts[gt_label_name] += 1

            best_same_index, best_same_iou = best_match(
                gt_mask,
                pred_masks,
                pred_labels,
                label_id=gt_label_id,
            )
            best_any_index, best_any_iou = best_match(gt_mask, pred_masks, pred_labels)
            same_class_ious[gt_label_name].append(best_same_iou)

            best_any_label_name = None
            if best_any_index is not None:
                best_any_label_name = categories.get(
                    int(pred_labels[best_any_index]),
                    str(int(pred_labels[best_any_index])),
                )

            reason = classify_failure_reason(
                gt_label_name,
                best_any_label_name,
                best_any_iou,
                best_same_iou,
                args.low_iou_threshold,
                args.miss_iou_threshold,
            )
            if reason is None:
                continue

            reason_counts[reason] += 1
            if best_any_label_name and best_any_label_name != gt_label_name:
                confusion_pairs[f"{gt_label_name}->{best_any_label_name}"] += 1

            match_index = best_any_index if best_any_index is not None else best_same_index
            pred_label_name = None
            pred_score = None
            pred_box = None
            pred_mask = None
            if match_index is not None:
                pred_label_name = categories.get(
                    int(pred_labels[match_index]),
                    str(int(pred_labels[match_index])),
                )
                pred_score = float(scores[match_index])
                pred_box = [float(value) for value in pred_boxes[match_index].tolist()]
                pred_mask = pred_masks[match_index]

            case = {
                "image": image_path.name,
                "annotation": annotation_path.name,
                "gt_instance_index": gt_instance_index,
                "gt_label": gt_label_name,
                "reason": reason,
                "best_same_class_iou": best_same_iou,
                "best_any_iou": best_any_iou,
                "matched_pred_label": pred_label_name,
                "matched_pred_score": pred_score,
                "gt_box": [float(value) for value in gt_box.tolist()],
                "pred_box": pred_box,
            }
            cases.append(case)

            if args.vis_dir and reason_visualized[reason] < args.vis_per_reason:
                output_path = (
                    Path(args.vis_dir)
                    / reason
                    / f"{annotation_path.stem}_{gt_instance_index:02d}.jpg"
                )
                draw_failure_case(
                    image_path,
                    gt_mask,
                    case["gt_box"],
                    gt_label_name,
                    pred_mask,
                    pred_box,
                    pred_label_name,
                    pred_score,
                    output_path,
                )
                reason_visualized[reason] += 1

        if (index + 1) % args.log_interval == 0:
            LOGGER.info("analyzed %s/%s images", index + 1, max_images)

    per_class = {}
    for label_name, ious in same_class_ious.items():
        per_class[label_name] = {
            "gt_instances": class_counts[label_name],
            "mean_best_same_class_iou": mean(ious) if ious else 0.0,
            "low_iou_or_missed_instances": sum(
                iou < args.low_iou_threshold for iou in ious
            ),
        }

    output = {
        "checkpoint": args.checkpoint,
        "max_images": max_images,
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "low_iou_threshold": args.low_iou_threshold,
        "miss_iou_threshold": args.miss_iou_threshold,
        "num_failure_cases": len(cases),
        "reason_counts": dict(reason_counts.most_common()),
        "confusion_pairs": dict(confusion_pairs.most_common()),
        "per_class": per_class,
        "cases": cases,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {key: value for key, value in output.items() if key != "cases"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
