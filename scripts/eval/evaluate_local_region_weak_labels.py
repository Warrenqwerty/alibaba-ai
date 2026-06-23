from __future__ import annotations

import argparse
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fashion_mm.models.instance_segmentation import FashionInstance
from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.models.local_region import parse_region_query
from fashion_mm.models.local_region import propose_region_from_landmarks
from fashion_mm.utils.config import load_config


DEFAULT_WEAK_QUERIES = [
    "这件衣服的领口",
    "衣服下方的下摆",
    "这件衣服的肩部",
]
WEAK_IOU_THRESHOLDS = (0.3, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 3.1.2 local regions against DeepFashion2 weak labels."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="DeepFashion2 image directory.",
    )
    parser.add_argument(
        "--anno-dir",
        required=True,
        help="DeepFashion2 annotation directory.",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_WEAK_QUERIES,
        help="Queries whose parsed regions can be approximated from landmarks.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument(
        "--output",
        default="outputs/local_region_weak_eval.json",
        help="Path to save weak-label summary and per-query records.",
    )
    return parser.parse_args()


def collect_annotations(anno_dir: Path, max_images: int | None) -> list[Path]:
    """Collect visible DeepFashion2 annotations in deterministic order."""
    annotation_paths = [
        path
        for path in sorted(anno_dir.glob("*.json"))
        if path.is_file() and not path.name.startswith(".")
    ]
    if max_images is not None:
        return annotation_paths[:max_images]
    return annotation_paths


def image_path_for_annotation(
    image_dir: Path,
    annotation_path: Path,
    annotation: dict[str, Any],
) -> Path:
    """Resolve a DeepFashion2 image path from annotation metadata."""
    candidates = []
    source = annotation.get("source")
    if isinstance(source, str):
        candidates.append(image_dir / source)
    candidates.append(image_dir / f"{annotation_path.stem}.jpg")
    candidates.append(image_dir / f"{annotation_path.stem}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for annotation: {annotation_path}")


def polygon_to_mask(
    segmentation: list[list[float]],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Rasterize DeepFashion2 polygon segmentation to a boolean mask."""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        points = [
            (float(polygon[index]), float(polygon[index + 1]))
            for index in range(0, len(polygon), 2)
        ]
        draw.polygon(points, outline=1, fill=1)
    return np.asarray(mask, dtype=bool)


def iter_items(annotation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return DeepFashion2 item entries in deterministic order."""
    return [
        (key, annotation[key])
        for key in sorted(annotation)
        if key.startswith("item") and isinstance(annotation[key], dict)
    ]


def match_annotation_item(
    instance: FashionInstance,
    annotation: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[str, dict[str, Any], np.ndarray, float] | None:
    """Match a predicted garment instance to the closest GT annotation item."""
    best_match: tuple[str, dict[str, Any], np.ndarray, float] | None = None
    for item_key, item in iter_items(annotation):
        mask = polygon_to_mask(item.get("segmentation", []), image_size)
        score = mask_iou(instance.mask, mask)
        if best_match is None or score > best_match[3]:
            best_match = (item_key, item, mask, score)
    return best_match


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute boolean mask IoU."""
    left = np.asarray(mask_a, dtype=bool)
    right = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize weak-label local-region evaluation records."""
    status_counts = Counter(record["status"] for record in records)
    weak_label_sources = Counter(
        record["weak_label_source"]
        for record in records
        if record.get("weak_label_source") is not None
    )
    weak_ious = [
        record["weak_iou"]
        for record in records
        if record.get("weak_iou") is not None
    ]
    garment_ious = [
        record["garment_iou"]
        for record in records
        if record.get("garment_iou") is not None
    ]
    by_region: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get("weak_iou") is not None:
            by_region[record["parsed_region"]].append(record["weak_iou"])

    summary: dict[str, Any] = {
        "num_records": len(records),
        "status_counts": dict(status_counts),
        "weak_label_source_counts": dict(weak_label_sources),
        "avg_garment_iou": mean(garment_ious) if garment_ious else 0.0,
        "avg_weak_iou": mean(weak_ious) if weak_ious else 0.0,
        "weak_hit_at": {
            str(threshold): _hit_rate(weak_ious, threshold)
            for threshold in WEAK_IOU_THRESHOLDS
        },
        "by_region": {
            region: {
                "num_records": len(values),
                "avg_weak_iou": mean(values),
                "weak_hit_at": {
                    str(threshold): _hit_rate(values, threshold)
                    for threshold in WEAK_IOU_THRESHOLDS
                },
            }
            for region, values in sorted(by_region.items())
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    anno_dir = Path(args.anno_dir)
    annotation_paths = collect_annotations(anno_dir, args.max_images)
    if not annotation_paths:
        raise ValueError(f"No annotations found in {anno_dir}")

    config = load_config(args.model_config)
    predictor = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    records: list[dict[str, Any]] = []
    for annotation_path in annotation_paths:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = image_path_for_annotation(image_dir, annotation_path, annotation)
        image_size = Image.open(image_path).size
        segmentation = predictor.predict(image_path)

        for query in args.queries:
            parsed_query = parse_region_query(query)
            result = localize_region_from_instances(segmentation, query)
            record: dict[str, Any] = {
                "image": str(image_path),
                "annotation": str(annotation_path),
                "query_text": query,
                "parsed_region": parsed_query.region,
                "status": result.status,
                "selected_region": (
                    result.proposal.proposal.region if result.proposal else None
                ),
                "weak_iou": None,
                "matched_item": None,
                "garment_iou": None,
                "local_region_latency_ms": result.latency_ms,
            }

            if result.selected_instance is None or result.proposal is None:
                records.append(record)
                continue

            match = match_annotation_item(result.selected_instance, annotation, image_size)
            if match is None:
                record["status"] = "no_annotation_item"
                records.append(record)
                continue

            item_key, item, gt_garment_mask, garment_iou = match
            weak_target = propose_region_from_landmarks(
                gt_garment_mask,
                item["bounding_box"],
                item.get("landmarks", []),
                parsed_query.region or "",
            )
            record.update(
                {
                    "matched_item": item_key,
                    "garment_iou": garment_iou,
                    "weak_label_source": weak_target.source,
                    "weak_iou": mask_iou(result.proposal.proposal.mask, weak_target.mask),
                }
            )
            records.append(record)

    summary = {
        "image_dir": str(image_dir),
        "anno_dir": str(anno_dir),
        "num_images": len(annotation_paths),
        "queries": args.queries,
        **summarize_records(records),
        "records": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


def _hit_rate(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value >= threshold for value in values) / len(values)


if __name__ == "__main__":
    main()
