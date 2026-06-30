from __future__ import annotations

import argparse
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import LearnedRegionRanker
from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.utils.config import load_config


MANUAL_IOU_THRESHOLDS = (0.3, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 3.1.2 local-region localization against a small manual "
            "bbox benchmark. This evaluator does not use landmarks or weak labels."
        )
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Manual JSONL with image, query_text, target_region, and target_bbox.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ranker-checkpoint",
        default=None,
        help="Optional local-region ranker checkpoint.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/local_region_manual_eval.json",
        help="Path to save manual-eval summary and per-query records.",
    )
    return parser.parse_args()


def load_manual_records(
    jsonl_path: str | Path,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Load labeled manual bbox records from JSONL."""
    records: list[dict[str, Any]] = []
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if payload.get("label_status") == "skip":
                continue
            if payload.get("target_bbox") is None:
                continue
            records.append(parse_manual_record(payload, line_number=line_number))
            if max_records is not None and len(records) >= max_records:
                break
    return records


def parse_manual_record(
    payload: dict[str, Any],
    *,
    line_number: int | None = None,
) -> dict[str, Any]:
    """Validate one manual annotation record."""
    prefix = f"line {line_number}: " if line_number is not None else ""
    image = payload.get("image")
    query_text = payload.get("query_text")
    target_bbox = payload.get("target_bbox")
    if not isinstance(image, str) or not image:
        raise ValueError(f"{prefix}record must contain non-empty image")
    if not isinstance(query_text, str) or not query_text:
        raise ValueError(f"{prefix}record must contain non-empty query_text")
    bbox = parse_bbox(target_bbox, field_name=f"{prefix}target_bbox")
    target_region = payload.get("target_region")
    if target_region is not None and not isinstance(target_region, str):
        raise ValueError(f"{prefix}target_region must be a string when present")
    return {
        **payload,
        "image": image,
        "query_text": query_text,
        "target_region": target_region,
        "target_bbox": bbox,
    }


def parse_bbox(
    value: Any,
    *,
    field_name: str = "bbox",
) -> tuple[float, float, float, float]:
    """Parse an xyxy bbox and reject malformed or empty boxes."""
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"{field_name} must be [x1, y1, x2, y2]")
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numbers") from exc
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field_name} must satisfy x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize manual benchmark records."""
    status_counts = Counter(record["status"] for record in records)
    selected_regions = Counter(
        record["selected_region"]
        for record in records
        if record.get("selected_region") is not None
    )
    ranker_backends = Counter(
        record["ranker_backend"]
        for record in records
        if record.get("ranker_backend") is not None
    )
    ious = [
        record["manual_bbox_iou"]
        for record in records
        if record.get("manual_bbox_iou") is not None
    ]
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        region = record.get("target_region") or "unknown"
        by_region[region].append(record)

    return {
        "num_records": len(records),
        "status_counts": dict(status_counts),
        "ranker_backend_counts": dict(ranker_backends),
        "selected_region_counts": dict(selected_regions),
        "avg_manual_bbox_iou": mean(ious) if ious else 0.0,
        "manual_hit_at": {
            str(threshold): _hit_rate(ious, threshold)
            for threshold in MANUAL_IOU_THRESHOLDS
        },
        "by_region": {
            region: _summarize_region_records(region_records)
            for region, region_records in sorted(by_region.items())
        },
    }


def evaluate_manual_records(
    manual_records: list[dict[str, Any]],
    *,
    model_config: str,
    checkpoint: str,
    device: str | None = None,
    ranker_checkpoint: str | None = None,
) -> list[dict[str, Any]]:
    """Run full 3.1.2 inference and compare selected bbox with manual bbox."""
    config = load_config(model_config)
    predictor = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=checkpoint,
        device=device,
    )
    ranker = (
        LearnedRegionRanker(ranker_checkpoint, device=device)
        if ranker_checkpoint
        else None
    )

    records: list[dict[str, Any]] = []
    segmentation_cache: dict[str, Any] = {}
    for manual_record in manual_records:
        image_path = str(manual_record["image"])
        if image_path not in segmentation_cache:
            segmentation_cache[image_path] = predictor.predict(image_path)
        segmentation = segmentation_cache[image_path]
        result = localize_region_from_instances(
            segmentation,
            manual_record["query_text"],
            ranker=ranker,
        )

        predicted_box = (
            result.proposal.proposal.box
            if result.proposal is not None
            else None
        )
        manual_iou = (
            box_iou(predicted_box, manual_record["target_bbox"])
            if predicted_box is not None
            else None
        )
        records.append(
            {
                "id": manual_record.get("id"),
                "image": image_path,
                "query_text": manual_record["query_text"],
                "target_region": manual_record.get("target_region"),
                "target_bbox": list(manual_record["target_bbox"]),
                "status": result.status,
                "ranker_backend": result.ranker_backend,
                "selected_region": (
                    result.proposal.proposal.region if result.proposal else None
                ),
                "predicted_bbox": list(predicted_box) if predicted_box else None,
                "manual_bbox_iou": manual_iou,
                "segmentation_inference_time_ms": segmentation.inference_time_ms,
                "local_region_latency_ms": result.latency_ms,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    manual_records = load_manual_records(args.annotations, max_records=args.max_records)
    if not manual_records:
        raise ValueError(
            "No labeled manual records found. Fill target_bbox and set "
            "label_status to labeled before running evaluation."
        )

    records = evaluate_manual_records(
        manual_records,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        device=args.device,
        ranker_checkpoint=args.ranker_checkpoint,
    )
    summary = {
        "annotations": str(Path(args.annotations)),
        "num_labeled_records": len(manual_records),
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


def _summarize_region_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [
        record["manual_bbox_iou"]
        for record in records
        if record.get("manual_bbox_iou") is not None
    ]
    return {
        "num_records": len(records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "avg_manual_bbox_iou": mean(ious) if ious else 0.0,
        "manual_hit_at": {
            str(threshold): _hit_rate(ious, threshold)
            for threshold in MANUAL_IOU_THRESHOLDS
        },
        "selected_region_counts": dict(
            Counter(
                record["selected_region"]
                for record in records
                if record.get("selected_region") is not None
            )
        ),
    }


def _hit_rate(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value >= threshold for value in values) / len(values)


if __name__ == "__main__":
    main()
