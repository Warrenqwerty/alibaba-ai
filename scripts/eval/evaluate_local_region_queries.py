from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.models.local_region.visualization import draw_local_region_result
from fashion_mm.utils.config import load_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_QUERIES = [
    "这件衣服的领口",
    "左边的袖口",
    "右边的袖口",
    "衣服下方的下摆",
    "这件衣服上的碎花图案",
    "衣服上的拉链",
    "右侧的口袋",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-vocabulary 3.1.2 local-region sanity evaluation."
    )
    parser.add_argument("--image-dir", required=True, help="Directory of RGB images.")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_QUERIES,
        help="Natural-language queries to run on each image.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument(
        "--output",
        default="outputs/local_region_query_eval.json",
        help="Path to save summary and per-query records.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Optional directory for selected visualization images.",
    )
    parser.add_argument("--vis-count", type=int, default=20)
    return parser.parse_args()


def collect_images(image_dir: Path, max_images: int | None) -> list[Path]:
    """Collect visible image files in deterministic order."""
    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if max_images is not None:
        return image_paths[:max_images]
    return image_paths


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize query-localization records without ground-truth labels."""
    status_counts = Counter(record["status"] for record in records)
    selected_regions = Counter(
        record["region"]["region"]
        for record in records
        if record.get("region") is not None
    )
    ranker_backends = Counter(record["ranker_backend"] for record in records)
    latencies = [record["latency_ms"] for record in records]
    scores = [
        record["region"]["match_score"]
        for record in records
        if record.get("region") is not None
    ]
    return {
        "num_records": len(records),
        "status_counts": dict(status_counts),
        "selected_region_counts": dict(selected_regions),
        "ranker_backend_counts": dict(ranker_backends),
        "avg_local_region_latency_ms": mean(latencies) if latencies else 0.0,
        "avg_match_score": mean(scores) if scores else 0.0,
    }


def main() -> None:
    args = parse_args()
    image_paths = collect_images(Path(args.image_dir), args.max_images)
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    config = load_config(args.model_config)
    predictor = FashionInstanceSegmentationPredictor(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    records: list[dict[str, Any]] = []
    segmentation_times: list[float] = []
    visualized = 0
    for image_path in image_paths:
        segmentation = predictor.predict(image_path)
        segmentation_times.append(segmentation.inference_time_ms)
        for query in args.queries:
            result = localize_region_from_instances(segmentation, query)
            record = {
                "image": str(image_path),
                "query_text": query,
                "segmentation_inference_time_ms": segmentation.inference_time_ms,
                **result.to_dict(include_mask=False),
            }
            records.append(record)

            if args.vis_dir and visualized < args.vis_count:
                safe_query = _safe_stem(query)
                output_path = (
                    Path(args.vis_dir)
                    / f"{image_path.stem}_{visualized:03d}_{safe_query}.jpg"
                )
                draw_local_region_result(image_path, result, output_path)
                visualized += 1

    summary = {
        "image_dir": str(Path(args.image_dir)),
        "num_images": len(image_paths),
        "queries": args.queries,
        "avg_segmentation_inference_time_ms": mean(segmentation_times),
        **summarize_records(records),
        "records": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, ensure_ascii=False, indent=2))


def _safe_stem(text: str, max_chars: int = 24) -> str:
    safe_chars = [char if char.isalnum() else "_" for char in text[:max_chars]]
    return "".join(safe_chars).strip("_") or "query"


if __name__ == "__main__":
    main()
