from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import LearnedRegionRanker
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.models.local_region.query import ParsedRegionQuery
from fashion_mm.models.local_region.query import parse_region_query
from fashion_mm.models.local_region.visualization import draw_local_region_result
from fashion_mm.utils.config import load_config
from scripts.eval.evaluate_gated_hybrid_manual_labels import (
    DEFAULT_GROUNDING_MODEL_NAME,
    DEFAULT_GROUNDING_REGIONS,
)
from scripts.eval.evaluate_local_region_queries import DEFAULT_QUERIES
from scripts.eval.evaluate_local_region_queries import collect_images
from scripts.eval.evaluate_local_region_queries import _safe_stem
from scripts.eval.evaluate_pretrained_grounding_manual_labels import BACKEND_NAMES
from scripts.eval.evaluate_pretrained_grounding_manual_labels import HFZeroShotGrounder
from scripts.eval.evaluate_pretrained_grounding_manual_labels import build_prompts
from scripts.inference.predict_gated_hybrid_local_region import draw_grounding_result
from scripts.inference.predict_gated_hybrid_local_region import grounding_payload
from scripts.inference.predict_gated_hybrid_local_region import should_use_grounding_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a batch sanity evaluation for the explicit gated 3.1.2 policy: "
            "configured semantic regions use pretrained grounding, all other "
            "queries use the heuristic local-region pipeline."
        )
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
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="3.1.1 instance-segmentation checkpoint for heuristic-routed queries.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ranker-checkpoint",
        default=None,
        help="Optional experimental learned local-region ranker for heuristic route.",
    )
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument(
        "--grounding-regions",
        nargs="+",
        default=list(DEFAULT_GROUNDING_REGIONS),
        help="Parsed region names routed to pretrained grounding.",
    )
    parser.add_argument(
        "--grounding-model-name",
        default=DEFAULT_GROUNDING_MODEL_NAME,
        help="HuggingFace model name or local model directory.",
    )
    parser.add_argument(
        "--grounding-backend",
        choices=BACKEND_NAMES,
        default="auto",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("english", "chinese", "both"),
        default="english",
    )
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument(
        "--output",
        default="outputs/local_region_gated_query_eval.json",
        help="Path to save summary and per-query records.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Optional directory for selected visualization images.",
    )
    parser.add_argument("--vis-count", type=int, default=20)
    return parser.parse_args()


def summarize_gated_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize gated query-localization records without ground-truth labels."""
    status_counts = Counter(record["status"] for record in records)
    selected_regions = Counter(
        record["region"]["region"]
        for record in records
        if record.get("region") is not None
    )
    ranker_backends = Counter(record["ranker_backend"] for record in records)
    route_counts = Counter(record["gated_policy_route"] for record in records)
    latencies = [record["latency_ms"] for record in records]
    route_latencies: dict[str, list[float]] = {}
    for record in records:
        route_latencies.setdefault(record["gated_policy_route"], []).append(
            record["latency_ms"]
        )
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
        "gated_policy_route_counts": dict(route_counts),
        "avg_local_region_latency_ms": mean(latencies) if latencies else 0.0,
        "avg_local_region_latency_by_route_ms": {
            route: mean(values) for route, values in route_latencies.items()
        },
        "avg_match_score": mean(scores) if scores else 0.0,
    }


def parsed_queries_for_route(
    queries: list[str],
) -> list[tuple[str, ParsedRegionQuery]]:
    """Parse query strings once so route decisions are deterministic and visible."""
    return [(query, parse_region_query(query)) for query in queries]


def main() -> None:
    args = parse_args()
    image_paths = collect_images(Path(args.image_dir), args.max_images)
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    grounding_regions = set(args.grounding_regions)
    parsed_queries = parsed_queries_for_route(args.queries)
    has_grounding_route = any(
        should_use_grounding_route(parsed_query, grounding_regions)
        for _, parsed_query in parsed_queries
    )
    has_heuristic_route = any(
        not should_use_grounding_route(parsed_query, grounding_regions)
        for _, parsed_query in parsed_queries
    )

    predictor = None
    ranker = None
    if has_heuristic_route:
        config = load_config(args.model_config)
        predictor = FashionInstanceSegmentationPredictor(
            config=config,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        ranker = (
            LearnedRegionRanker(args.ranker_checkpoint, device=args.device)
            if args.ranker_checkpoint
            else None
        )

    grounder = (
        HFZeroShotGrounder(
            args.grounding_model_name,
            backend=args.grounding_backend,
            device=args.device,
            score_threshold=args.score_threshold,
        )
        if has_grounding_route
        else None
    )

    records: list[dict[str, Any]] = []
    segmentation_times: list[float] = []
    visualized = 0
    for image_path in image_paths:
        segmentation = None
        pil_image = None
        if has_heuristic_route:
            assert predictor is not None
            segmentation = predictor.predict(image_path)
            segmentation_times.append(segmentation.inference_time_ms)

        for query, parsed_query in parsed_queries:
            if should_use_grounding_route(parsed_query, grounding_regions):
                if grounder is None:
                    raise RuntimeError("Grounding route selected, but grounder is missing.")
                if pil_image is None:
                    pil_image = Image.open(image_path).convert("RGB")
                record = evaluate_grounding_query(
                    image_path,
                    query,
                    parsed_query,
                    pil_image,
                    grounder=grounder,
                    prompt_mode=args.prompt_mode,
                    grounding_model_name=args.grounding_model_name,
                )
                if args.vis_dir and visualized < args.vis_count:
                    output_path = visualization_path(Path(args.vis_dir), image_path, visualized, query)
                    draw_grounding_result(image_path, record, output_path)
                    visualized += 1
                records.append(record)
                continue

            if segmentation is None:
                raise RuntimeError("Heuristic route selected, but segmentation is missing.")
            result = localize_region_from_instances(segmentation, query, ranker=ranker)
            record = {
                "image": str(image_path),
                "query_text": query,
                "segmentation_inference_time_ms": segmentation.inference_time_ms,
                **result.to_dict(include_mask=False),
                "gated_policy_route": "heuristic",
            }
            records.append(record)
            if args.vis_dir and visualized < args.vis_count:
                output_path = visualization_path(Path(args.vis_dir), image_path, visualized, query)
                draw_local_region_result(image_path, result, output_path)
                visualized += 1

    summary = {
        "image_dir": str(Path(args.image_dir)),
        "num_images": len(image_paths),
        "queries": args.queries,
        "grounding_regions": sorted(grounding_regions),
        "grounding_model_name": args.grounding_model_name if has_grounding_route else None,
        "grounding_backend": args.grounding_backend if has_grounding_route else None,
        "prompt_mode": args.prompt_mode,
        "score_threshold": args.score_threshold,
        "avg_segmentation_inference_time_ms": (
            mean(segmentation_times) if segmentation_times else 0.0
        ),
        **summarize_gated_records(records),
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


def evaluate_grounding_query(
    image_path: Path,
    query: str,
    parsed_query: ParsedRegionQuery,
    image: Image.Image,
    *,
    grounder: HFZeroShotGrounder,
    prompt_mode: str,
    grounding_model_name: str,
) -> dict[str, Any]:
    prompts = build_prompts(
        query,
        parsed_query.region,
        prompt_mode=prompt_mode,
    )
    start = time.perf_counter()
    prediction = grounder.predict(image, prompts)
    latency_ms = (time.perf_counter() - start) * 1000.0
    payload = grounding_payload(
        image_path=image_path,
        query=query,
        parsed_query=parsed_query,
        prediction=prediction,
        prompts=prompts,
        grounder_backend=grounder.backend,
        grounding_model_name=grounding_model_name,
        latency_ms=latency_ms,
    )
    payload["query_text"] = query
    payload["segmentation_inference_time_ms"] = None
    return payload


def visualization_path(
    vis_dir: Path,
    image_path: Path,
    index: int,
    query: str,
) -> Path:
    safe_query = _safe_stem(query)
    return vis_dir / f"{image_path.stem}_{index:03d}_{safe_query}.jpg"


if __name__ == "__main__":
    main()
