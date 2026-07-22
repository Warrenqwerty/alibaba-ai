from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fashion_mm.models.attributes import FashionAttributePredictor
from fashion_mm.utils.latency import summarize_timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark warm single-image latency for the 3.1.3 image-and-mask "
            "attribute contract."
        )
    )
    parser.add_argument("image", help="Path to the RGB product image.")
    parser.add_argument(
        "--mask",
        required=True,
        help="Path to the target-region mask.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attributes", nargs="+", default=None)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")
    if args.target_ms <= 0:
        raise ValueError("--target-ms must be positive.")

    predictor = FashionAttributePredictor(args.checkpoint, device=args.device)
    for _ in range(args.warmup_runs):
        predictor.predict(args.image, args.mask, attributes=args.attributes)

    preprocessing: list[float] = []
    inference: list[float] = []
    reported_total: list[float] = []
    wall_total: list[float] = []
    result = None
    for _ in range(args.runs):
        started_at = time.perf_counter()
        result = predictor.predict(
            args.image,
            args.mask,
            attributes=args.attributes,
        )
        wall_total.append((time.perf_counter() - started_at) * 1000.0)
        preprocessing.append(result.preprocessing_time_ms)
        inference.append(result.inference_time_ms)
        reported_total.append(result.total_time_ms)

    if result is None:
        raise RuntimeError("Latency benchmark produced no results.")

    wall_summary = summarize_timings(wall_total)
    payload = {
        "image": str(Path(args.image)),
        "mask": str(Path(args.mask)),
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(args.device),
        "attributes": [item.attribute_name for item in result.predictions],
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.runs,
        "preprocessing_ms": summarize_timings(preprocessing),
        "model_inference_ms": summarize_timings(inference),
        "reported_total_ms": summarize_timings(reported_total),
        "wall_total_ms": wall_summary,
        "latency_target": {
            "threshold_ms": args.target_ms,
            "primary_metric": "wall_total_ms.p95",
            "primary_value_ms": wall_summary["p95"],
            "passed": wall_summary["p95"] <= args.target_ms,
            "observed_max_ms": wall_summary["max"],
            "max_passed": wall_summary["max"] <= args.target_ms,
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
