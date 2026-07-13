from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.local_region import box_iou
from scripts.eval.evaluate_local_region_manual_labels import load_manual_records
from scripts.eval.evaluate_pretrained_grounding_manual_labels import BACKEND_NAMES
from scripts.eval.evaluate_pretrained_grounding_manual_labels import HFZeroShotGrounder
from scripts.eval.evaluate_pretrained_grounding_manual_labels import PROMPT_PROFILES
from scripts.eval.evaluate_pretrained_grounding_manual_labels import build_prompts
from scripts.eval.evaluate_pretrained_grounding_manual_labels import summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GroundingDINO/OWL prompt profiles on one manual benchmark. "
            "The model is loaded once and reused for every profile."
        )
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model-name", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="auto")
    parser.add_argument("--prompt-mode", choices=("english", "chinese", "both"), default="english")
    parser.add_argument(
        "--prompt-profiles",
        nargs="+",
        choices=PROMPT_PROFILES,
        default=list(PROMPT_PROFILES),
        help="Prompt profiles evaluated in the provided order.",
    )
    parser.add_argument(
        "--target-regions",
        nargs="+",
        default=["pattern", "pocket"],
        help="Manual target regions included in the ablation.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/local_region_grounding_prompt_profiles.json",
    )
    return parser.parse_args()


def select_target_regions(
    records: list[dict[str, Any]],
    target_regions: set[str],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if str(record.get("target_region") or "") in target_regions
    ]


def evaluate_prompt_profiles(
    manual_records: list[dict[str, Any]],
    *,
    model_name: str,
    backend: str,
    prompt_mode: str,
    prompt_profiles: list[str],
    device: str | None,
    score_threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate profiles fairly while reusing one pretrained grounding model."""
    if not manual_records:
        raise ValueError("No manual records available for prompt-profile evaluation")
    grounder = HFZeroShotGrounder(
        model_name,
        backend=backend,
        device=device,
        score_threshold=score_threshold,
    )
    image_cache: dict[str, Image.Image] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    for profile in prompt_profiles:
        records: list[dict[str, Any]] = []
        for manual_record in manual_records:
            image_path = str(manual_record["image"])
            if image_path not in image_cache:
                image_cache[image_path] = Image.open(image_path).convert("RGB")
            prompts = build_prompts(
                manual_record["query_text"],
                manual_record.get("target_region"),
                prompt_mode=prompt_mode,
                prompt_profile=profile,
            )
            start = time.perf_counter()
            prediction = grounder.predict(image_cache[image_path], prompts)
            latency_ms = (time.perf_counter() - start) * 1000.0
            best = prediction["best"]
            predicted_bbox = tuple(best["bbox"]) if best is not None else None
            records.append(
                {
                    "id": manual_record.get("id"),
                    "image": image_path,
                    "query_text": manual_record["query_text"],
                    "target_region": manual_record.get("target_region"),
                    "target_bbox": list(manual_record["target_bbox"]),
                    "status": prediction["status"],
                    "ranker_backend": f"prompt_profile_grounding_{grounder.backend}",
                    "selected_region": best["prompt"] if best is not None else None,
                    "predicted_bbox": list(predicted_bbox) if predicted_bbox else None,
                    "manual_bbox_iou": (
                        box_iou(predicted_bbox, manual_record["target_bbox"])
                        if predicted_bbox is not None
                        else 0.0
                    ),
                    "local_region_latency_ms": latency_ms,
                    "score": best["score"] if best is not None else None,
                    "prompt_mode": prompt_mode,
                    "prompt_profile": profile,
                    "prompts": prompts,
                    "detections": prediction["detections"][:5],
                }
            )
        results[profile] = records
    return results


def profile_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_records(records)
    summary["status_counts"] = dict(Counter(record["status"] for record in records))
    summary["by_region"] = {
        region: summarize_records(
            [record for record in records if record.get("target_region") == region]
        )
        for region in sorted({str(record.get("target_region") or "unknown") for record in records})
    }
    return summary


def main() -> None:
    args = parse_args()
    all_records = load_manual_records(args.annotations, max_records=args.max_records)
    target_regions = set(args.target_regions)
    manual_records = select_target_regions(all_records, target_regions)
    if not manual_records:
        raise ValueError(f"No labeled records found for target regions: {sorted(target_regions)}")
    profile_records = evaluate_prompt_profiles(
        manual_records,
        model_name=args.model_name,
        backend=args.backend,
        prompt_mode=args.prompt_mode,
        prompt_profiles=args.prompt_profiles,
        device=args.device,
        score_threshold=args.score_threshold,
    )
    output = {
        "annotations": str(Path(args.annotations)),
        "model_name": args.model_name,
        "backend": args.backend,
        "prompt_mode": args.prompt_mode,
        "prompt_profiles": args.prompt_profiles,
        "target_regions": sorted(target_regions),
        "score_threshold": args.score_threshold,
        "num_labeled_records": len(manual_records),
        "profiles": {
            profile: {
                **profile_summary(records),
                "records": records,
            }
            for profile, records in profile_records.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {
        **{key: value for key, value in output.items() if key != "profiles"},
        "profiles": {
            profile: {key: value for key, value in payload.items() if key != "records"}
            for profile, payload in output["profiles"].items()
        },
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
