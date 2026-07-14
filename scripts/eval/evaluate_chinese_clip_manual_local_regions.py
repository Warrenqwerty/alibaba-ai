from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.instance_segmentation import FashionInstanceSegmentationPredictor
from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import generate_open_vocab_candidates
from fashion_mm.models.local_region import parse_region_query
from fashion_mm.models.local_region import select_garment_instance
from fashion_mm.utils.config import load_config
from scripts.eval.evaluate_chinese_clip_local_region_ranker import candidate_matches_parsed_region
from scripts.eval.evaluate_chinese_clip_local_region_ranker import crop_candidate
from scripts.eval.evaluate_chinese_clip_local_region_ranker import encode_images
from scripts.eval.evaluate_chinese_clip_local_region_ranker import encode_text
from scripts.eval.evaluate_chinese_clip_local_region_ranker import load_chinese_clip
from scripts.eval.evaluate_chinese_clip_local_region_ranker import parse_region_prior_weights
from scripts.eval.evaluate_local_region_manual_labels import load_manual_records
from scripts.eval.evaluate_local_region_manual_labels import summarize_records


DEFAULT_MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Chinese-CLIP crop reranking against manual 3.1.2 "
            "bbox labels. This is an offline pretrained baseline, not training."
        )
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument(
        "--region-prior-weights",
        default="0.0,0.05,0.1,0.2",
        help=(
            "Comma-separated weights added to candidates matching the parsed "
            "region. Keep this small; the visual-text score remains primary."
        ),
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/local_region_manual_eval_chinese_clip.json",
    )
    return parser.parse_args()


def score_manual_record(
    manual_record: dict[str, Any],
    *,
    segmentation: Any,
    image: Image.Image,
    model: Any,
    processor: Any,
    device: torch.device,
    image_batch_size: int,
    region_prior_weights: tuple[float, ...],
) -> dict[float, dict[str, Any]]:
    """Score all 3.1.1-constrained local crops for one manual query."""
    parsed_query = parse_region_query(manual_record["query_text"])
    selected_instance = select_garment_instance(segmentation, parsed_query)
    if selected_instance is None:
        return {
            weight: empty_prediction_record(
                manual_record,
                parsed_query.region,
                "no_garment_instance",
            )
            for weight in region_prior_weights
        }
    candidates = generate_open_vocab_candidates(
        selected_instance.mask,
        selected_instance.box,
        category_text=selected_instance.label_name,
    )
    candidates = [candidate for candidate in candidates if candidate.box is not None]
    if not candidates:
        return {
            weight: empty_prediction_record(
                manual_record,
                parsed_query.region,
                "no_region_candidate",
                selected_instance=selected_instance,
            )
            for weight in region_prior_weights
        }

    start = time.perf_counter()
    text_features = encode_text(manual_record["query_text"], model, processor, device)
    crops = [crop_candidate(image, candidate.box) for candidate in candidates]
    image_features = encode_images(
        crops,
        model,
        processor,
        device,
        image_batch_size=image_batch_size,
    )
    clip_scores = (image_features @ text_features.T).squeeze(1).detach().cpu()
    prior_scores = torch.tensor(
        [
            1.0 if candidate_matches_parsed_region(candidate.region, parsed_query.region) else 0.0
            for candidate in candidates
        ],
        dtype=torch.float32,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    results: dict[float, dict[str, Any]] = {}
    for weight in region_prior_weights:
        blended_scores = clip_scores + float(weight) * prior_scores
        best_index = int(torch.argmax(blended_scores).item())
        candidate = candidates[best_index]
        predicted_box = candidate.box
        results[weight] = {
            "id": manual_record.get("id"),
            "image": str(manual_record["image"]),
            "query_text": manual_record["query_text"],
            "target_region": manual_record.get("target_region"),
            "target_bbox": list(manual_record["target_bbox"]),
            "status": "ok",
            "ranker_backend": "chinese_clip_crop_reranker",
            "selected_region": candidate.region,
            "predicted_bbox": list(predicted_box) if predicted_box is not None else None,
            "manual_bbox_iou": box_iou(predicted_box, manual_record["target_bbox"]),
            "local_region_latency_ms": latency_ms,
            "clip_score": float(clip_scores[best_index].item()),
            "region_prior": float(prior_scores[best_index].item()),
            "region_prior_weight": float(weight),
            "match_score": float(blended_scores[best_index].item()),
            "selected_instance": selected_instance.to_dict(include_mask=False),
        }
    return results


def empty_prediction_record(
    manual_record: dict[str, Any],
    parsed_region: str | None,
    status: str,
    *,
    selected_instance: Any = None,
) -> dict[str, Any]:
    return {
        "id": manual_record.get("id"),
        "image": str(manual_record["image"]),
        "query_text": manual_record["query_text"],
        "target_region": manual_record.get("target_region"),
        "target_bbox": list(manual_record["target_bbox"]),
        "status": status,
        "ranker_backend": "chinese_clip_crop_reranker",
        "selected_region": parsed_region,
        "predicted_bbox": None,
        "manual_bbox_iou": 0.0,
        "selected_instance": (
            selected_instance.to_dict(include_mask=False)
            if selected_instance is not None
            else None
        ),
    }


def evaluate_chinese_clip_manual_records(
    manual_records: list[dict[str, Any]],
    *,
    model_config: str,
    checkpoint: str,
    model_name: str,
    device: str | None,
    image_batch_size: int,
    region_prior_weights: tuple[float, ...],
) -> dict[float, list[dict[str, Any]]]:
    """Evaluate frozen Chinese-CLIP on a single manual benchmark split."""
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor = load_chinese_clip(model_name, torch_device)
    predictor = FashionInstanceSegmentationPredictor(
        config=load_config(model_config),
        checkpoint_path=checkpoint,
        device=torch_device,
    )
    records_by_weight = {weight: [] for weight in region_prior_weights}
    segmentation_cache: dict[str, Any] = {}
    image_cache: dict[str, Image.Image] = {}
    with torch.no_grad():
        for manual_record in manual_records:
            image_path = str(manual_record["image"])
            if image_path not in segmentation_cache:
                segmentation_cache[image_path] = predictor.predict(image_path)
            if image_path not in image_cache:
                image_cache[image_path] = Image.open(image_path).convert("RGB")
            scored = score_manual_record(
                manual_record,
                segmentation=segmentation_cache[image_path],
                image=image_cache[image_path],
                model=model,
                processor=processor,
                device=torch_device,
                image_batch_size=image_batch_size,
                region_prior_weights=region_prior_weights,
            )
            for weight, record in scored.items():
                records_by_weight[weight].append(record)
    return records_by_weight


def finalize_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_records(records)
    summary["status_counts"] = dict(Counter(record["status"] for record in records))
    summary["avg_local_region_latency_ms"] = mean(
        float(record.get("local_region_latency_ms") or 0.0) for record in records
    )
    return summary


def format_weight(weight: float) -> str:
    return f"{weight:g}"


def main() -> None:
    args = parse_args()
    manual_records = load_manual_records(args.annotations, max_records=args.max_records)
    if not manual_records:
        raise ValueError("No labeled manual records found")
    region_prior_weights = parse_region_prior_weights(args.region_prior_weights)
    records_by_weight = evaluate_chinese_clip_manual_records(
        manual_records,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        device=args.device,
        image_batch_size=args.image_batch_size,
        region_prior_weights=region_prior_weights,
    )
    output = {
        "annotations": str(Path(args.annotations)),
        "model_name": args.model_name,
        "region_prior_weights": list(region_prior_weights),
        "num_labeled_records": len(manual_records),
        "runs": {
            format_weight(weight): {
                **finalize_run(records),
                "records": records,
            }
            for weight, records in records_by_weight.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                **{key: value for key, value in output.items() if key != "runs"},
                "runs": {
                    weight: {key: value for key, value in run.items() if key != "records"}
                    for weight, run in output["runs"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
