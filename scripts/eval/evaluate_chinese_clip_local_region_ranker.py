from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from fashion_mm.data_loaders import LocalRegionCandidateRecord
from fashion_mm.data_loaders import iter_local_region_candidate_records
from fashion_mm.models.local_region import parse_region_query
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)
DEFAULT_MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Chinese-CLIP reranking for 3.1.2 local candidates."
    )
    parser.add_argument("--candidates", required=True, help="Candidate JSONL path.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face Chinese-CLIP model name or local model directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda when available.",
    )
    parser.add_argument("--max-groups", type=int, default=2000)
    parser.add_argument("--skip-groups", type=int, default=0)
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=32,
        help="Candidate crop batch size for image encoding.",
    )
    parser.add_argument(
        "--region-prior-weights",
        default="0.0",
        help=(
            "Comma-separated weights added when a candidate region matches the "
            "region parsed from the query. Use multiple values for a sweep."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor = load_chinese_clip(args.model_name, device)
    region_prior_weights = parse_region_prior_weights(args.region_prior_weights)
    metrics = evaluate_chinese_clip_ranker(
        candidates_path=args.candidates,
        model=model,
        processor=processor,
        device=device,
        max_groups=args.max_groups,
        skip_groups=args.skip_groups,
        image_batch_size=args.image_batch_size,
        model_name=args.model_name,
        region_prior_weights=region_prior_weights,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def load_chinese_clip(model_name: str, device: torch.device):
    """Load a Hugging Face Chinese-CLIP model and processor."""
    try:
        from transformers import AutoProcessor
        from transformers import ChineseCLIPModel
    except ImportError as error:
        raise RuntimeError(
            "Chinese-CLIP evaluation requires transformers. Install it on AutoDL "
            "with: pip install 'transformers>=4.37.0' sentencepiece"
        ) from error

    LOGGER.info("Loading Chinese-CLIP model: %s", model_name)
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = ChineseCLIPModel.from_pretrained(model_name).to(device)
    except OSError as error:
        raise RuntimeError(
            "Could not load Chinese-CLIP model files. If AutoDL cannot reach "
            "huggingface.co, rerun with `HF_ENDPOINT=https://hf-mirror.com` or "
            "download the model to a local directory and pass that path via "
            "`--model-name`."
        ) from error
    model.eval()
    return model, processor


def evaluate_chinese_clip_ranker(
    *,
    candidates_path: str | Path,
    model,
    processor,
    device: torch.device,
    max_groups: int | None,
    skip_groups: int,
    image_batch_size: int,
    model_name: str,
    region_prior_weights: tuple[float, ...] = (0.0,),
) -> dict[str, Any]:
    """Rank each query's candidate crops by Chinese-CLIP cosine similarity."""
    target_region_counts: Counter[str] = Counter()
    summaries = {
        weight: _empty_summary()
        for weight in region_prior_weights
    }
    group_count = 0

    with torch.no_grad():
        for group in iter_candidate_groups(
            candidates_path,
            max_groups=max_groups,
            skip_groups=skip_groups,
        ):
            result = score_candidate_group(
                group,
                model=model,
                processor=processor,
                device=device,
                image_batch_size=image_batch_size,
            )
            group_count += 1
            target_region_counts[group[0].target_region] += 1
            for weight in region_prior_weights:
                selected = select_scored_candidate(result, weight)
                _update_summary(summaries[weight], selected)

            if group_count % 100 == 0:
                first_weight = region_prior_weights[0]
                LOGGER.info(
                    "groups=%s avg_top1_iou=%.4f",
                    group_count,
                    summaries[first_weight]["iou_sum"] / group_count,
                )

    runs = {
        _format_weight(weight): _finalize_summary(summary)
        for weight, summary in summaries.items()
    }
    output = {
        "candidates": str(candidates_path),
        "model_name": model_name,
        "num_groups": group_count,
        "region_prior_weights": list(region_prior_weights),
        "target_region_counts": dict(target_region_counts),
        "runs": runs,
    }
    if len(region_prior_weights) == 1:
        output.update(runs[_format_weight(region_prior_weights[0])])
    return output


def iter_candidate_groups(
    jsonl_path: str | Path,
    *,
    max_groups: int | None = None,
    skip_groups: int = 0,
) -> Iterator[list[LocalRegionCandidateRecord]]:
    """Stream adjacent candidate rows grouped by one query target."""
    current_key: tuple[Any, ...] | None = None
    current_group: list[LocalRegionCandidateRecord] = []
    seen_groups = 0
    yielded_groups = 0

    for record in iter_local_region_candidate_records(jsonl_path):
        key = _group_key(record)
        if current_key is None:
            current_key = key
        if key != current_key:
            if seen_groups >= skip_groups:
                yield current_group
                yielded_groups += 1
                if max_groups is not None and yielded_groups >= max_groups:
                    return
            seen_groups += 1
            current_key = key
            current_group = []
        current_group.append(record)

    if current_group and seen_groups >= skip_groups:
        yield current_group


def score_candidate_group(
    group: list[LocalRegionCandidateRecord],
    *,
    model,
    processor,
    device: torch.device,
    image_batch_size: int,
) -> dict[str, Any]:
    """Score one query's candidate crops with frozen Chinese-CLIP."""
    text_features = encode_text(group[0].query, model, processor, device)
    image = Image.open(group[0].image).convert("RGB")
    crops = [crop_candidate(image, record.candidate_box) for record in group]
    image_features = encode_images(
        crops,
        model,
        processor,
        device,
        image_batch_size=image_batch_size,
    )
    scores = image_features @ text_features.T
    base_scores = scores.squeeze(1).detach().cpu()
    parsed_region = parse_region_query(group[0].query).region
    prior_scores = torch.tensor(
        [
            1.0 if candidate_matches_parsed_region(record.candidate_region, parsed_region)
            else 0.0
            for record in group
        ],
        dtype=torch.float32,
    )
    return {
        "group": group,
        "base_scores": base_scores,
        "prior_scores": prior_scores,
        "parsed_region": parsed_region,
    }


def select_scored_candidate(
    scored_group: dict[str, Any],
    region_prior_weight: float,
) -> dict[str, Any]:
    group = scored_group["group"]
    base_scores = scored_group["base_scores"]
    prior_scores = scored_group["prior_scores"]
    blended_scores = base_scores + float(region_prior_weight) * prior_scores
    best_index = int(torch.argmax(blended_scores).detach().cpu())
    selected = group[best_index]
    return {
        "query": selected.query,
        "target_region": selected.target_region,
        "selected_region": selected.candidate_region,
        "selected_iou": selected.iou,
        "selected_score": float(blended_scores[best_index].detach().cpu()),
        "clip_score": float(base_scores[best_index].detach().cpu()),
        "region_prior": float(prior_scores[best_index].detach().cpu()),
        "parsed_region": scored_group["parsed_region"],
    }


def candidate_matches_parsed_region(
    candidate_region: str,
    parsed_region: str | None,
) -> bool:
    if parsed_region is None:
        return False
    if candidate_region == parsed_region:
        return True
    side_stripped = candidate_region.removeprefix("left_").removeprefix("right_")
    return side_stripped == parsed_region


def encode_text(query: str, model, processor, device: torch.device) -> torch.Tensor:
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    features = _as_feature_tensor(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(features, dim=-1)


def encode_images(
    crops: list[Image.Image],
    model,
    processor,
    device: torch.device,
    *,
    image_batch_size: int,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    for start in range(0, len(crops), image_batch_size):
        batch = crops[start : start + image_batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        image_features = _as_feature_tensor(model.get_image_features(**inputs))
        features.append(torch.nn.functional.normalize(image_features, dim=-1))
    return torch.cat(features, dim=0)


def _as_feature_tensor(output) -> torch.Tensor:
    """Convert Chinese-CLIP feature outputs across Transformers versions."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported feature output type: {type(output)!r}")


def parse_region_prior_weights(value: str) -> tuple[float, ...]:
    weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not weights:
        raise ValueError("At least one region prior weight is required.")
    return weights


def crop_candidate(
    image: Image.Image,
    box: tuple[float, float, float, float],
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = _clip_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (1, 1), color=(0, 0, 0))
    return image.crop((x1, y1, x2, y2))


def _clip_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(min(int(round(x1)), image_width), 0),
        max(min(int(round(y1)), image_height), 0),
        max(min(int(round(x2)), image_width), 0),
        max(min(int(round(y2)), image_height), 0),
    )


def _group_key(record: LocalRegionCandidateRecord) -> tuple[Any, ...]:
    return (
        record.image,
        record.annotation,
        record.item_key,
        record.query,
        record.target_region,
        record.target_region_box,
        record.garment_box,
    )


def _empty_summary() -> dict[str, Any]:
    return {
        "num_records": 0,
        "iou_sum": 0.0,
        "weak_hit_at": {"0.3": 0, "0.5": 0},
        "selected_region_counts": Counter(),
        "by_region": {},
    }


def _update_summary(summary: dict[str, Any], selected: dict[str, Any]) -> None:
    target_region = str(selected["target_region"])
    selected_region = str(selected["selected_region"])
    iou = float(selected["selected_iou"])
    summary["num_records"] += 1
    summary["iou_sum"] += iou
    summary["selected_region_counts"][selected_region] += 1
    for threshold in summary["weak_hit_at"]:
        summary["weak_hit_at"][threshold] += int(iou >= float(threshold))

    region_summary = summary["by_region"].setdefault(target_region, _empty_summary())
    region_summary["num_records"] += 1
    region_summary["iou_sum"] += iou
    region_summary["selected_region_counts"][selected_region] += 1
    for threshold in region_summary["weak_hit_at"]:
        region_summary["weak_hit_at"][threshold] += int(iou >= float(threshold))


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    num_records = int(summary["num_records"])
    return {
        "num_records": num_records,
        "avg_top1_iou": summary["iou_sum"] / max(num_records, 1),
        "weak_hit_at": {
            threshold: count / max(num_records, 1)
            for threshold, count in summary["weak_hit_at"].items()
        },
        "selected_region_counts": dict(summary["selected_region_counts"]),
        "by_region": {
            region: _finalize_summary(values)
            for region, values in summary["by_region"].items()
        },
    }


def _format_weight(weight: float) -> str:
    return f"{weight:g}"


if __name__ == "__main__":
    main()
