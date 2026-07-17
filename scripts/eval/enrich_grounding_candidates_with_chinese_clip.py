from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.cross_validate_grounding_candidate_selector import selector_candidates
from scripts.eval.evaluate_chinese_clip_local_region_ranker import crop_candidate
from scripts.eval.evaluate_chinese_clip_local_region_ranker import encode_images
from scripts.eval.evaluate_chinese_clip_local_region_ranker import encode_text
from scripts.eval.evaluate_chinese_clip_local_region_ranker import load_chinese_clip


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
DEFAULT_REGIONS = ("cuff", "pocket", "pattern", "waist", "zipper")
REGION_PROMPTS = {
    "cuff": "服装袖子末端的袖口",
    "pocket": "衣服上可以装东西的口袋",
    "pattern": "衣服表面的印花或图案",
    "waist": "衣服腰部的腰头或腰带区域",
    "zipper": "衣服上用于开合的拉链",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach frozen Chinese-CLIP tight/context crop scores to grounding "
            "candidates. Manual target boxes are never used to create features."
        )
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--prompt-profile",
        choices=("query", "region_ensemble"),
        default="region_ensemble",
    )
    parser.add_argument("--context-scale", type=float, default=1.6)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def contextual_box(
    box: list[float] | tuple[float, ...],
    image_size: tuple[int, int],
    scale: float,
) -> tuple[float, float, float, float]:
    if scale < 1.0:
        raise ValueError("context scale must be at least 1.0")
    image_width, image_height = image_size
    x1, y1, x2, y2 = [float(value) for value in box]
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    half_width = max((x2 - x1) * scale * 0.5, 0.5)
    half_height = max((y2 - y1) * scale * 0.5, 0.5)
    return (
        max(center_x - half_width, 0.0),
        max(center_y - half_height, 0.0),
        min(center_x + half_width, float(image_width)),
        min(center_y + half_height, float(image_height)),
    )


def record_text_prompts(
    record: dict[str, Any],
    prompt_profile: str,
) -> list[str]:
    query = str(record.get("query_text") or "").strip()
    if prompt_profile == "query":
        return [query]
    region = str(record.get("target_region") or "")
    canonical = REGION_PROMPTS.get(region)
    return list(dict.fromkeys(prompt for prompt in (query, canonical) if prompt))


def encode_text_ensemble(
    prompts: list[str],
    model: Any,
    processor: Any,
    device: torch.device,
) -> torch.Tensor:
    if not prompts:
        raise ValueError("At least one visual-text prompt is required")
    features = torch.cat(
        [encode_text(prompt, model, processor, device) for prompt in prompts],
        dim=0,
    )
    return torch.nn.functional.normalize(features.mean(dim=0, keepdim=True), dim=-1)


def relative_rank_scores(scores: torch.Tensor) -> list[float]:
    values = [float(value) for value in scores]
    if len(values) <= 1:
        return [1.0] * len(values)
    denominator = float(len(values) - 1)
    return [
        1.0 - sum(other > value for other in values) / denominator
        for value in values
    ]


def score_record_candidates(
    record: dict[str, Any],
    image: Image.Image,
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    prompt_profile: str,
    context_scale: float,
    image_batch_size: int,
) -> list[dict[str, Any]]:
    candidates = selector_candidates(record)
    if not candidates:
        return []
    text_features = encode_text_ensemble(
        record_text_prompts(record, prompt_profile),
        model,
        processor,
        device,
    )
    tight_crops = [crop_candidate(image, candidate["bbox"]) for candidate in candidates]
    context_crops = [
        crop_candidate(
            image,
            contextual_box(candidate["bbox"], image.size, context_scale),
        )
        for candidate in candidates
    ]
    image_features = encode_images(
        tight_crops + context_crops,
        model,
        processor,
        device,
        image_batch_size=image_batch_size,
    )
    num_candidates = len(candidates)
    tight_scores = (image_features[:num_candidates] @ text_features.T).squeeze(1).cpu()
    context_scores = (image_features[num_candidates:] @ text_features.T).squeeze(1).cpu()
    tight_ranks = relative_rank_scores(tight_scores)
    context_ranks = relative_rank_scores(context_scores)
    scored = []
    for index, candidate in enumerate(candidates):
        tight_score = float(tight_scores[index].item())
        context_score = float(context_scores[index].item())
        scored.append(
            {
                "bbox": [float(value) for value in candidate["bbox"]],
                "candidate_source": candidate.get("candidate_source"),
                "candidate_rank": candidate.get("candidate_rank"),
                "prompt": candidate.get("prompt"),
                "tight_score": tight_score,
                "context_score": context_score,
                "max_score": max(tight_score, context_score),
                "mean_score": (tight_score + context_score) * 0.5,
                "tight_rank_score": tight_ranks[index],
                "context_rank_score": context_ranks[index],
            }
        )
    return scored


def enrich_records(
    records: list[dict[str, Any]],
    *,
    regions: set[str],
    model: Any,
    processor: Any,
    device: torch.device,
    prompt_profile: str,
    context_scale: float,
    image_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    enriched = []
    num_scored_records = 0
    num_scored_candidates = 0
    cached_path: str | None = None
    cached_image: Image.Image | None = None
    with torch.no_grad():
        for record in records:
            updated = dict(record)
            region = str(record.get("target_region") or "")
            if region in regions:
                image_path = str(record["image"])
                if image_path != cached_path:
                    if cached_image is not None:
                        cached_image.close()
                    cached_image = Image.open(image_path).convert("RGB")
                    cached_path = image_path
                scores = score_record_candidates(
                    record,
                    cached_image,
                    model=model,
                    processor=processor,
                    device=device,
                    prompt_profile=prompt_profile,
                    context_scale=context_scale,
                    image_batch_size=image_batch_size,
                )
                updated["visual_candidate_scores"] = scores
                num_scored_records += 1
                num_scored_candidates += len(scores)
                if num_scored_records % 20 == 0:
                    LOGGER.info(
                        "scored_records=%s scored_candidates=%s",
                        num_scored_records,
                        num_scored_candidates,
                    )
            enriched.append(updated)
    if cached_image is not None:
        cached_image.close()
    return enriched, {
        "num_scored_records": num_scored_records,
        "num_scored_candidates": num_scored_candidates,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.context_scale < 1.0:
        raise ValueError("--context-scale must be at least 1.0")
    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {args.eval_json}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor = load_chinese_clip(args.model_name, device)
    enriched_records, counts = enrich_records(
        records,
        regions=set(args.regions),
        model=model,
        processor=processor,
        device=device,
        prompt_profile=args.prompt_profile,
        context_scale=args.context_scale,
        image_batch_size=args.image_batch_size,
    )
    output = {
        **{key: value for key, value in payload.items() if key != "records"},
        "visual_candidate_enrichment": {
            "source_eval_json": str(Path(args.eval_json)),
            "model_name": args.model_name,
            "regions": args.regions,
            "prompt_profile": args.prompt_profile,
            "context_scale": args.context_scale,
            **counts,
        },
        "records": enriched_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output["visual_candidate_enrichment"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
