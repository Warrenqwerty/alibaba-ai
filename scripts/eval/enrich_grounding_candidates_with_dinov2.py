from __future__ import annotations

import argparse
import hashlib
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

from scripts.eval.cross_validate_grounding_candidate_selector import candidate_key
from scripts.eval.cross_validate_grounding_candidate_selector import (
    DINO_PROJECTION_DIM,
)
from scripts.eval.cross_validate_grounding_candidate_selector import selector_candidates
from scripts.eval.cross_validate_grounding_candidate_selector import (
    visual_scores_by_box,
)
from scripts.eval.enrich_grounding_candidates_with_chinese_clip import contextual_box
from scripts.eval.evaluate_chinese_clip_local_region_ranker import crop_candidate


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "facebook/dinov2-base"
DEFAULT_REGIONS = ("cuff", "pocket", "pattern", "waist", "zipper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach frozen DINOv2 tight/context crop embeddings to grounding "
            "candidates. Target boxes and IoU labels are never read."
        )
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--context-scale", type=float, default=1.6)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_dinov2(model_name: str, device: torch.device) -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError(
            "DINOv2 enrichment requires transformers. Install it with "
            "`pip install 'transformers>=4.37.0'`."
        ) from error

    LOGGER.info("Loading DINOv2 model: %s", model_name)
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except OSError as error:
        raise RuntimeError(
            "Could not load DINOv2 model files. On AutoDL, rerun with "
            "`HF_ENDPOINT=https://hf-mirror.com` or pass a local model path."
        ) from error
    model.eval()
    return model, processor


def dinov2_feature_tensor(output: Any) -> torch.Tensor:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first[:, 0] if first.ndim == 3 else first
    raise TypeError(f"Unsupported DINOv2 feature output type: {type(output)!r}")


def encode_dinov2_images(
    crops: list[Image.Image],
    model: Any,
    processor: Any,
    device: torch.device,
    *,
    image_batch_size: int,
) -> torch.Tensor:
    if image_batch_size <= 0:
        raise ValueError("image_batch_size must be positive")
    features = []
    for start in range(0, len(crops), image_batch_size):
        inputs = processor(
            images=crops[start : start + image_batch_size],
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model(**inputs)
        features.append(
            torch.nn.functional.normalize(
                dinov2_feature_tensor(output).float(),
                dim=-1,
            )
        )
    return torch.cat(features, dim=0)


def deterministic_projection(
    input_dim: int,
    *,
    output_dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(
        input_dim,
        output_dim,
        generator=generator,
        dtype=torch.float32,
    )
    projection = torch.nn.functional.normalize(projection, dim=0)
    return projection.to(device)


def projection_fingerprint(projection: torch.Tensor) -> str:
    values = (
        projection.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(values).hexdigest()


def project_dinov2_features(
    features: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    if features.shape[-1] != projection.shape[0]:
        raise ValueError(
            "DINOv2 feature/projection mismatch: "
            f"{features.shape[-1]} vs {projection.shape[0]}"
        )
    return torch.nn.functional.normalize(features @ projection, dim=-1)


def score_record_candidates(
    record: dict[str, Any],
    image: Image.Image,
    *,
    model: Any,
    processor: Any,
    projection: torch.Tensor,
    device: torch.device,
    context_scale: float,
    image_batch_size: int,
) -> list[dict[str, Any]]:
    candidates = selector_candidates(record)
    if not candidates:
        return []
    tight_crops = [crop_candidate(image, candidate["bbox"]) for candidate in candidates]
    context_crops = [
        crop_candidate(
            image,
            contextual_box(candidate["bbox"], image.size, context_scale),
        )
        for candidate in candidates
    ]
    full_features = encode_dinov2_images(
        tight_crops + context_crops,
        model,
        processor,
        device,
        image_batch_size=image_batch_size,
    )
    projected = project_dinov2_features(full_features, projection).cpu()
    num_candidates = len(candidates)
    tight_features = projected[:num_candidates]
    context_features = projected[num_candidates:]
    similarities = (tight_features * context_features).sum(dim=1)
    previous_rows = visual_scores_by_box(record)
    scored = []
    for index, candidate in enumerate(candidates):
        box = [float(value) for value in candidate["bbox"]]
        row = dict(previous_rows.get(candidate_key(box), {}))
        row.update(
            {
                "bbox": box,
                "candidate_source": candidate.get("candidate_source"),
                "candidate_rank": candidate.get("candidate_rank"),
                "prompt": candidate.get("prompt"),
                "dinov2_tight_embedding": tight_features[index].tolist(),
                "dinov2_context_embedding": context_features[index].tolist(),
                "dinov2_tight_context_similarity": float(similarities[index].item()),
            }
        )
        scored.append(row)
    return scored


def enrich_records(
    records: list[dict[str, Any]],
    *,
    regions: set[str],
    model: Any,
    processor: Any,
    projection: torch.Tensor,
    device: torch.device,
    context_scale: float,
    image_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    enriched = []
    num_scored_records = 0
    num_scored_candidates = 0
    cached_path: str | None = None
    cached_image: Image.Image | None = None
    with torch.inference_mode():
        for record in records:
            updated = dict(record)
            if str(record.get("target_region") or "") in regions:
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
                    projection=projection,
                    device=device,
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
    model, processor = load_dinov2(args.model_name, device)
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    projection = deterministic_projection(
        hidden_size,
        output_dim=DINO_PROJECTION_DIM,
        seed=args.projection_seed,
        device=device,
    )
    enriched_records, counts = enrich_records(
        records,
        regions=set(args.regions),
        model=model,
        processor=processor,
        projection=projection,
        device=device,
        context_scale=args.context_scale,
        image_batch_size=args.image_batch_size,
    )
    metadata = {
        "source_eval_json": str(Path(args.eval_json)),
        "model_name": args.model_name,
        "regions": args.regions,
        "context_scale": args.context_scale,
        "projection_dim": DINO_PROJECTION_DIM,
        "projection_seed": args.projection_seed,
        "projection_fingerprint": projection_fingerprint(projection),
        "target_bbox_used_for_features": False,
        **counts,
    }
    output = {
        **{key: value for key, value in payload.items() if key != "records"},
        "dinov2_candidate_enrichment": metadata,
        "records": enriched_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
