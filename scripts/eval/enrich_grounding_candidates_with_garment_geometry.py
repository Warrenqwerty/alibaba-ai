from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.instance_segmentation import (
    FashionInstanceSegmentationPredictor,
)
from fashion_mm.models.local_region import parse_region_query
from fashion_mm.models.local_region import select_garment_instance
from fashion_mm.utils.config import load_config


LOGGER = logging.getLogger(__name__)
DEFAULT_REGIONS = ("cuff", "waist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach online predicted-garment boxes to saved local-region "
            "candidate records. Target boxes are never used to create features."
        )
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument(
        "--model-config",
        default="configs/model/instance_segmentation_deepfashion2.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def enrich_records(
    records: list[dict[str, Any]],
    *,
    predictor: FashionInstanceSegmentationPredictor,
    regions: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    enriched: list[dict[str, Any]] = []
    cached_image_path: str | None = None
    cached_segmentation: Any = None
    processed_images: set[str] = set()
    num_segmentation_inferences = 0
    num_enriched_records = 0
    num_missing_instances = 0

    for record in records:
        updated = dict(record)
        region = str(record.get("target_region") or "")
        if region not in regions:
            enriched.append(updated)
            continue

        image_path = str(record["image"])
        if image_path != cached_image_path:
            cached_segmentation = predictor.predict(image_path)
            cached_image_path = image_path
            processed_images.add(image_path)
            num_segmentation_inferences += 1
        instance = select_garment_instance(
            cached_segmentation,
            parse_region_query(str(record.get("query_text") or "")),
        )
        updated["online_garment_instance"] = (
            instance.to_dict(include_mask=False) if instance is not None else None
        )
        if instance is None:
            num_missing_instances += 1
        else:
            num_enriched_records += 1
        enriched.append(updated)

        processed = num_enriched_records + num_missing_instances
        if processed % 100 == 0:
            LOGGER.info(
                "processed_records=%s segmentation_inferences=%s missing=%s",
                processed,
                num_segmentation_inferences,
                num_missing_instances,
            )

    return enriched, {
        "num_scored_records": num_enriched_records + num_missing_instances,
        "num_records_with_online_garment_instance": num_enriched_records,
        "num_records_without_online_garment_instance": num_missing_instances,
        "num_unique_images": len(processed_images),
        "num_segmentation_inferences": num_segmentation_inferences,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {args.eval_json}")

    predictor = FashionInstanceSegmentationPredictor(
        config=load_config(args.model_config),
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    enriched_records, counts = enrich_records(
        records,
        predictor=predictor,
        regions=set(args.regions),
    )
    metadata = {
        "source_eval_json": str(Path(args.eval_json)),
        "model_config": args.model_config,
        "checkpoint": args.checkpoint,
        "regions": args.regions,
        "selection_method": "online_segmentation_select_garment_instance",
        "target_bbox_used_for_features": False,
        **counts,
    }
    output = {
        **{key: value for key, value in payload.items() if key != "records"},
        "online_garment_geometry_enrichment": metadata,
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
