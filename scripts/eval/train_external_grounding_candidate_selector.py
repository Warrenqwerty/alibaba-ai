from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.cross_validate_grounding_candidate_selector import (
    CANDIDATE_FEATURE_SCHEMA,
    ManualCandidateSelector,
)
from scripts.eval.cross_validate_grounding_candidate_selector import candidate_examples
from scripts.eval.cross_validate_grounding_candidate_selector import (
    choose_nested_region_policies,
)
from scripts.eval.cross_validate_grounding_candidate_selector import image_grouped_folds
from scripts.eval.cross_validate_grounding_candidate_selector import (
    keep_current_candidate_record,
)
from scripts.eval.cross_validate_grounding_candidate_selector import (
    parse_threshold_grid,
)
from scripts.eval.cross_validate_grounding_candidate_selector import (
    record_has_complete_dinov2_embeddings,
)
from scripts.eval.cross_validate_grounding_candidate_selector import (
    select_conservative_candidate_record,
)
from scripts.eval.cross_validate_grounding_candidate_selector import (
    selector_diagnostics,
)
from scripts.eval.cross_validate_grounding_candidate_selector import (
    train_conservative_selector,
)
from scripts.eval.evaluate_local_region_manual_labels import summarize_records


DEFAULT_REGIONS = ("cuff", "waist")
DINO_COMPATIBILITY_FIELDS = (
    "model_name",
    "context_scale",
    "projection_dim",
    "projection_seed",
    "projection_fingerprint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a conservative candidate selector on independent online "
            "DeepFashion2 weak records, then evaluate once on a frozen manual set."
        )
    )
    parser.add_argument("--train-eval-json", required=True)
    parser.add_argument("--test-eval-json", required=True)
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--selector-architecture",
        choices=("mlp", "linear"),
        default="linear",
    )
    parser.add_argument(
        "--thresholds",
        default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    parser.add_argument("--max-calibration-lost-hits", type=int, default=0)
    parser.add_argument("--min-calibration-net-gain", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_eval_payload(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {path}")
    return payload, records


def validate_external_training_payload(payload: dict[str, Any]) -> None:
    if payload.get("supervision_type") != "landmark_pseudo_label_only":
        raise ValueError(
            "External selector training requires landmark-only weak supervision."
        )
    if payload.get("candidate_generation_uses_target_bbox") is not False:
        raise ValueError(
            "Training payload must state candidate_generation_uses_target_bbox=false."
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("External selector training payload has no records.")
    invalid = [
        record.get("id")
        for record in records
        if record.get("weak_label_source") != "landmark_pseudo_label"
        or record.get("evaluation_target") != "landmark_pseudo_label_only"
    ]
    if invalid:
        raise ValueError(
            "External selector training records must all use landmark-only "
            f"targets; first invalid ids: {invalid[:3]}"
        )


def validate_dinov2_enrichment_compatibility(
    train_payload: dict[str, Any],
    test_payload: dict[str, Any],
) -> None:
    train_metadata = train_payload.get("dinov2_candidate_enrichment")
    test_metadata = test_payload.get("dinov2_candidate_enrichment")
    if train_metadata is None and test_metadata is None:
        return
    if not isinstance(train_metadata, dict) or not isinstance(test_metadata, dict):
        raise ValueError(
            "DINOv2 candidate enrichment must be present on both train and test."
        )
    for label, metadata in (("train", train_metadata), ("test", test_metadata)):
        if metadata.get("target_bbox_used_for_features") is not False:
            raise ValueError(
                f"{label} DINOv2 enrichment must state "
                "target_bbox_used_for_features=false."
            )
        if not metadata.get("projection_fingerprint"):
            raise ValueError(f"{label} DINOv2 enrichment has no projection fingerprint.")
    mismatches = [
        field
        for field in DINO_COMPATIBILITY_FIELDS
        if train_metadata.get(field) != test_metadata.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Train/test DINOv2 enrichment settings differ: "
            + ", ".join(mismatches)
        )


def validate_dinov2_record_coverage(
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    if payload.get("dinov2_candidate_enrichment") is None:
        return
    incomplete = [
        record.get("id")
        for record in records
        if not record_has_complete_dinov2_embeddings(record)
    ]
    if incomplete:
        raise ValueError(
            f"{label} contains records without complete DINOv2 candidate "
            f"embeddings; first ids: {incomplete[:3]}"
        )


def selected_record_indices(
    records: list[dict[str, Any]],
    regions: set[str],
) -> list[int]:
    return [
        index
        for index, record in enumerate(records)
        if str(record.get("target_region") or "") in regions
    ]


def image_sizes_for_records(
    records: list[dict[str, Any]],
) -> dict[str, tuple[int, int]]:
    sizes = {}
    for record in records:
        image_path = str(record["image"])
        if image_path not in sizes:
            with Image.open(image_path) as image:
                sizes[image_path] = image.size
    return sizes


def build_examples(
    records: list[dict[str, Any]],
) -> list[tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]]:
    sizes = image_sizes_for_records(records)
    return [
        candidate_examples(record, sizes[str(record["image"])]) for record in records
    ]


def calibration_rows(
    records: list[dict[str, Any]],
    examples: list[tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    indices: list[int],
    model: ManualCandidateSelector,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for index in indices:
        alternative = select_conservative_candidate_record(
            records[index],
            examples[index],
            model,
            device,
            override_threshold=0.0,
        )
        if not alternative.get("selector_overrode_current"):
            continue
        rows.append(
            {
                "target_region": str(records[index].get("target_region") or ""),
                "probability": float(
                    alternative.get("selector_override_probability") or 0.0
                ),
                "current_iou": float(records[index].get("manual_bbox_iou") or 0.0),
                "alternative_iou": float(alternative.get("manual_bbox_iou") or 0.0),
            }
        )
    return rows


def select_with_external_policy(
    record: dict[str, Any],
    example: tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]],
    model: ManualCandidateSelector,
    device: torch.device,
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if policy is None or not policy["enabled"]:
        selected = keep_current_candidate_record(record, example)
    else:
        selected = select_conservative_candidate_record(
            record,
            example,
            model,
            device,
            override_threshold=float(policy["threshold"]),
        )
    selected["selector_region_enabled"] = bool(policy and policy["enabled"])
    selected["selector_region_threshold"] = (
        policy.get("threshold") if policy is not None else None
    )
    selected["selector_training_source"] = "independent_deepfashion2_train_weak_labels"
    return selected


def ensure_disjoint_images(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:
    train_images = {str(record["image"]) for record in train_records}
    test_images = {str(record["image"]) for record in test_records}
    overlap = train_images & test_images
    if overlap:
        raise ValueError(
            "External training and frozen test images overlap; first examples: "
            + ", ".join(sorted(overlap)[:3])
        )


def save_selector_checkpoint(
    path: str | Path,
    model: ManualCandidateSelector,
    *,
    policies: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    input_dim: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "selector_architecture": args.selector_architecture,
            "selection_policy": "conservative_pairwise",
            "region_policies": policies,
            "regions": args.regions,
            "seed": args.seed,
            "candidate_feature_schema": CANDIDATE_FEATURE_SCHEMA,
            "training_source": "independent_deepfashion2_train_weak_labels",
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    if args.calibration_folds < 2:
        raise ValueError("calibration_folds must be at least 2")
    if args.max_calibration_lost_hits < 0:
        raise ValueError("max_calibration_lost_hits cannot be negative")
    if args.min_calibration_net_gain < 1:
        raise ValueError("min_calibration_net_gain must be positive")
    thresholds = parse_threshold_grid(args.thresholds)
    regions = set(args.regions)
    train_payload, all_train_records = load_eval_payload(args.train_eval_json)
    test_payload, all_test_records = load_eval_payload(args.test_eval_json)
    validate_external_training_payload(train_payload)
    validate_dinov2_enrichment_compatibility(train_payload, test_payload)
    train_indices_in_payload = selected_record_indices(all_train_records, regions)
    test_indices_in_payload = selected_record_indices(all_test_records, regions)
    train_records = [all_train_records[index] for index in train_indices_in_payload]
    test_records = [all_test_records[index] for index in test_indices_in_payload]
    if not train_records:
        raise ValueError("No requested regions found in external training records")
    if not test_records:
        raise ValueError("No requested regions found in frozen test records")
    validate_dinov2_record_coverage(train_payload, train_records, label="train")
    validate_dinov2_record_coverage(test_payload, test_records, label="test")
    ensure_disjoint_images(train_records, test_records)

    train_examples = build_examples(train_records)
    folds = image_grouped_folds(
        train_records,
        num_folds=args.calibration_folds,
        seed=args.seed,
    )
    calibration_indices = folds[0]
    fit_indices = sorted(set(range(len(train_records))) - set(calibration_indices))
    device = torch.device(args.device)
    model = train_conservative_selector(
        train_examples,
        fit_indices,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
        architecture=args.selector_architecture,
    )
    rows = calibration_rows(
        train_records,
        train_examples,
        calibration_indices,
        model,
        device,
    )
    policies = choose_nested_region_policies(
        rows,
        regions=regions,
        thresholds=thresholds,
        max_lost_hits=args.max_calibration_lost_hits,
        min_net_gain=args.min_calibration_net_gain,
    )
    # The frozen benchmark is featurized only after training and all thresholds
    # have been fixed from the independent weak calibration split.
    test_examples = build_examples(test_records)
    selected_test_records = [
        select_with_external_policy(
            record,
            example,
            model,
            device,
            policies.get(str(record.get("target_region") or "")),
        )
        for record, example in zip(test_records, test_examples, strict=True)
    ]
    full_test_records = [dict(record) for record in all_test_records]
    for payload_index, selected in zip(
        test_indices_in_payload,
        selected_test_records,
        strict=True,
    ):
        full_test_records[payload_index] = selected

    input_dim = train_examples[fit_indices[0]][0].shape[1] * 3
    if args.model_output:
        save_selector_checkpoint(
            args.model_output,
            model,
            policies=policies,
            args=args,
            input_dim=input_dim,
        )
    result = {
        "train_eval_json": str(Path(args.train_eval_json)),
        "frozen_test_eval_json": str(Path(args.test_eval_json)),
        "training_supervision": "landmark_pseudo_label_only",
        "test_labels_used_for_training_or_calibration": False,
        "train_test_image_overlap": 0,
        "regions": args.regions,
        "selector_architecture": args.selector_architecture,
        "selection_policy": "conservative_pairwise",
        "num_epochs": args.num_epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "candidate_feature_schema": CANDIDATE_FEATURE_SCHEMA,
        "calibration_policy": "image_grouped_first_fold",
        "calibration_folds": args.calibration_folds,
        "num_fit_records": len(fit_indices),
        "num_calibration_records": len(calibration_indices),
        "calibration_thresholds": list(thresholds),
        "calibration_region_policies": policies,
        "calibration_region_counts": dict(
            Counter(row["target_region"] for row in rows)
        ),
        "num_train_records_with_visual_scores": sum(
            bool(record.get("visual_candidate_scores")) for record in train_records
        ),
        "num_test_records_with_visual_scores": sum(
            bool(record.get("visual_candidate_scores")) for record in test_records
        ),
        "num_train_records_with_dinov2_embeddings": sum(
            record_has_complete_dinov2_embeddings(record)
            for record in train_records
        ),
        "num_test_records_with_dinov2_embeddings": sum(
            record_has_complete_dinov2_embeddings(record)
            for record in test_records
        ),
        "weak_train_baseline_summary": summarize_records(train_records),
        "frozen_test_baseline_summary": summarize_records(all_test_records),
        "frozen_test_summary": summarize_records(full_test_records),
        "frozen_test_selector_diagnostics": selector_diagnostics(
            test_records,
            selected_test_records,
        ),
        "model_output": str(Path(args.model_output)) if args.model_output else None,
        "visual_candidate_enrichment": {
            "train": train_payload.get("visual_candidate_enrichment"),
            "test": test_payload.get("visual_candidate_enrichment"),
        },
        "dinov2_candidate_enrichment": {
            "train": train_payload.get("dinov2_candidate_enrichment"),
            "test": test_payload.get("dinov2_candidate_enrichment"),
        },
        "records": full_test_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
