import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.build_deepfashion2_local_region_queries import (
    QUERY_TEMPLATES,
    build_records_for_annotation,
    export_record_visualizations,
)
from scripts.data.build_online_local_region_weak_candidates import (
    merge_weak_metadata,
    sample_weak_query_records,
    validate_weak_group,
    weak_query_to_eval_record,
)
from scripts.eval.train_external_grounding_candidate_selector import (
    ensure_disjoint_images,
    main as selector_main,
    validate_dinov2_enrichment_compatibility,
    validate_dinov2_record_coverage,
    validate_external_training_payload,
    validate_online_garment_geometry_compatibility,
    validate_online_garment_geometry_coverage,
)


def raw_landmarks(
    count: int,
    points: dict[int, tuple[int, int, int]],
) -> list[int]:
    values = [0, 0, 0] * count
    for index, (x, y, visibility) in points.items():
        offset = (index - 1) * 3
        values[offset : offset + 3] = [x, y, visibility]
    return values


def test_landmark_only_query_builder_keeps_supported_cuffs(tmp_path):
    image_path = tmp_path / "000001.jpg"
    annotation_path = tmp_path / "000001.json"
    Image.new("RGB", (120, 120), color="white").save(image_path)
    annotation = {
        "item1": {
            "category_id": 1,
            "category_name": "short_sleeved_shirt",
            "bounding_box": [10, 10, 110, 110],
            "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
            "landmarks": raw_landmarks(
                25,
                {
                    9: (20, 50, 2),
                    10: (20, 65, 2),
                    22: (100, 50, 2),
                    23: (100, 65, 2),
                },
            ),
        }
    }

    records = build_records_for_annotation(
        image_path,
        annotation_path,
        annotation,
        ["left_cuff", "right_cuff", "waist"],
        landmark_only=True,
    )

    assert len(records) == len(QUERY_TEMPLATES["left_cuff"]) * 2
    assert {record["region"] for record in records} == {
        "left_cuff",
        "right_cuff",
    }
    assert {record["source"] for record in records} == {"landmark_pseudo_label"}
    assert {record["num_items_in_image"] for record in records} == {1}
    assert {record["side_convention"] for record in records} == {
        "garment_wearer_front_view"
    }

    vis_dir = tmp_path / "visualizations"
    assert export_record_visualizations(records, vis_dir, max_records=10) == 2
    assert len(list(vis_dir.glob("*.jpg"))) == 2


def test_online_weak_sampler_uses_one_template_and_filters_contamination(tmp_path):
    query_path = tmp_path / "queries.jsonl"
    rows = []
    for query in QUERY_TEMPLATES["left_cuff"]:
        rows.append(
            {
                "image": "/train/one.jpg",
                "item_key": "item1",
                "region": "left_cuff",
                "query": query,
                "region_box": [1, 2, 3, 4],
                "source": "landmark_pseudo_label",
                "num_items_in_image": 1,
            }
        )
    rows.append(
        {
            "image": "/train/two.jpg",
            "item_key": "item1",
            "region": "waist",
            "query": "腰线位置",
            "region_box": [1, 2, 3, 4],
            "source": "rule_baseline",
            "num_items_in_image": 1,
        }
    )
    rows.append(
        {
            "image": "/train/three.jpg",
            "item_key": "item1",
            "region": "waist",
            "query": "腰线位置",
            "region_box": [1, 2, 3, 4],
            "source": "landmark_pseudo_label",
            "num_items_in_image": 2,
        }
    )
    query_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    sampled, eligible = sample_weak_query_records(
        query_path,
        regions={"cuff", "waist"},
        max_records=None,
        seed=42,
        allow_multi_item_images=False,
    )

    assert eligible == 1
    assert len(sampled) == 1
    assert sampled[0]["region"] == "left_cuff"
    converted = weak_query_to_eval_record(sampled[0], 0)
    assert converted["target_region"] == "cuff"
    assert converted["weak_region_variant"] == "left_cuff"


def test_online_weak_group_validation_rejects_mixed_targets():
    group = [
        {
            "image": "/train/one.jpg",
            "item_key": "item1",
            "region": "waist",
            "region_box": [1, 2, 3, 4],
            "source": "landmark_pseudo_label",
        },
        {
            "image": "/train/one.jpg",
            "item_key": "item1",
            "region": "waist",
            "region_box": [5, 6, 7, 8],
            "source": "landmark_pseudo_label",
        },
    ]

    with pytest.raises(ValueError, match="region_box"):
        validate_weak_group(group)


def test_weak_metadata_merge_marks_landmark_evaluation_target():
    weak_record = {
        "id": "weak-1",
        "image": "/train/one.jpg",
        "weak_label_source": "landmark_pseudo_label",
    }
    merged = merge_weak_metadata(
        [{"id": "weak-1", "manual_bbox_iou": 0.4}],
        [weak_record],
    )

    assert merged[0]["weak_bbox_iou"] == 0.4
    assert merged[0]["evaluation_target"] == "landmark_pseudo_label_only"


def test_external_training_payload_rejects_manual_or_rule_targets():
    valid = {
        "supervision_type": "landmark_pseudo_label_only",
        "candidate_generation_uses_target_bbox": False,
        "records": [
            {
                "id": "weak-1",
                "weak_label_source": "landmark_pseudo_label",
                "evaluation_target": "landmark_pseudo_label_only",
            }
        ],
    }
    validate_external_training_payload(valid)

    invalid = {
        **valid,
        "records": [{**valid["records"][0], "weak_label_source": "rule_baseline"}],
    }
    with pytest.raises(ValueError, match="landmark-only"):
        validate_external_training_payload(invalid)


def test_external_training_and_frozen_test_images_must_be_disjoint():
    with pytest.raises(ValueError, match="overlap"):
        ensure_disjoint_images(
            [{"image": "/shared/one.jpg"}],
            [{"image": "/shared/one.jpg"}],
        )


def dinov2_metadata(fingerprint="projection-a"):
    return {
        "model_name": "facebook/dinov2-base",
        "context_scale": 1.6,
        "projection_dim": 64,
        "projection_seed": 42,
        "projection_fingerprint": fingerprint,
        "target_bbox_used_for_features": False,
    }


def dinov2_spatial_metadata(fingerprint="spatial-projection-a"):
    return {
        "model_name": "facebook/dinov2-base",
        "feature_mode": "spatial_pyramid",
        "regions": ["cuff"],
        "context_scale": 1.6,
        "projection_dim": 128,
        "projection_seed": 42,
        "projection_fingerprint": fingerprint,
        "spatial_components": [
            "cls",
            "patch_mean",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
            "center",
            "border",
        ],
        "target_bbox_used_for_features": False,
    }


def garment_geometry_metadata(checkpoint="checkpoint.pt"):
    return {
        "model_config": "configs/model/instance_segmentation_deepfashion2.yaml",
        "checkpoint": checkpoint,
        "regions": ["cuff", "waist"],
        "selection_method": "online_segmentation_select_garment_instance",
        "target_bbox_used_for_features": False,
    }


def test_external_selector_requires_matching_dinov2_projection_metadata():
    metadata = dinov2_metadata()
    validate_dinov2_enrichment_compatibility(
        {"dinov2_candidate_enrichment": metadata},
        {"dinov2_candidate_enrichment": dict(metadata)},
    )
    validate_dinov2_enrichment_compatibility(
        {"dinov2_candidate_enrichment": metadata},
        {"dinov2_candidate_enrichment": {**metadata, "feature_mode": "global"}},
    )
    validate_dinov2_enrichment_compatibility({}, {})

    with pytest.raises(ValueError, match="both train and test"):
        validate_dinov2_enrichment_compatibility(
            {"dinov2_candidate_enrichment": metadata},
            {},
        )
    with pytest.raises(ValueError, match="projection_fingerprint"):
        validate_dinov2_enrichment_compatibility(
            {"dinov2_candidate_enrichment": metadata},
            {"dinov2_candidate_enrichment": dinov2_metadata("projection-b")},
        )

    spatial = dinov2_spatial_metadata()
    validate_dinov2_enrichment_compatibility(
        {"dinov2_spatial_candidate_enrichment": spatial},
        {"dinov2_spatial_candidate_enrichment": dict(spatial)},
    )
    with pytest.raises(ValueError, match="projection_fingerprint"):
        validate_dinov2_enrichment_compatibility(
            {"dinov2_spatial_candidate_enrichment": spatial},
            {
                "dinov2_spatial_candidate_enrichment": (
                    dinov2_spatial_metadata("spatial-projection-b")
                )
            },
        )
    different_regions = {**spatial, "regions": ["waist"]}
    with pytest.raises(ValueError, match="regions"):
        validate_dinov2_enrichment_compatibility(
            {"dinov2_spatial_candidate_enrichment": spatial},
            {"dinov2_spatial_candidate_enrichment": different_regions},
        )


def candidate_record(image_path, record_id, *, weak):
    record = {
        "id": record_id,
        "image": str(image_path),
        "query_text": "左边的袖口",
        "target_region": "cuff",
        "target_bbox": [0, 0, 10, 10],
        "predicted_bbox": [20, 0, 30, 10],
        "selected_region": "left sleeve cuff",
        "manual_bbox_iou": 0.0,
        "status": "ok",
        "ranker_backend": "gated_hybrid_grounding_auto",
        "score": 0.8,
        "gated_policy_route": "grounding",
        "grounding_model_name": "google/owlv2-large-patch14-ensemble",
        "diagnostic_grounding_candidate": {
            "grounding_model_name": "IDEA-Research/grounding-dino-base",
            "detections": [
                {
                    "bbox": [0, 0, 10, 10],
                    "score": 0.7,
                    "prompt": "sleeve cuff",
                }
            ],
        },
    }
    if weak:
        record.update(
            {
                "weak_label_source": "landmark_pseudo_label",
                "evaluation_target": "landmark_pseudo_label_only",
            }
        )
    return record


def test_external_selector_rejects_partial_dinov2_candidate_coverage(tmp_path):
    image_path = tmp_path / "train.jpg"
    record = candidate_record(image_path, "train-1", weak=True)
    embedding = [0.0] * 64
    record["visual_candidate_scores"] = [
        {
            "bbox": [20, 0, 30, 10],
            "dinov2_tight_embedding": embedding,
            "dinov2_context_embedding": embedding,
        },
        {
            "bbox": [0, 0, 10, 10],
            "dinov2_tight_embedding": embedding,
            "dinov2_context_embedding": embedding,
        },
    ]
    payload = {"dinov2_candidate_enrichment": dinov2_metadata()}

    validate_dinov2_record_coverage(payload, [record], label="train")
    record["visual_candidate_scores"].pop()

    with pytest.raises(ValueError, match="complete DINOv2"):
        validate_dinov2_record_coverage(payload, [record], label="train")


def test_external_selector_checks_spatial_coverage_only_for_enriched_regions(
    tmp_path,
):
    image_path = tmp_path / "train.jpg"
    cuff_record = candidate_record(image_path, "cuff-1", weak=True)
    waist_record = {
        **candidate_record(image_path, "waist-1", weak=True),
        "target_region": "waist",
    }
    embedding = [0.0] * 128
    cuff_record["visual_candidate_scores"] = [
        {
            "bbox": candidate["bbox"],
            "dinov2_spatial_tight_embedding": embedding,
            "dinov2_spatial_context_embedding": embedding,
        }
        for candidate in [
            {"bbox": [20, 0, 30, 10]},
            {"bbox": [0, 0, 10, 10]},
        ]
    ]
    payload = {
        "dinov2_spatial_candidate_enrichment": dinov2_spatial_metadata(),
    }

    validate_dinov2_record_coverage(
        payload,
        [cuff_record, waist_record],
        label="train",
    )
    cuff_record["visual_candidate_scores"].pop()

    with pytest.raises(ValueError, match="complete DINOv2 spatial"):
        validate_dinov2_record_coverage(
            payload,
            [cuff_record, waist_record],
            label="train",
        )


def test_external_selector_validates_online_garment_geometry_metadata(tmp_path):
    metadata = garment_geometry_metadata()
    validate_online_garment_geometry_compatibility(
        {"online_garment_geometry_enrichment": metadata},
        {"online_garment_geometry_enrichment": dict(metadata)},
    )
    with pytest.raises(ValueError, match="checkpoint"):
        validate_online_garment_geometry_compatibility(
            {"online_garment_geometry_enrichment": metadata},
            {
                "online_garment_geometry_enrichment": (
                    garment_geometry_metadata("other.pt")
                )
            },
        )

    image_path = tmp_path / "train.jpg"
    record = candidate_record(image_path, "cuff-1", weak=True)
    payload = {"online_garment_geometry_enrichment": metadata}
    with pytest.raises(ValueError, match="without online garment geometry"):
        validate_online_garment_geometry_coverage(payload, [record], label="train")

    record["online_garment_instance"] = None
    validate_online_garment_geometry_coverage(payload, [record], label="train")


def test_external_selector_main_trains_before_frozen_test(tmp_path, monkeypatch):
    train_records = []
    for index in range(4):
        image_path = tmp_path / f"train-{index}.jpg"
        Image.new("RGB", (40, 40), color="white").save(image_path)
        train_records.append(candidate_record(image_path, f"train-{index}", weak=True))
    test_image = tmp_path / "test.jpg"
    Image.new("RGB", (40, 40), color="white").save(test_image)
    test_records = [candidate_record(test_image, "test-1", weak=False)]

    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    output_path = tmp_path / "result.json"
    train_path.write_text(
        json.dumps(
            {
                "supervision_type": "landmark_pseudo_label_only",
                "candidate_generation_uses_target_bbox": False,
                "records": train_records,
            }
        ),
        encoding="utf-8",
    )
    test_path.write_text(json.dumps({"records": test_records}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_external_grounding_candidate_selector.py",
            "--train-eval-json",
            str(train_path),
            "--test-eval-json",
            str(test_path),
            "--regions",
            "cuff",
            "--calibration-folds",
            "2",
            "--num-epochs",
            "2",
            "--selector-architecture",
            "linear",
            "--thresholds",
            "0.0",
            "--output",
            str(output_path),
        ],
    )

    selector_main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["test_labels_used_for_training_or_calibration"] is False
    assert result["train_test_image_overlap"] == 0
    assert result["calibration_region_policies"]["cuff"]["enabled"] is True
    assert result["frozen_test_summary"]["manual_hit_at"]["0.3"] == 1.0
