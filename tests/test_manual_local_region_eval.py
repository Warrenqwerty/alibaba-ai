import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.build_local_region_manual_eval_manifest import (
    balanced_region_records,
    build_class_aware_manifest_records,
    build_manifest_records,
    filter_existing_records,
    filter_records_by_target_regions,
    limit_records,
    load_existing_record_keys,
    manual_record_key,
    queries_for_category,
)
from scripts.data.merge_local_region_manual_eval_labels import (
    merge_labeled_records,
    record_key,
)
from scripts.data.build_local_region_manual_label_audit_manifest import (
    AUDIT_INSTRUCTION,
    build_audit_records,
)
from scripts.data.annotate_local_region_bboxes import (
    default_output_path,
    load_annotation_records,
    next_unlabeled_index,
    update_record,
    write_annotation_records,
)
from scripts.eval.evaluate_local_region_manual_labels import (
    parse_bbox,
    parse_manual_record,
    summarize_records,
)
from scripts.eval.export_local_region_manual_failures import (
    export_failure_cases,
    safe_stem,
    select_failure_records,
    write_failure_review_html,
)
from scripts.eval.evaluate_pretrained_grounding_manual_labels import (
    build_prompts,
    detections_from_hf_output,
    grounding_dino_text_prompt,
    summarize_records as summarize_pretrained_grounding_records,
)
from scripts.eval.evaluate_grounding_prompt_profiles import select_target_regions
from scripts.eval.evaluate_gated_hybrid_manual_labels import parse_grounding_routes
from scripts.eval.evaluate_gated_hybrid_manual_labels import parse_grounding_route_profiles
from scripts.eval.evaluate_gated_hybrid_manual_labels import parse_grounding_route_thresholds
from scripts.eval.evaluate_gated_hybrid_manual_labels import resolve_grounding_routes
from scripts.eval.evaluate_gated_hybrid_manual_labels import resolve_cli_grounding_policy
from scripts.eval.evaluate_gated_hybrid_manual_labels import resolve_prompt_profile
from scripts.eval.evaluate_gated_hybrid_manual_labels import resolve_score_threshold
from scripts.eval.evaluate_gated_hybrid_manual_labels import grounding_fallback_reason
from scripts.eval.evaluate_gated_hybrid_manual_labels import should_route_to_grounding
from scripts.eval.evaluate_gated_hybrid_manual_labels import evaluate_grounding_record
from scripts.eval.evaluate_gated_hybrid_manual_labels import diagnostic_grounding_payload
from scripts.eval.evaluate_gated_hybrid_manual_labels import (
    attach_grounding_fallback_provenance,
)
from scripts.inference.predict_gated_hybrid_local_region import (
    canonical_grounding_region,
    grounding_payload,
    should_use_grounding_route,
)
from fashion_mm.models.local_region.query import parse_region_query
from fashion_mm.models.local_region import filter_grounding_detections_to_garment
from fashion_mm.models.local_region import grounding_box_mask_coverage
from scripts.eval.evaluate_gated_hybrid_queries import (
    draw_reference_bbox,
    group_records_by_image,
    load_manifest_query_records,
    parsed_queries_for_route,
    summarize_gated_records,
)
from scripts.data.build_gated_hybrid_demo_manifest import (
    manifest_record,
    select_demo_records,
)
from scripts.eval.compare_local_region_manual_evals import (
    compare_evals,
    parse_fixed_region_policy,
)
from scripts.eval.analyze_gated_hybrid_confidence import (
    choose_best_threshold,
    common_records,
    confidence_gated_records,
)
from scripts.eval.export_gated_hybrid_policy_deltas import (
    export_policy_deltas,
    paired_policy_deltas,
    write_policy_delta_html,
)
from scripts.eval.analyze_local_region_routing_oracle import build_routing_oracle
from scripts.eval.analyze_grounding_wearer_side_selection import (
    desired_image_side,
    select_wearer_side_detection,
)
from scripts.eval.analyze_grounding_candidate_oracle import build_candidate_oracle
from scripts.eval.analyze_grounding_candidate_oracle import best_manual_candidate
from scripts.eval.evaluate_chinese_clip_manual_local_regions import (
    empty_prediction_record,
    finalize_run as finalize_chinese_clip_manual_run,
)
from scripts.eval.cross_validate_grounding_candidate_selector import (
    calibrate_nested_region_policies,
    candidate_examples,
    choose_nested_region_policies,
    image_grouped_folds,
    keep_current_candidate_record,
    pairwise_recovery_examples,
    parse_threshold_grid,
    select_candidate_record,
    select_conservative_candidate_record,
    selector_candidates,
    train_conservative_selector,
    train_selector,
    transition_counts_at_threshold,
    visual_scores_by_box,
)
from scripts.eval.enrich_grounding_candidates_with_chinese_clip import contextual_box
from scripts.eval.enrich_grounding_candidates_with_chinese_clip import record_text_prompts
from scripts.eval.enrich_grounding_candidates_with_chinese_clip import relative_rank_scores
from scripts.eval.enrich_grounding_candidates_with_chinese_clip import score_record_candidates
from scripts.eval.enrich_grounding_candidates_with_dinov2 import (
    deterministic_projection,
)
from scripts.eval.enrich_grounding_candidates_with_dinov2 import (
    project_dinov2_features,
)
from scripts.eval.enrich_grounding_candidates_with_dinov2 import (
    projection_fingerprint,
)
from scripts.eval.enrich_grounding_candidates_with_dinov2 import (
    score_record_candidates as score_dinov2_record_candidates,
)


def test_manual_manifest_records_start_unlabeled(tmp_path):
    image_path = tmp_path / "000001.jpg"
    Image.new("RGB", (80, 100)).save(image_path)

    records = build_manifest_records(
        [image_path],
        ["这件衣服的领口", "右侧的口袋"],
    )

    assert len(records) == 2
    assert records[0]["image"] == str(image_path)
    assert records[0]["target_region"] == "neckline"
    assert records[0]["target_bbox"] is None
    assert records[0]["label_status"] == "unlabeled"
    assert records[0]["image_width"] == 80
    assert records[0]["image_height"] == 100
    assert records[1]["target_region"] == "pocket"


def test_class_aware_manifest_uses_category_queries(tmp_path):
    image_path = tmp_path / "000001.jpg"
    anno_dir = tmp_path / "annos"
    anno_dir.mkdir()
    Image.new("RGB", (80, 100)).save(image_path)
    (anno_dir / "000001.json").write_text(
        '{"source": "000001.jpg", "item1": {"category_id": 8}}',
        encoding="utf-8",
    )

    records = build_class_aware_manifest_records([image_path], anno_dir)

    assert [record["query_text"] for record in records] == list(queries_for_category(8))
    assert {record["target_region"] for record in records} == {
        "waist",
        "hem",
        "pocket",
        "zipper",
        "pattern",
    }
    assert all(record["category_name"] == "trousers" for record in records)
    assert all(record["source_item_key"] == "item1" for record in records)


def test_class_aware_queries_include_the_referred_garment():
    trouser_queries = queries_for_category(8)
    outerwear_queries = queries_for_category(4)

    assert "这条裤子右侧的口袋" in trouser_queries
    assert "这条裤子上的拉链" in trouser_queries
    assert "这件外套左侧的袖口" in outerwear_queries
    assert parse_region_query("这件外套右侧的口袋").region == "pocket"


def test_manual_manifest_filters_target_regions(tmp_path):
    image_path = tmp_path / "000001.jpg"
    Image.new("RGB", (80, 100)).save(image_path)
    records = build_manifest_records(
        [image_path],
        ["这件衣服的领口", "衣服上的拉链", "这件衣服上的碎花图案"],
    )

    filtered = filter_records_by_target_regions(records, {"pattern", "zipper"})

    assert [record["target_region"] for record in filtered] == ["zipper", "pattern"]


def test_manual_manifest_excludes_existing_records(tmp_path):
    image_path = tmp_path / "000001.jpg"
    Image.new("RGB", (80, 100)).save(image_path)
    records = build_manifest_records(
        [image_path],
        ["衣服上的拉链", "这件衣服上的碎花图案"],
    )
    existing = tmp_path / "existing.jsonl"
    existing.write_text(
        (
            '{"image": "%s", "query_text": "衣服上的拉链", '
            '"target_region": "zipper", "target_bbox": [1, 2, 3, 4], '
            '"label_status": "labeled"}\n'
        )
        % image_path,
        encoding="utf-8",
    )

    keys = load_existing_record_keys([str(existing)])
    filtered = filter_existing_records(records, keys)

    assert manual_record_key(records[0]) in keys
    assert [record["target_region"] for record in filtered] == ["pattern"]


def test_manual_manifest_balances_target_regions():
    records = [
        {"id": "p1", "target_region": "pattern"},
        {"id": "p2", "target_region": "pattern"},
        {"id": "z1", "target_region": "zipper"},
        {"id": "pocket1", "target_region": "pocket"},
        {"id": "pocket2", "target_region": "pocket"},
    ]

    selected = balanced_region_records(
        records,
        max_records=4,
        region_order=["pattern", "zipper", "pocket"],
    )

    assert [record["id"] for record in selected] == ["p1", "z1", "pocket1", "p2"]
    assert limit_records(
        records,
        2,
        balance_target_regions=False,
    ) == records[:2]


def test_parse_manual_record_validates_bbox_shape():
    with pytest.raises(ValueError, match="target_bbox"):
        parse_manual_record(
            {
                "image": "/tmp/1.jpg",
                "query_text": "这件衣服的领口",
                "target_region": "neckline",
                "target_bbox": [1, 2, 3],
            }
        )


def test_parse_bbox_rejects_empty_box():
    with pytest.raises(ValueError, match="x2 > x1"):
        parse_bbox([10, 10, 5, 20])


def test_manual_eval_summary_uses_manual_bbox_iou():
    summary = summarize_records(
        [
            {
                "status": "ok",
                "target_region": "neckline",
                "ranker_backend": "heuristic_text_region_ranker",
                "selected_region": "neckline",
                "manual_bbox_iou": 0.6,
            },
            {
                "status": "ok",
                "target_region": "hem",
                "ranker_backend": "hybrid_candidate_listwise_context_ranker",
                "selected_region": "hem",
                "manual_bbox_iou": 0.2,
            },
        ]
    )

    assert summary["num_records"] == 2
    assert summary["avg_manual_bbox_iou"] == pytest.approx(0.4)
    assert summary["manual_hit_at"]["0.3"] == pytest.approx(0.5)
    assert summary["manual_hit_at"]["0.5"] == pytest.approx(0.5)
    assert summary["by_region"]["neckline"]["avg_manual_bbox_iou"] == 0.6


def test_pretrained_grounding_prompt_builder_uses_region_and_side():
    pocket_prompts = build_prompts(
        "右侧的口袋",
        "pocket",
        prompt_mode="english",
    )
    pattern_prompts = build_prompts(
        "这件衣服上的碎花图案",
        None,
        prompt_mode="both",
    )

    assert pocket_prompts[0] == "right pocket"
    assert "pocket" in pocket_prompts
    assert "这件衣服上的碎花图案" in pattern_prompts
    assert "floral pattern" in pattern_prompts


def test_pretrained_grounding_prompt_profiles_keep_side_and_context():
    precise = build_prompts(
        "右侧的口袋",
        "pocket",
        prompt_mode="english",
        prompt_profile="precise",
    )
    fashion = build_prompts(
        "右侧的口袋",
        "pocket",
        prompt_mode="english",
        prompt_profile="fashion",
    )

    assert precise == ["right pocket"]
    assert fashion == ["right pocket on clothing"]


def test_wearer_side_detection_selection_uses_opposite_image_side():
    detections = [
        {"bbox": [60, 10, 90, 30], "score": 0.9, "prompt": "right sleeve cuff"},
        {"bbox": [10, 10, 35, 30], "score": 0.8, "prompt": "sleeve cuff"},
    ]

    selected, status = select_wearer_side_detection(
        detections,
        query_text="这件上衣右侧的袖口",
        image_width=100,
        min_score_ratio=0.5,
    )

    assert status == "side_candidate"
    assert selected == detections[1]
    assert desired_image_side("right") == "left"
    assert desired_image_side("left") == "right"


def test_wearer_side_detection_selection_rejects_weak_side_candidate():
    detections = [
        {"bbox": [60, 10, 90, 30], "score": 0.9, "prompt": "right sleeve cuff"},
        {"bbox": [10, 10, 35, 30], "score": 0.2, "prompt": "sleeve cuff"},
    ]

    selected, status = select_wearer_side_detection(
        detections,
        query_text="这件上衣右侧的袖口",
        image_width=100,
        min_score_ratio=0.5,
    )

    assert status == "no_credible_side_candidate"
    assert selected == detections[0]


def test_grounding_candidate_oracle_reports_recoverable_failures():
    records = [
        {
            "id": "cuff-1",
            "target_region": "cuff",
            "target_bbox": [0, 0, 10, 10],
            "predicted_bbox": [20, 20, 30, 30],
            "manual_bbox_iou": 0.0,
            "detections": [
                {"bbox": [20, 20, 30, 30], "score": 0.9, "prompt": "cuff"},
                {"bbox": [0, 0, 10, 10], "score": 0.7, "prompt": "sleeve cuff"},
            ],
        },
        {
            "id": "pocket-1",
            "target_region": "pocket",
            "target_bbox": [0, 0, 10, 10],
            "predicted_bbox": [0, 0, 10, 10],
            "manual_bbox_iou": 1.0,
            "grounding_detections": [
                {"bbox": [20, 20, 30, 30], "score": 0.9, "prompt": "pocket"},
            ],
        },
    ]

    oracle_records, diagnostics = build_candidate_oracle(
        records,
        regions={"cuff", "pocket"},
        hit_threshold=0.3,
    )

    assert oracle_records[0]["manual_bbox_iou"] == pytest.approx(1.0)
    assert oracle_records[0]["candidate_oracle_rank"] == 2
    assert oracle_records[1]["manual_bbox_iou"] == pytest.approx(1.0)
    assert diagnostics["by_region"]["cuff"]["recoverable_failures"] == 1
    assert diagnostics["by_region"]["cuff"]["records_with_grounding_candidates"] == 1
    assert diagnostics["by_region"]["pocket"]["oracle_hits"] == 1
    assert diagnostics["oracle_source_counts"] == {
        "grounding_candidate": 1,
        "current_selection": 1,
    }


def test_grounding_candidate_oracle_can_include_heuristic_candidate():
    candidate, iou = best_manual_candidate(
        {
            "target_bbox": [0, 0, 10, 10],
            "detections": [
                {"bbox": [20, 20, 30, 30], "score": 0.9, "prompt": "pocket"},
            ],
            "heuristic_candidate": {
                "predicted_bbox": [0, 0, 10, 10],
                "selected_region": "right_pocket",
            },
        }
    )

    assert iou == pytest.approx(1.0)
    assert candidate["candidate_source"] == "heuristic"
    assert candidate["candidate_rank"] is None


def test_grounding_candidate_oracle_can_include_diagnostic_grounding():
    candidate, iou = best_manual_candidate(
        {
            "target_bbox": [0, 0, 10, 10],
            "diagnostic_grounding_candidate": {
                "detections": [
                    {
                        "bbox": [0, 0, 10, 10],
                        "score": 0.4,
                        "prompt": "clothing zipper",
                    }
                ]
            },
        }
    )

    assert iou == pytest.approx(1.0)
    assert candidate["candidate_source"] == "diagnostic_grounding"
    assert candidate["candidate_rank"] == 1


def test_grounding_candidate_oracle_unions_selected_and_diagnostic_detections():
    candidate, iou = best_manual_candidate(
        {
            "target_bbox": [0, 0, 10, 10],
            "detections": [
                {"bbox": [20, 20, 30, 30], "score": 0.9, "prompt": "cuff"},
            ],
            "diagnostic_grounding_candidate": {
                "detections": [
                    {"bbox": [0, 0, 10, 10], "score": 0.5, "prompt": "sleeve cuff"},
                ]
            },
        }
    )

    assert iou == pytest.approx(1.0)
    assert candidate["candidate_source"] == "diagnostic_grounding"
    assert candidate["candidate_rank"] == 1


def test_diagnostic_grounding_payload_keeps_candidate_provenance():
    payload = diagnostic_grounding_payload(
        {
            "status": "ok",
            "grounding_model_name": "model-b",
            "detections": [{"bbox": [1, 2, 3, 4], "score": 0.7}],
            "gated_policy_route": "grounding",
        }
    )

    assert payload["grounding_model_name"] == "model-b"
    assert payload["detections"][0]["bbox"] == [1, 2, 3, 4]
    assert "gated_policy_route" not in payload


def test_selector_candidates_union_sources_and_deduplicate_boxes():
    candidates = selector_candidates(
        {
            "predicted_bbox": [0, 0, 10, 10],
            "selected_region": "cuff",
            "score": 0.9,
            "detections": [
                {"bbox": [0, 0, 10, 10], "score": 0.9, "prompt": "cuff"},
                {"bbox": [10, 0, 20, 10], "score": 0.6, "prompt": "sleeve cuff"},
            ],
            "diagnostic_grounding_candidate": {
                "detections": [
                    {"bbox": [20, 0, 30, 10], "score": 0.5, "prompt": "cuff"},
                ]
            },
            "heuristic_candidate": {
                "predicted_bbox": [30, 0, 40, 10],
                "selected_region": "left_cuff",
            },
        }
    )

    assert len(candidates) == 4
    assert [candidate["candidate_source"] for candidate in candidates] == [
        "current",
        "grounding",
        "diagnostic_grounding",
        "heuristic",
    ]


def test_candidate_examples_build_fixed_features_and_manual_ious():
    features, ious, candidates = candidate_examples(
        {
            "image": "/tmp/example.jpg",
            "query_text": "这件上衣右侧的袖口",
            "target_region": "cuff",
            "target_bbox": [0, 0, 10, 10],
            "predicted_bbox": [20, 0, 30, 10],
            "selected_region": "right_cuff",
            "score": 0.8,
            "gated_policy_route": "grounding",
            "grounding_model_name": "google/owlv2-large-patch14-ensemble",
            "detections": [
                {"bbox": [0, 0, 10, 10], "score": 0.7, "prompt": "sleeve cuff"},
            ],
        },
        (100, 80),
    )

    assert features.ndim == 2
    assert features.shape[0] == len(candidates) == 2
    assert ious.tolist() == pytest.approx([0.0, 1.0])


def test_visual_candidate_scores_attach_by_bbox():
    record = {
        "image": "/tmp/example.jpg",
        "query_text": "衣服上的拉链",
        "target_region": "zipper",
        "target_bbox": [0, 0, 10, 10],
        "predicted_bbox": [20, 0, 30, 10],
        "diagnostic_grounding_candidate": {
            "detections": [
                {"bbox": [0, 0, 10, 10], "score": 0.8, "prompt": "zipper"},
            ]
        },
        "visual_candidate_scores": [
            {
                "bbox": [20, 0, 30, 10],
                "tight_score": 0.2,
                "context_score": 0.3,
                "max_score": 0.3,
                "mean_score": 0.25,
                "tight_rank_score": 0.0,
                "context_rank_score": 0.0,
            },
            {
                "bbox": [0, 0, 10, 10],
                "tight_score": 0.8,
                "context_score": 0.7,
                "max_score": 0.8,
                "mean_score": 0.75,
                "tight_rank_score": 1.0,
                "context_rank_score": 1.0,
            },
        ],
    }

    indexed = visual_scores_by_box(record)
    features, _, candidates = candidate_examples(record, (100, 80))

    assert len(indexed) == 2
    assert features.shape[0] == len(candidates) == 2
    assert candidates[0]["visual_tight_score"] == pytest.approx(0.2)
    assert candidates[1]["visual_context_score"] == pytest.approx(0.7)


def test_chinese_clip_candidate_enrichment_uses_tight_and_context_crops(monkeypatch):
    record = {
        "image": "/tmp/example.jpg",
        "query_text": "衣服上的拉链",
        "target_region": "zipper",
        "predicted_bbox": [20, 0, 30, 10],
        "diagnostic_grounding_candidate": {
            "detections": [
                {"bbox": [0, 0, 10, 10], "score": 0.8, "prompt": "zipper"},
            ]
        },
    }
    monkeypatch.setattr(
        "scripts.eval.enrich_grounding_candidates_with_chinese_clip.encode_text_ensemble",
        lambda *args, **kwargs: torch.tensor([[1.0, 0.0]]),
    )
    monkeypatch.setattr(
        "scripts.eval.enrich_grounding_candidates_with_chinese_clip.encode_images",
        lambda *args, **kwargs: torch.tensor(
            [
                [0.2, 0.8],
                [0.9, 0.1],
                [0.3, 0.7],
                [0.7, 0.3],
            ]
        ),
    )

    scores = score_record_candidates(
        record,
        Image.new("RGB", (100, 80)),
        model=object(),
        processor=object(),
        device=torch.device("cpu"),
        prompt_profile="region_ensemble",
        context_scale=1.6,
        image_batch_size=8,
    )

    assert len(scores) == 2
    assert scores[0]["tight_score"] == pytest.approx(0.2)
    assert scores[1]["tight_score"] == pytest.approx(0.9)
    assert scores[0]["context_score"] == pytest.approx(0.3)
    assert scores[1]["context_score"] == pytest.approx(0.7)
    assert scores[1]["tight_rank_score"] == pytest.approx(1.0)


def test_chinese_clip_enrichment_helpers_are_fixed_and_bounded():
    assert contextual_box([0, 0, 20, 10], (100, 80), 2.0) == pytest.approx(
        (0.0, 0.0, 30.0, 15.0)
    )
    assert relative_rank_scores(torch.tensor([0.2, 0.8, 0.5])) == pytest.approx(
        [0.0, 1.0, 0.5]
    )
    assert record_text_prompts(
        {
            "query_text": "衣服上的拉链",
            "target_region": "zipper",
        },
        "region_ensemble",
    ) == ["衣服上的拉链", "衣服上用于开合的拉链"]


def test_dinov2_projection_is_deterministic_and_normalized():
    first = deterministic_projection(
        4,
        output_dim=64,
        seed=42,
        device=torch.device("cpu"),
    )
    second = deterministic_projection(
        4,
        output_dim=64,
        seed=42,
        device=torch.device("cpu"),
    )
    different = deterministic_projection(
        4,
        output_dim=64,
        seed=43,
        device=torch.device("cpu"),
    )
    projected = project_dinov2_features(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        first,
    )

    assert torch.equal(first, second)
    assert projection_fingerprint(first) == projection_fingerprint(second)
    assert projection_fingerprint(first) != projection_fingerprint(different)
    assert projected.shape == (2, 64)
    assert torch.linalg.vector_norm(projected, dim=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_dinov2_enrichment_preserves_clip_scores_without_target_box(monkeypatch):
    record = {
        "image": "/tmp/example.jpg",
        "query_text": "左边的袖口",
        "target_region": "cuff",
        "predicted_bbox": [20, 0, 30, 10],
        "diagnostic_grounding_candidate": {
            "detections": [
                {"bbox": [0, 0, 10, 10], "score": 0.8, "prompt": "sleeve cuff"},
            ]
        },
        "visual_candidate_scores": [
            {
                "bbox": [20, 0, 30, 10],
                "tight_score": 0.2,
                "context_score": 0.3,
                "max_score": 0.3,
                "mean_score": 0.25,
                "tight_rank_score": 0.0,
                "context_rank_score": 0.0,
            }
        ],
    }
    monkeypatch.setattr(
        "scripts.eval.enrich_grounding_candidates_with_dinov2.encode_dinov2_images",
        lambda *args, **kwargs: torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    )
    projection = deterministic_projection(
        4,
        output_dim=64,
        seed=42,
        device=torch.device("cpu"),
    )

    scores = score_dinov2_record_candidates(
        record,
        Image.new("RGB", (100, 80)),
        model=object(),
        processor=object(),
        projection=projection,
        device=torch.device("cpu"),
        context_scale=1.6,
        image_batch_size=8,
    )

    assert len(scores) == 2
    assert scores[0]["tight_score"] == pytest.approx(0.2)
    assert len(scores[0]["dinov2_tight_embedding"]) == 64
    assert len(scores[1]["dinov2_context_embedding"]) == 64
    assert scores[0]["dinov2_tight_context_similarity"] == pytest.approx(1.0)

    enriched_record = {
        **record,
        "target_bbox": [0, 0, 10, 10],
        "visual_candidate_scores": scores,
    }
    enriched_features, _, candidates = candidate_examples(
        enriched_record,
        (100, 80),
    )
    legacy_features, _, _ = candidate_examples(
        {**enriched_record, "visual_candidate_scores": []},
        (100, 80),
    )

    assert enriched_features.shape == legacy_features.shape
    assert len(candidates[0]["visual_dinov2_tight_embedding"]) == 64
    assert not torch.equal(enriched_features, legacy_features)


def test_image_grouped_folds_do_not_split_one_image():
    records = [
        {"image": "a.jpg"},
        {"image": "a.jpg"},
        {"image": "b.jpg"},
        {"image": "c.jpg"},
        {"image": "d.jpg"},
    ]

    folds = image_grouped_folds(records, num_folds=2, seed=42)

    fold_for_index = {
        index: fold_index
        for fold_index, indices in enumerate(folds)
        for index in indices
    }
    assert fold_for_index[0] == fold_for_index[1]
    assert sorted(index for fold in folds for index in fold) == list(range(5))


def test_manual_candidate_selector_trains_and_selects_on_cpu():
    records = [
        {
            "id": f"record-{index}",
            "image": f"image-{index}.jpg",
            "query_text": "衣服上的拉链",
            "target_region": "zipper",
            "target_bbox": [0, 0, 10, 10],
            "predicted_bbox": [20, 0, 30, 10],
            "selected_region": "zipper",
            "diagnostic_grounding_candidate": {
                "grounding_model_name": "IDEA-Research/grounding-dino-base",
                "detections": [
                    {"bbox": [0, 0, 10, 10], "score": 0.6, "prompt": "zipper"},
                ],
            },
        }
        for index in range(4)
    ]
    examples = [candidate_examples(record, (100, 80)) for record in records]

    model = train_selector(
        examples,
        [0, 1, 2],
        hidden_dim=16,
        num_epochs=2,
        learning_rate=0.003,
        weight_decay=0.01,
        seed=42,
        device=torch.device("cpu"),
    )
    selected = select_candidate_record(
        records[3],
        examples[3],
        model,
        torch.device("cpu"),
    )

    assert selected["selector_source"] in {"current", "diagnostic_grounding"}
    assert len(selected["predicted_bbox"]) == 4


def test_pairwise_recovery_labels_only_miss_to_hit_overrides():
    records = [
        {
            "image": "miss.jpg",
            "query_text": "衣服上的拉链",
            "target_region": "zipper",
            "target_bbox": [0, 0, 10, 10],
            "predicted_bbox": [20, 0, 30, 10],
            "diagnostic_grounding_candidate": {
                "detections": [
                    {"bbox": [0, 0, 10, 10], "score": 0.8, "prompt": "zipper"},
                ]
            },
        },
        {
            "image": "hit.jpg",
            "query_text": "衣服上的拉链",
            "target_region": "zipper",
            "target_bbox": [0, 0, 10, 10],
            "predicted_bbox": [0, 0, 10, 10],
            "diagnostic_grounding_candidate": {
                "detections": [
                    {"bbox": [20, 0, 30, 10], "score": 0.8, "prompt": "zipper"},
                ]
            },
        },
    ]
    examples = [candidate_examples(record, (100, 80)) for record in records]

    pair_features, labels = pairwise_recovery_examples(examples, [0, 1])

    assert pair_features.shape[0] == 2
    assert labels.tolist() == [1.0, 0.0]


def test_conservative_selector_respects_override_threshold():
    records = [
        {
            "id": f"pair-{index}",
            "image": f"pair-{index}.jpg",
            "query_text": "衣服上的拉链",
            "target_region": "zipper",
            "target_bbox": [0, 0, 10, 10],
            "predicted_bbox": [20, 0, 30, 10],
            "selected_region": "zipper",
            "diagnostic_grounding_candidate": {
                "grounding_model_name": "IDEA-Research/grounding-dino-base",
                "detections": [
                    {"bbox": [0, 0, 10, 10], "score": 0.8, "prompt": "zipper"},
                ],
            },
        }
        for index in range(4)
    ]
    examples = [candidate_examples(record, (100, 80)) for record in records]
    model = train_conservative_selector(
        examples,
        [0, 1, 2],
        hidden_dim=16,
        num_epochs=2,
        learning_rate=0.003,
        weight_decay=0.01,
        seed=42,
        device=torch.device("cpu"),
    )

    kept = select_conservative_candidate_record(
        records[3],
        examples[3],
        model,
        torch.device("cpu"),
        override_threshold=1.0,
    )
    overridden = select_conservative_candidate_record(
        records[3],
        examples[3],
        model,
        torch.device("cpu"),
        override_threshold=0.0,
    )

    assert kept["selector_source"] == "current"
    assert kept["selector_overrode_current"] is False
    assert overridden["selector_source"] == "diagnostic_grounding"
    assert overridden["selector_overrode_current"] is True


def test_nested_region_policy_enables_only_safe_inner_oof_gain():
    rows = [
        {
            "target_region": "cuff",
            "probability": 0.9,
            "current_iou": 0.1,
            "alternative_iou": 0.8,
        },
        {
            "target_region": "cuff",
            "probability": 0.8,
            "current_iou": 0.8,
            "alternative_iou": 0.1,
        },
        {
            "target_region": "pocket",
            "probability": 0.9,
            "current_iou": 0.8,
            "alternative_iou": 0.1,
        },
    ]

    policies = choose_nested_region_policies(
        rows,
        regions={"cuff", "pocket"},
        thresholds=(0.5, 0.85, 0.95),
        max_lost_hits=0,
        min_net_gain=1,
    )

    assert policies["cuff"] == {
        "enabled": True,
        "threshold": 0.85,
        "num_inner_oof_records": 2,
        "num_overrides": 1,
        "gained_hit": 1,
        "lost_hit": 0,
        "net_gain": 1,
    }
    assert policies["pocket"]["enabled"] is False


def test_nested_threshold_helpers_and_disabled_region_keep_current():
    assert parse_threshold_grid("0.9,0.5,0.9") == (0.5, 0.9)
    assert transition_counts_at_threshold(
        [
            {
                "probability": 0.8,
                "current_iou": 0.1,
                "alternative_iou": 0.7,
            },
            {
                "probability": 0.7,
                "current_iou": 0.8,
                "alternative_iou": 0.1,
            },
        ],
        0.75,
    ) == {
        "num_overrides": 1,
        "gained_hit": 1,
        "lost_hit": 0,
        "net_gain": 1,
    }

    record = {
        "image": "example.jpg",
        "query_text": "衣服上的拉链",
        "target_region": "zipper",
        "target_bbox": [0, 0, 10, 10],
        "predicted_bbox": [20, 0, 30, 10],
        "selected_region": "zipper",
        "manual_bbox_iou": 0.0,
        "diagnostic_grounding_candidate": {
            "detections": [
                {"bbox": [0, 0, 10, 10], "score": 0.8, "prompt": "zipper"},
            ]
        },
    }
    example = candidate_examples(record, (100, 80))
    kept = keep_current_candidate_record(record, example)

    assert kept["predicted_bbox"] == [20.0, 0.0, 30.0, 10.0]
    assert kept["selector_source"] == "current"
    assert kept["selector_overrode_current"] is False


def test_nested_region_calibration_produces_inner_oof_policies():
    records = [
        {
            "image": f"nested-{index}.jpg",
            "query_text": "衣服上的拉链",
            "target_region": "zipper",
            "target_bbox": [0, 0, 10, 10],
            "predicted_bbox": [20, 0, 30, 10],
            "selected_region": "zipper",
            "manual_bbox_iou": 0.0,
            "diagnostic_grounding_candidate": {
                "detections": [
                    {"bbox": [0, 0, 10, 10], "score": 0.8, "prompt": "zipper"},
                ]
            },
        }
        for index in range(6)
    ]
    examples = [candidate_examples(record, (100, 80)) for record in records]

    policies = calibrate_nested_region_policies(
        examples,
        records,
        list(range(6)),
        regions={"zipper"},
        thresholds=(0.0,),
        num_inner_folds=2,
        max_lost_hits=0,
        min_net_gain=1,
        hidden_dim=8,
        num_epochs=2,
        learning_rate=0.003,
        weight_decay=0.01,
        architecture="linear",
        seed=42,
        device=torch.device("cpu"),
    )

    assert policies["zipper"]["enabled"] is True
    assert policies["zipper"]["num_inner_oof_records"] == 6
    assert policies["zipper"]["gained_hit"] == 6


def test_manual_grounding_record_applies_validated_wearer_side_selection(tmp_path):
    image_path = tmp_path / "cuff.jpg"
    Image.new("RGB", (100, 80)).save(image_path)

    class FakeGrounder:
        backend = "owlv2"
        model_name = "fake-owlv2"
        score_threshold = 0.05

        def predict(self, image, prompts):
            detections = [
                {
                    "bbox": [60, 10, 90, 30],
                    "score": 0.9,
                    "prompt": "right sleeve cuff",
                },
                {
                    "bbox": [10, 10, 35, 30],
                    "score": 0.8,
                    "prompt": "sleeve cuff",
                },
            ]
            return {"status": "ok", "best": detections[0], "detections": detections}

    record = evaluate_grounding_record(
        {
            "id": "cuff-1",
            "image": str(image_path),
            "query_text": "这件上衣右侧的袖口",
            "target_region": "cuff",
            "target_bbox": [10, 10, 35, 30],
        },
        grounder=FakeGrounder(),
        image_cache={},
        prompt_mode="english",
        prompt_profile="precise",
        apply_wearer_side_selection=True,
        wearer_side_min_score_ratio=0.5,
    )

    assert record["predicted_bbox"] == [10.0, 10.0, 35.0, 30.0]
    assert record["manual_bbox_iou"] == pytest.approx(1.0)
    assert record["wearer_side_selection_status"] == "side_candidate"


def test_grounding_candidate_generation_does_not_depend_on_target_bbox(tmp_path):
    image_path = tmp_path / "pocket.jpg"
    Image.new("RGB", (100, 80)).save(image_path)

    class FakeGrounder:
        backend = "grounding_dino"
        model_name = "fake-grounding-dino"
        score_threshold = 0.15

        def predict(self, image, prompts):
            detection = {
                "bbox": [10, 10, 30, 30],
                "score": 0.8,
                "prompt": "pocket",
            }
            return {"status": "ok", "best": detection, "detections": [detection]}

    common = {
        "image": str(image_path),
        "query_text": "右侧的口袋",
        "target_region": "pocket",
    }
    first = evaluate_grounding_record(
        {**common, "target_bbox": [10, 10, 30, 30]},
        grounder=FakeGrounder(),
        image_cache={},
        prompt_mode="english",
        prompt_profile="ensemble",
    )
    second = evaluate_grounding_record(
        {**common, "target_bbox": [60, 40, 80, 60]},
        grounder=FakeGrounder(),
        image_cache={},
        prompt_mode="english",
        prompt_profile="ensemble",
    )

    assert first["predicted_bbox"] == second["predicted_bbox"]
    assert first["detections"] == second["detections"]
    assert first["manual_bbox_iou"] != second["manual_bbox_iou"]


def test_grounding_fallback_preserves_candidate_model_provenance():
    fallback = {"gated_policy_route": "heuristic_fallback_no_detection"}
    attach_grounding_fallback_provenance(
        fallback,
        {
            "grounding_model_name": "google/owlv2-large-patch14-ensemble",
            "grounding_score_threshold": 0.05,
            "prompt_profile": "precise",
            "prompts": ["right sleeve cuff"],
        },
    )

    assert fallback["grounding_model_name"].startswith("google/owlv2")
    assert fallback["grounding_score_threshold"] == 0.05
    assert fallback["prompt_profile"] == "precise"


def test_prompt_profile_target_region_filter_is_exact():
    records = [
        {"id": "pattern", "target_region": "pattern"},
        {"id": "pocket", "target_region": "pocket"},
        {"id": "hem", "target_region": "hem"},
    ]

    selected = select_target_regions(records, {"pattern", "pocket"})

    assert [record["id"] for record in selected] == ["pattern", "pocket"]


def test_grounding_garment_filter_rejects_background_detection():
    garment_mask = np.zeros((10, 10), dtype=bool)
    garment_mask[2:8, 2:8] = True
    detections = [
        {"prompt": "pocket", "score": 0.9, "bbox": [0, 0, 2, 2]},
        {"prompt": "pocket", "score": 0.8, "bbox": [3, 3, 6, 6]},
    ]

    filtered = filter_grounding_detections_to_garment(
        detections,
        garment_mask,
        min_mask_coverage=0.5,
    )

    assert grounding_box_mask_coverage(detections[0]["bbox"], garment_mask) == 0.0
    assert len(filtered) == 1
    assert filtered[0]["score"] == 0.8
    assert filtered[0]["garment_mask_coverage"] == pytest.approx(1.0)


def test_gated_hybrid_routes_only_configured_regions_to_grounding():
    grounding_regions = {"pattern", "pocket"}

    assert should_route_to_grounding(
        {"target_region": "pattern"},
        grounding_regions,
    )
    assert should_route_to_grounding(
        {"target_region": "pocket"},
        grounding_regions,
    )
    assert not should_route_to_grounding(
        {"target_region": "zipper"},
        grounding_regions,
    )
    assert not should_route_to_grounding(
        {"target_region": "hem"},
        grounding_regions,
    )


def test_explicit_grounding_routes_override_single_model_defaults():
    routes = parse_grounding_routes(
        [
            "pattern=IDEA-Research/grounding-dino-tiny",
            "pocket=IDEA-Research/grounding-dino-base",
            "cuff=IDEA-Research/grounding-dino-base",
        ]
    )

    resolved = resolve_grounding_routes(
        grounding_regions={"pattern", "pocket"},
        grounding_model_name="IDEA-Research/grounding-dino-tiny",
        grounding_routes=routes,
    )

    assert resolved == {
        "pattern": "IDEA-Research/grounding-dino-tiny",
        "pocket": "IDEA-Research/grounding-dino-base",
        "cuff": "IDEA-Research/grounding-dino-base",
    }


def test_explicit_grounding_routes_validate_syntax():
    with pytest.raises(ValueError, match="REGION=MODEL_NAME"):
        parse_grounding_routes(["pocket"])


def test_explicit_grounding_route_profiles_override_default_profile():
    profiles = parse_grounding_route_profiles(["cuff=precise", "waist=ensemble"])

    assert resolve_prompt_profile(
        "cuff",
        default_profile="ensemble",
        route_profiles=profiles,
    ) == "precise"
    assert resolve_prompt_profile(
        "pocket",
        default_profile="ensemble",
        route_profiles=profiles,
    ) == "ensemble"
    with pytest.raises(ValueError, match="Unsupported prompt profile"):
        parse_grounding_route_profiles(["cuff=unknown"])


def test_explicit_grounding_route_thresholds_override_default_threshold():
    thresholds = parse_grounding_route_thresholds(["cuff=0.05", "waist=0.05"])

    assert resolve_score_threshold(
        "cuff",
        default_threshold=0.15,
        route_thresholds=thresholds,
    ) == pytest.approx(0.05)
    assert resolve_score_threshold(
        "pocket",
        default_threshold=0.15,
        route_thresholds=thresholds,
    ) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_grounding_route_thresholds(["cuff=1.1"])


def test_cli_grounding_policy_forwards_route_profiles_and_thresholds():
    args = Namespace(
        grounding_regions=["pattern", "pocket"],
        grounding_model_name="IDEA-Research/grounding-dino-tiny",
        grounding_routes=[
            "pattern=IDEA-Research/grounding-dino-tiny",
            "cuff=google/owlv2-large-patch14-ensemble",
        ],
        grounding_route_profiles=["cuff=precise"],
        grounding_route_thresholds=["cuff=0.05"],
    )

    regions, routes, profiles, thresholds, resolved = resolve_cli_grounding_policy(args)

    assert regions == {"pattern", "pocket"}
    assert routes == {
        "pattern": "IDEA-Research/grounding-dino-tiny",
        "cuff": "google/owlv2-large-patch14-ensemble",
    }
    assert profiles == {"cuff": "precise"}
    assert thresholds == {"cuff": pytest.approx(0.05)}
    assert resolved == routes


def test_grounding_no_detection_fallback_is_explicit_and_opt_in():
    assert grounding_fallback_reason(
        {"status": "no_detection"},
        constrain_grounding_to_garment=False,
        fallback_on_no_detection=False,
    ) is None
    assert grounding_fallback_reason(
        {"status": "no_detection"},
        constrain_grounding_to_garment=False,
        fallback_on_no_detection=True,
    ) == "no_detection"
    assert grounding_fallback_reason(
        {"status": "no_detection_in_selected_garment"},
        constrain_grounding_to_garment=True,
        fallback_on_no_detection=False,
    ) == "garment_filter"


def test_single_image_gated_hybrid_routes_by_parsed_query():
    grounding_regions = {"pattern", "pocket"}

    assert should_use_grounding_route(
        parse_region_query("这件衣服上的碎花图案"),
        grounding_regions,
    )
    assert should_use_grounding_route(
        parse_region_query("右侧的口袋"),
        grounding_regions,
    )
    assert not should_use_grounding_route(
        parse_region_query("衣服上的拉链"),
        grounding_regions,
    )
    assert not should_use_grounding_route(
        parse_region_query("衣服下方的下摆"),
        grounding_regions,
    )


def test_single_image_grounding_payload_matches_local_region_shape(tmp_path):
    prediction = {
        "status": "ok",
        "best": {"prompt": "floral pattern", "score": 0.91, "bbox": [1, 2, 11, 12]},
        "detections": [
            {"prompt": "floral pattern", "score": 0.91, "bbox": [1, 2, 11, 12]},
            {"prompt": "pattern", "score": 0.42, "bbox": [4, 5, 14, 15]},
        ],
    }

    payload = grounding_payload(
        image_path=tmp_path / "image.jpg",
        query="这件衣服上的碎花图案",
        parsed_query=parse_region_query("这件衣服上的碎花图案"),
        prediction=prediction,
        prompts=["pattern", "floral pattern"],
        grounder_backend="auto",
        grounding_model_name="IDEA-Research/grounding-dino-tiny",
        latency_ms=12.5,
    )

    assert payload["gated_policy_route"] == "grounding"
    assert payload["ranker_backend"] == "gated_hybrid_grounding_auto"
    assert payload["query"]["region"] == "pattern"
    assert payload["region"]["region"] == "pattern"
    assert payload["region"]["raw_grounding_prompt"] == "floral pattern"
    assert payload["region"]["box"] == [1.0, 2.0, 11.0, 12.0]
    assert payload["candidate_regions"][1]["match_score"] == pytest.approx(0.42)


def test_canonical_grounding_region_keeps_side_aware_pocket_label():
    assert canonical_grounding_region(
        parse_region_query("右侧的口袋"),
        "right clothing pocket",
    ) == "right_pocket"
    assert canonical_grounding_region(
        parse_region_query("左边的袖口"),
        "cuff",
    ) == "left_cuff"
    assert canonical_grounding_region(
        parse_region_query("这件衣服上的碎花图案"),
        "floral pattern",
    ) == "pattern"


def test_batch_gated_query_summary_counts_routes():
    summary = summarize_gated_records(
        [
            {
                "status": "ok",
                "ranker_backend": "gated_hybrid_grounding_auto",
                "gated_policy_route": "grounding",
                "latency_ms": 20.0,
                "region": {"region": "floral pattern", "match_score": 0.8},
            },
            {
                "status": "ok",
                "ranker_backend": "heuristic_text_region_ranker",
                "gated_policy_route": "heuristic",
                "latency_ms": 10.0,
                "region": {"region": "hem", "match_score": 3.0},
            },
        ]
    )

    assert summary["gated_policy_route_counts"] == {
        "grounding": 1,
        "heuristic": 1,
    }
    assert summary["avg_local_region_latency_ms"] == pytest.approx(15.0)
    assert summary["avg_local_region_latency_by_route_ms"]["grounding"] == 20.0
    assert summary["selected_region_counts"] == {"floral pattern": 1, "hem": 1}


def test_batch_gated_query_parser_preserves_query_order():
    parsed = parsed_queries_for_route(["这件衣服上的碎花图案", "衣服下方的下摆"])

    assert [query for query, _ in parsed] == ["这件衣服上的碎花图案", "衣服下方的下摆"]
    assert [parsed_query.region for _, parsed_query in parsed] == ["pattern", "hem"]


def test_batch_gated_query_manifest_loads_image_query_records(tmp_path):
    manifest = tmp_path / "demo_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"id": "a", "image": "/tmp/1.jpg", "query_text": "这件衣服上的碎花图案"}',
                '{"image": "/tmp/2.jpg", "query": "右侧的口袋", "note": "valid pocket"}',
                '{"image": "/tmp/1.jpg", "query_text": "衣服下方的下摆"}',
            ]
        ),
        encoding="utf-8",
    )

    records = load_manifest_query_records(manifest)
    grouped = group_records_by_image(records)

    assert [record["query_text"] for record in records] == [
        "这件衣服上的碎花图案",
        "右侧的口袋",
        "衣服下方的下摆",
    ]
    assert records[0]["id"] == "a"
    assert records[1]["id"] == "2__000001"
    assert records[1]["note"] == "valid pocket"
    assert list(grouped) == ["/tmp/1.jpg", "/tmp/2.jpg"]
    assert [record["id"] for record in grouped["/tmp/1.jpg"]] == ["a", "1__000002"]


def test_gated_demo_manifest_selects_successful_records_by_region_and_iou():
    records = [
        {
            "id": "pattern_low",
            "image": "/tmp/pattern_low.jpg",
            "query_text": "这件衣服上的碎花图案",
            "target_region": "pattern",
            "target_bbox": [1, 2, 10, 12],
            "status": "ok",
            "predicted_bbox": [0, 0, 1, 1],
            "manual_bbox_iou": 0.35,
            "gated_policy_route": "grounding",
        },
        {
            "id": "pattern_high",
            "image": "/tmp/pattern_high.jpg",
            "query_text": "这件衣服上的碎花图案",
            "target_region": "pattern",
            "target_bbox": [1, 2, 10, 12],
            "status": "ok",
            "predicted_bbox": [0, 0, 1, 1],
            "manual_bbox_iou": 0.8,
            "gated_policy_route": "grounding",
        },
        {
            "id": "hem_high",
            "image": "/tmp/hem.jpg",
            "query_text": "衣服下方的下摆",
            "target_region": "hem",
            "target_bbox": [1, 2, 10, 12],
            "status": "ok",
            "predicted_bbox": [0, 0, 1, 1],
            "manual_bbox_iou": 0.7,
            "gated_policy_route": "heuristic",
        },
        {
            "id": "shoulder_failed",
            "image": "/tmp/shoulder.jpg",
            "query_text": "这件衣服的肩部",
            "target_region": "shoulder",
            "target_bbox": [1, 2, 10, 12],
            "status": "no_detection",
            "predicted_bbox": None,
            "manual_bbox_iou": 0.9,
            "gated_policy_route": "heuristic",
        },
    ]

    selected, available = select_demo_records(
        records,
        target_regions=["pattern", "hem", "shoulder"],
        per_region=1,
        min_iou=0.3,
    )

    assert [record["id"] for record in selected] == ["pattern_high", "hem_high"]
    assert available == {"pattern": 2, "hem": 1, "shoulder": 0}
    payload = manifest_record(selected[0], 0)
    assert payload["target_region"] == "pattern"
    assert payload["gated_policy_route"] == "grounding"
    assert payload["selection_manual_bbox_iou"] == pytest.approx(0.8)
    assert payload["reference_bbox"] == [1, 2, 10, 12]


def test_batch_gated_query_draws_manifest_reference_bbox(tmp_path):
    output = tmp_path / "reference.jpg"
    Image.new("RGB", (40, 40), "white").save(output)

    draw_reference_bbox(output, [5, 6, 30, 31])

    image = Image.open(output).convert("RGB")
    assert image.getpixel((5, 6))[1] > 120
    assert image.getpixel((5, 6))[0] < 80


def test_confidence_gate_falls_back_to_heuristic_for_low_score_grounding():
    gated_records = [
        {
            "id": "pattern", "image": "/tmp/pattern.jpg", "query_text": "图案",
            "target_region": "pattern", "gated_policy_route": "grounding",
            "score": 0.8, "manual_bbox_iou": 0.9,
        },
        {
            "id": "pocket", "image": "/tmp/pocket.jpg", "query_text": "口袋",
            "target_region": "pocket", "gated_policy_route": "grounding",
            "score": 0.2, "manual_bbox_iou": 0.0,
        },
        {
            "id": "hem", "image": "/tmp/hem.jpg", "query_text": "下摆",
            "target_region": "hem", "gated_policy_route": "heuristic",
            "score": None, "manual_bbox_iou": 0.6,
        },
    ]
    heuristic_records = [
        {**record, "manual_bbox_iou": value}
        for record, value in zip(gated_records, [0.1, 0.5, 0.6], strict=True)
    ]
    gated_by_key, heuristic_by_key, keys = common_records(gated_records, heuristic_records)

    records, source_counts = confidence_gated_records(
        keys,
        gated_by_key=gated_by_key,
        heuristic_by_key=heuristic_by_key,
        grounding_regions={"pattern", "pocket"},
        confidence_threshold=0.3,
    )

    by_id = {record["id"]: record for record in records}
    assert by_id["pattern"]["manual_bbox_iou"] == pytest.approx(0.9)
    assert by_id["pocket"]["manual_bbox_iou"] == pytest.approx(0.5)
    assert by_id["hem"]["manual_bbox_iou"] == pytest.approx(0.6)
    assert source_counts == {
        "grounding": 1,
        "heuristic_fallback": 1,
        "unchanged_gated_policy": 1,
    }


def test_confidence_gate_selects_best_calibration_threshold():
    results = [
        {
            "confidence_threshold": 0.2,
            "semantic_summary": {
                "avg_manual_bbox_iou": 0.4,
                "manual_hit_at": {"0.3": 0.5, "0.5": 0.3},
            },
        },
        {
            "confidence_threshold": 0.3,
            "semantic_summary": {
                "avg_manual_bbox_iou": 0.45,
                "manual_hit_at": {"0.3": 0.4, "0.5": 0.3},
            },
        },
    ]

    selected = choose_best_threshold(results)

    assert selected["confidence_threshold"] == pytest.approx(0.3)


def test_grounding_dino_text_prompt_joins_phrases_with_periods():
    assert grounding_dino_text_prompt(["neckline", "sleeve cuff."]) == (
        "neckline. sleeve cuff."
    )


def test_pretrained_grounding_summary_uses_manual_iou():
    summary = summarize_pretrained_grounding_records(
        [
            {
                "status": "ok",
                "target_region": "neckline",
                "selected_region": "collar",
                "manual_bbox_iou": 0.7,
            },
            {
                "status": "no_detection",
                "target_region": "cuff",
                "selected_region": None,
                "manual_bbox_iou": 0.0,
            },
        ]
    )

    assert summary["num_records"] == 2
    assert summary["status_counts"] == {"ok": 1, "no_detection": 1}
    assert summary["avg_manual_bbox_iou"] == pytest.approx(0.35)
    assert summary["manual_hit_at"]["0.3"] == pytest.approx(0.5)
    assert summary["by_region"]["cuff"]["avg_manual_bbox_iou"] == 0.0


def test_detections_from_hf_output_maps_prompt_labels():
    processed = {
        "scores": [0.2, 0.8],
        "labels": [0, 1],
        "boxes": [[0, 0, 10, 10], [1, 2, 11, 12]],
    }

    detections = detections_from_hf_output(processed, ["neckline", "cuff"])

    assert detections[0]["prompt"] == "cuff"
    assert detections[0]["score"] == pytest.approx(0.8)
    assert detections[0]["bbox"] == [1.0, 2.0, 11.0, 12.0]


def test_detections_from_hf_output_accepts_text_labels():
    processed = {
        "scores": [0.8],
        "labels": ["sleeve cuff"],
        "boxes": [[1, 2, 11, 12]],
    }

    detections = detections_from_hf_output(processed, ["neckline", "sleeve cuff"])

    assert detections[0]["prompt"] == "sleeve cuff"
    assert detections[0]["prompt_index"] == 1


def test_compare_manual_evals_builds_region_hybrid_oracle():
    heuristic = {
        "name": "heuristic",
        "path": "/tmp/heuristic.json",
        "summary": {},
        "records": [
            {
                "image": "/tmp/1.jpg",
                "query_text": "衣服下方的下摆",
                "target_region": "hem",
                "manual_bbox_iou": 0.6,
            },
            {
                "image": "/tmp/2.jpg",
                "query_text": "这件衣服上的碎花图案",
                "target_region": "pattern",
                "manual_bbox_iou": 0.2,
            },
        ],
    }
    grounding_dino = {
        "name": "grounding_dino",
        "path": "/tmp/grounding_dino.json",
        "summary": {},
        "records": [
            {
                "image": "/tmp/1.jpg",
                "query_text": "衣服下方的下摆",
                "target_region": "hem",
                "manual_bbox_iou": 0.1,
            },
            {
                "image": "/tmp/2.jpg",
                "query_text": "这件衣服上的碎花图案",
                "target_region": "pattern",
                "manual_bbox_iou": 0.8,
            },
        ],
    }

    comparison = compare_evals([heuristic, grounding_dino])

    assert comparison["region_policy"] == {
        "hem": "heuristic",
        "pattern": "grounding_dino",
    }
    assert comparison["region_hybrid_oracle"]["avg_manual_bbox_iou"] == pytest.approx(0.7)


def test_compare_manual_evals_uses_id_to_disambiguate_duplicate_image_queries():
    heuristic = {
        "name": "heuristic",
        "path": "/tmp/heuristic.json",
        "summary": {},
        "records": [
            {
                "id": "000001_item1__000001",
                "image": "/tmp/1.jpg",
                "query_text": "右侧的口袋",
                "target_region": "pocket",
                "manual_bbox_iou": 0.1,
            },
            {
                "id": "000001_item2__000002",
                "image": "/tmp/1.jpg",
                "query_text": "右侧的口袋",
                "target_region": "pocket",
                "manual_bbox_iou": 0.2,
            },
        ],
    }
    grounding_dino = {
        "name": "grounding_dino",
        "path": "/tmp/grounding_dino.json",
        "summary": {},
        "records": [
            {
                "id": "000001_item1__000001",
                "image": "/tmp/1.jpg",
                "query_text": "右侧的口袋",
                "target_region": "pocket",
                "manual_bbox_iou": 0.3,
            },
            {
                "id": "000001_item2__000002",
                "image": "/tmp/1.jpg",
                "query_text": "右侧的口袋",
                "target_region": "pocket",
                "manual_bbox_iou": 0.4,
            },
        ],
    }

    comparison = compare_evals([heuristic, grounding_dino])

    assert comparison["num_common_records"] == 2
    assert comparison["per_region"]["pocket"]["best_eval"] == "grounding_dino"
    assert comparison["per_eval"]["heuristic"]["avg_manual_bbox_iou"] == pytest.approx(0.15)


def test_compare_manual_evals_builds_fixed_region_hybrid():
    heuristic = {
        "name": "heuristic",
        "path": "/tmp/heuristic.json",
        "summary": {},
        "records": [
            {
                "image": "/tmp/1.jpg",
                "query_text": "衣服下方的下摆",
                "target_region": "hem",
                "manual_bbox_iou": 0.6,
            },
            {
                "image": "/tmp/2.jpg",
                "query_text": "这件衣服上的碎花图案",
                "target_region": "pattern",
                "manual_bbox_iou": 0.2,
            },
        ],
    }
    grounding_dino = {
        "name": "grounding_dino",
        "path": "/tmp/grounding_dino.json",
        "summary": {},
        "records": [
            {
                "image": "/tmp/1.jpg",
                "query_text": "衣服下方的下摆",
                "target_region": "hem",
                "manual_bbox_iou": 0.1,
            },
            {
                "image": "/tmp/2.jpg",
                "query_text": "这件衣服上的碎花图案",
                "target_region": "pattern",
                "manual_bbox_iou": 0.8,
            },
        ],
    }
    policy = parse_fixed_region_policy(
        ["pattern=grounding_dino"],
        default_eval="heuristic",
        regions=["hem", "pattern"],
    )

    comparison = compare_evals(
        [heuristic, grounding_dino],
        fixed_region_policy=policy,
    )

    assert comparison["fixed_region_hybrid"]["region_policy"] == {
        "hem": "heuristic",
        "pattern": "grounding_dino",
    }
    assert comparison["fixed_region_hybrid"]["summary"]["avg_manual_bbox_iou"] == (
        pytest.approx(0.7)
    )
    assert [
        record["hybrid_source_eval"]
        for record in comparison["fixed_region_hybrid_records"]
    ] == ["heuristic", "grounding_dino"]


def test_annotator_default_output_path():
    assert default_output_path("/tmp/manual_manifest.jsonl") == Path(
        "/tmp/manual_manifest_labeled.jsonl"
    )


def test_annotator_updates_and_persists_records(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "labels.jsonl"
    manifest.write_text(
        '{"image": "/tmp/1.jpg", "query_text": "这件衣服的领口", '
        '"target_bbox": null, "label_status": "unlabeled"}\n',
        encoding="utf-8",
    )

    records = load_annotation_records(manifest, output)
    update_record(
        records,
        0,
        target_bbox=[10.2, 11.8, 50.1, 60.9],
        label_status="labeled",
        notes="ok",
    )
    write_annotation_records(records, output)
    restored = load_annotation_records(manifest, output)

    assert restored[0]["target_bbox"] == [10, 12, 50, 61]
    assert restored[0]["label_status"] == "labeled"
    assert restored[0]["notes"] == "ok"


def test_annotator_next_unlabeled_wraps():
    records = [
        {"label_status": "labeled"},
        {"label_status": "skip"},
        {"label_status": "unlabeled"},
    ]

    assert next_unlabeled_index(records, start=0) == 2
    assert next_unlabeled_index(records, start=2) == 2


def test_merge_manual_labels_keeps_labeled_and_deduplicates(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(
            [
                '{"image": "/tmp/1.jpg", "query_text": "q", "target_region": "hem", '
                '"target_bbox": [1, 2, 3, 4], "label_status": "labeled"}',
                '{"image": "/tmp/2.jpg", "query_text": "q2", "target_region": "cuff", '
                '"target_bbox": null, "label_status": "unlabeled"}',
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(
        '{"image": "/tmp/1.jpg", "query_text": "q", "target_region": "hem", '
        '"target_bbox": [5, 6, 7, 8], "label_status": "labeled"}\n',
        encoding="utf-8",
    )

    merged, summary = merge_labeled_records([first, second])

    assert len(merged) == 1
    assert merged[0]["target_bbox"] == [5, 6, 7, 8]
    assert merged[0]["merge_source"] == str(second)
    assert summary["num_duplicate_keys_replaced"] == 1
    assert summary["input_label_status_counts"] == {"labeled": 2, "unlabeled": 1}


def test_merge_record_key_uses_image_query_region():
    assert record_key(
        {
            "image": "/tmp/1.jpg",
            "query_text": "这件衣服的领口",
            "target_region": "neckline",
            "target_bbox": [1, 2, 3, 4],
        }
    ) == "fallback:/tmp/1.jpg\t这件衣服的领口\tneckline"


def test_merge_manual_labels_keeps_same_image_query_when_ids_differ(tmp_path):
    first = tmp_path / "first.jsonl"
    first.write_text(
        "\n".join(
            [
                '{"id": "000001_item1__000001", "image": "/tmp/1.jpg", '
                '"query_text": "右侧的口袋", "target_region": "pocket", '
                '"target_bbox": [1, 2, 3, 4], "label_status": "labeled"}',
                '{"id": "000001_item2__000002", "image": "/tmp/1.jpg", '
                '"query_text": "右侧的口袋", "target_region": "pocket", '
                '"target_bbox": [5, 6, 7, 8], "label_status": "labeled"}',
            ]
        ),
        encoding="utf-8",
    )

    merged, summary = merge_labeled_records([first])

    assert len(merged) == 2
    assert summary["num_duplicate_keys_replaced"] == 0


def test_merge_manual_labels_can_remove_existing_record_from_audit_skip(tmp_path):
    initial = tmp_path / "initial.jsonl"
    audit = tmp_path / "audit.jsonl"
    initial.write_text(
        '{"id": "zipper-1", "image": "/tmp/1.jpg", "query_text": "裤子上的拉链", '
        '"target_region": "zipper", "target_bbox": [1, 2, 3, 4], "label_status": "labeled"}\n',
        encoding="utf-8",
    )
    audit.write_text(
        '{"id": "zipper-1", "image": "/tmp/1.jpg", "query_text": "裤子上的拉链", '
        '"target_region": "zipper", "target_bbox": null, "label_status": "skip"}\n',
        encoding="utf-8",
    )

    merged, summary = merge_labeled_records(
        [initial, audit],
        skip_removes_existing=True,
    )

    assert merged == []
    assert summary["num_existing_records_removed_by_skip"] == 1


def test_audit_manifest_preserves_old_box_and_resets_review_status():
    annotations = [
        {
            "id": "pocket-1",
            "image": "/tmp/1.jpg",
            "query_text": "这条裤子右侧的口袋",
            "target_region": "pocket",
            "target_bbox": [10, 20, 30, 40],
            "label_status": "labeled",
        }
    ]
    eval_records = [
        {
            "id": "pocket-1",
            "image": "/tmp/1.jpg",
            "query_text": "这条裤子右侧的口袋",
            "target_region": "pocket",
            "manual_bbox_iou": 0.05,
        },
        {
            "id": "hem-1",
            "image": "/tmp/2.jpg",
            "query_text": "这条裤子的裤脚",
            "target_region": "hem",
            "manual_bbox_iou": 0.0,
        },
    ]

    records, summary = build_audit_records(
        annotations,
        eval_records,
        iou_threshold=0.3,
        regions={"pocket"},
    )

    assert len(records) == 1
    assert records[0]["target_bbox"] == [10, 20, 30, 40]
    assert records[0]["audit_original_target_bbox"] == [10, 20, 30, 40]
    assert records[0]["label_status"] == "unlabeled"
    assert records[0]["audit_instruction"] == AUDIT_INSTRUCTION
    assert summary["num_missing_annotation_keys"] == 0


def test_select_manual_failure_records_filters_region_and_iou():
    records = [
        {"manual_bbox_iou": 0.05, "target_region": "cuff", "image": "/tmp/2.jpg"},
        {"manual_bbox_iou": 0.2, "target_region": "cuff", "image": "/tmp/1.jpg"},
        {"manual_bbox_iou": 0.0, "target_region": "hem", "image": "/tmp/3.jpg"},
    ]

    selected = select_failure_records(
        records,
        iou_threshold=0.1,
        regions={"cuff"},
    )

    assert len(selected) == 1
    assert selected[0]["target_region"] == "cuff"
    assert selected[0]["manual_bbox_iou"] == 0.05


def test_failure_safe_stem_removes_path_punctuation():
    assert safe_stem("abc/def ghi") == "abc_def_ghi"


def test_write_failure_review_html_uses_relative_visualization_path(tmp_path):
    image = tmp_path / "000_cuff_iou0.000_case.jpg"
    image.write_bytes(b"fake")
    output = tmp_path / "failure_review.html"

    write_failure_review_html(
        {
            "iou_threshold": 0.1,
            "num_exported_cases": 1,
            "num_input_records": 2,
            "cases": [
                {
                    "id": "case/1",
                    "query_text": "左边的袖口",
                    "target_region": "cuff",
                    "selected_region": "left_cuff",
                    "manual_bbox_iou": 0.0,
                    "visualization": str(image),
                }
            ],
        },
        output,
    )

    html = output.read_text(encoding="utf-8")
    assert 'src="000_cuff_iou0.000_case.jpg"' in html
    assert "左边的袖口" in html
    assert "case/1" in html


def test_failure_export_preserves_grounding_provenance(tmp_path):
    image_path = tmp_path / "pocket.jpg"
    Image.new("RGB", (80, 100), color="white").save(image_path)
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "pocket-1",
                        "image": str(image_path),
                        "query_text": "右侧的口袋",
                        "target_region": "pocket",
                        "selected_region": "right pocket",
                        "target_bbox": [10, 10, 25, 25],
                        "predicted_bbox": [40, 40, 55, 55],
                        "manual_bbox_iou": 0.0,
                        "gated_policy_route": "grounding",
                        "ranker_backend": "gated_hybrid_grounding_auto",
                        "grounding_model_name": "IDEA-Research/grounding-dino-base",
                        "prompt_profile": "ensemble",
                        "grounding_score_threshold": 0.15,
                        "score": 0.42,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "failures"
    summary = export_failure_cases(eval_path, output_dir, iou_threshold=0.3)
    output = output_dir / "failure_review.html"
    write_failure_review_html(summary, output)

    case = summary["cases"][0]
    assert case["grounding_model_name"] == "IDEA-Research/grounding-dino-base"
    assert case["prompt_profile"] == "ensemble"
    assert case["grounding_score_threshold"] == 0.15
    assert case["score"] == 0.42

    html = output.read_text(encoding="utf-8")
    assert "IDEA-Research/grounding-dino-base" in html
    assert "detection score: 0.420" in html


def test_policy_delta_pairs_only_material_grounding_changes(tmp_path):
    image_path = tmp_path / "000001.jpg"
    Image.new("RGB", (80, 100), color="white").save(image_path)
    baseline = [
        {
            "id": "pocket-1",
            "image": str(image_path),
            "query_text": "右侧的口袋",
            "target_region": "pocket",
            "target_bbox": [20, 20, 40, 40],
            "predicted_bbox": [0, 0, 10, 10],
            "selected_region": "right_pocket",
            "manual_bbox_iou": 0.0,
        },
        {
            "id": "hem-1",
            "image": str(image_path),
            "query_text": "衣服下方的下摆",
            "target_region": "hem",
            "target_bbox": [20, 60, 60, 90],
            "predicted_bbox": [20, 60, 60, 90],
            "manual_bbox_iou": 1.0,
        },
    ]
    candidate = [
        {
            **baseline[0],
            "predicted_bbox": [20, 20, 40, 40],
            "selected_region": "right pocket",
            "manual_bbox_iou": 1.0,
            "gated_policy_route": "grounding",
            "score": 0.42,
            "prompts": ["right pocket", "pocket"],
        },
        {
            **baseline[1],
            "gated_policy_route": "heuristic",
        },
    ]

    pairs = paired_policy_deltas(
        baseline,
        candidate,
        regions={"pocket"},
        candidate_routes={"grounding"},
        min_abs_delta=0.2,
    )

    assert len(pairs) == 1
    assert pairs[0]["change"] == "improved"
    assert pairs[0]["iou_delta"] == pytest.approx(1.0)

    baseline_json = tmp_path / "baseline.json"
    candidate_json = tmp_path / "candidate.json"
    baseline_json.write_text(json.dumps({"records": baseline}), encoding="utf-8")
    candidate_json.write_text(json.dumps({"records": candidate}), encoding="utf-8")
    output_dir = tmp_path / "review"
    summary = export_policy_deltas(
        baseline_json,
        candidate_json,
        output_dir,
        regions={"pocket"},
        candidate_routes={"grounding"},
        min_abs_delta=0.2,
    )
    html_path = output_dir / "policy_delta_review.html"
    write_policy_delta_html(summary, html_path)

    assert summary["num_exported_cases"] == 1
    assert (output_dir / "policy_delta_summary.json").exists()
    assert list(output_dir.glob("*.jpg"))
    html = html_path.read_text(encoding="utf-8")
    assert 'src="000_improved_pocket_delta+1.000_pocket-1.jpg"' in html
    assert "right pocket" in html


def test_routing_oracle_reports_per_record_upper_bound():
    baseline = [
        {"id": "pattern-1", "target_region": "pattern", "manual_bbox_iou": 0.1},
        {"id": "pocket-1", "target_region": "pocket", "manual_bbox_iou": 0.6},
        {"id": "hem-1", "target_region": "hem", "manual_bbox_iou": 0.4},
    ]
    candidate = [
        {"id": "pattern-1", "target_region": "pattern", "manual_bbox_iou": 0.8},
        {"id": "pocket-1", "target_region": "pocket", "manual_bbox_iou": 0.2},
        {"id": "hem-1", "target_region": "hem", "manual_bbox_iou": 0.4},
    ]

    result = build_routing_oracle(baseline, candidate)

    oracle = result["per_record_oracle"]
    assert oracle["summary"]["avg_manual_bbox_iou"] == pytest.approx(0.6)
    assert oracle["summary"]["manual_hit_at"]["0.3"] == pytest.approx(1.0)
    assert oracle["source_counts"] == {"baseline": 2, "candidate": 1}
    assert oracle["by_region"]["pattern"]["source_counts"] == {"candidate": 1}


def test_chinese_clip_manual_summary_handles_empty_prediction():
    record = empty_prediction_record(
        {
            "id": "pocket-1",
            "image": "/tmp/image.jpg",
            "query_text": "右侧的口袋",
            "target_region": "pocket",
            "target_bbox": [1, 2, 3, 4],
        },
        "pocket",
        "no_garment_instance",
    )

    summary = finalize_chinese_clip_manual_run([record])

    assert record["manual_bbox_iou"] == 0.0
    assert summary["status_counts"] == {"no_garment_instance": 1}
    assert summary["manual_hit_at"]["0.3"] == 0.0
