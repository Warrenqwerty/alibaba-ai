import json
import sys
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

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
from scripts.eval.evaluate_gated_hybrid_manual_labels import should_route_to_grounding
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
from scripts.eval.evaluate_chinese_clip_manual_local_regions import (
    empty_prediction_record,
    finalize_run as finalize_chinese_clip_manual_run,
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
