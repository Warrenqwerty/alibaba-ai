import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.build_local_region_manual_eval_manifest import (
    build_class_aware_manifest_records,
    build_manifest_records,
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
    summarize_records as summarize_pretrained_grounding_records,
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
    ) == ("/tmp/1.jpg", "这件衣服的领口", "neckline")


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
