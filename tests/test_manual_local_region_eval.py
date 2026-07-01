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
