import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.build_local_region_manual_eval_manifest import (
    build_manifest_records,
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
