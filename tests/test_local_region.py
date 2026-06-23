import numpy as np
from PIL import Image

from fashion_mm.models.instance_segmentation import FashionInstance
from fashion_mm.models.instance_segmentation import SegmentationResult
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.models.local_region import parse_region_query
from fashion_mm.models.local_region import propose_local_region
from fashion_mm.models.local_region import select_garment_instance


def test_parse_region_query_extracts_region_and_garment_hint():
    parsed = parse_region_query("这件连衣裙的领口是什么设计")

    assert parsed.region == "neckline"
    assert parsed.garment_hint == "dress"
    assert parsed.is_supported_region is True


def test_parse_region_query_marks_unsupported_region():
    parsed = parse_region_query("这个口袋的设计")

    assert parsed.region == "pocket"
    assert parsed.is_supported_region is True


def test_parse_region_query_collects_open_vocab_hints():
    parsed = parse_region_query("外套里面的内搭有碎花图案吗")

    assert parsed.region == "pattern"
    assert parsed.garment_hint == "outerwear"
    assert parsed.attribute_hints == ("floral",)
    assert parsed.relation_hints == ("outer", "inner")


def test_propose_neckline_region_is_clipped_by_garment_mask():
    garment_mask = np.zeros((100, 80), dtype=bool)
    garment_mask[10:90, 20:60] = True

    proposal = propose_local_region(
        garment_mask,
        garment_box=(20, 10, 60, 90),
        region="neckline",
    )

    assert proposal.status == "ok"
    assert proposal.box is not None
    assert proposal.box[1] < 25
    assert proposal.box[3] <= 25
    assert proposal.mask.sum() > 0
    assert np.logical_and(proposal.mask, ~garment_mask).sum() == 0


def test_propose_unsupported_region_returns_structured_fallback():
    garment_mask = np.ones((20, 20), dtype=bool)

    proposal = propose_local_region(
        garment_mask,
        garment_box=(0, 0, 20, 20),
        region="pocket",
    )

    assert proposal.status == "unsupported_region"
    assert proposal.box is None
    assert proposal.mask.sum() == 0


def test_select_garment_instance_prefers_query_hint():
    top = FashionInstance(
        mask=np.ones((10, 10), dtype=bool),
        box=(0.0, 0.0, 10.0, 10.0),
        label_id=1,
        label_name="top",
        score=0.99,
    )
    dress = FashionInstance(
        mask=np.ones((10, 10), dtype=bool),
        box=(0.0, 0.0, 10.0, 10.0),
        label_id=5,
        label_name="dress",
        score=0.80,
    )
    segmentation = SegmentationResult(image_size=(10, 10), instances=[top, dress])

    selected = select_garment_instance(
        segmentation,
        parse_region_query("这件连衣裙的领口"),
    )

    assert selected is dress


def test_localize_region_from_instances_returns_region_result():
    mask = np.zeros((100, 80), dtype=bool)
    mask[10:90, 20:60] = True
    instance = FashionInstance(
        mask=mask,
        box=(20.0, 10.0, 60.0, 90.0),
        label_id=1,
        label_name="top",
        score=0.95,
    )
    segmentation = SegmentationResult(image_size=(80, 100), instances=[instance])

    result = localize_region_from_instances(segmentation, "这件上衣的领口")

    assert result.status == "ok"
    assert result.selected_instance is instance
    assert result.proposal is not None
    assert result.proposal.proposal.region == "neckline"
    assert result.proposal.proposal.mask.sum() > 0
    assert result.ranker_backend == "heuristic_text_region_ranker"
    assert result.candidates


def test_localize_region_from_instances_supports_open_attribute_query():
    mask = np.zeros((100, 80), dtype=bool)
    mask[10:90, 20:60] = True
    instance = FashionInstance(
        mask=mask,
        box=(20.0, 10.0, 60.0, 90.0),
        label_id=1,
        label_name="top",
        score=0.95,
    )
    segmentation = SegmentationResult(image_size=(80, 100), instances=[instance])

    result = localize_region_from_instances(segmentation, "这件衣服上的碎花图案")

    assert result.status == "ok"
    assert result.proposal is not None
    assert result.proposal.proposal.region == "pattern"
    assert result.proposal.proposal.mask.sum() > 0


def test_localize_region_from_instances_uses_spatial_words_for_cuff_query():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:90] = True
    instance = FashionInstance(
        mask=mask,
        box=(10.0, 10.0, 90.0, 90.0),
        label_id=1,
        label_name="top",
        score=0.95,
    )
    segmentation = SegmentationResult(image_size=(100, 100), instances=[instance])

    result = localize_region_from_instances(segmentation, "左边的袖口")

    assert result.status == "ok"
    assert result.proposal is not None
    assert result.proposal.proposal.region == "left_cuff"
    assert result.proposal.proposal.box is not None
    assert result.proposal.proposal.box[0] < 40


def test_localize_region_from_instances_supports_pocket_query():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:90] = True
    instance = FashionInstance(
        mask=mask,
        box=(10.0, 10.0, 90.0, 90.0),
        label_id=1,
        label_name="top",
        score=0.95,
    )
    segmentation = SegmentationResult(image_size=(100, 100), instances=[instance])

    result = localize_region_from_instances(segmentation, "右侧的口袋")

    assert result.status == "ok"
    assert result.proposal is not None
    assert result.proposal.proposal.region == "right_pocket"
    assert result.proposal.proposal.box is not None
    assert result.proposal.proposal.box[0] > 50


def test_localize_region_from_instances_supports_zipper_query():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:90] = True
    instance = FashionInstance(
        mask=mask,
        box=(10.0, 10.0, 90.0, 90.0),
        label_id=4,
        label_name="outerwear",
        score=0.95,
    )
    segmentation = SegmentationResult(image_size=(100, 100), instances=[instance])

    result = localize_region_from_instances(segmentation, "这件外套的拉链")

    assert result.status == "ok"
    assert result.proposal is not None
    assert result.proposal.proposal.region == "zipper"
    assert result.proposal.proposal.box is not None
    assert 40 <= result.proposal.proposal.box[0] <= 50


def test_local_region_eval_collects_visible_images(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "evaluate_local_region_queries.py"
    )
    spec = importlib.util.spec_from_file_location("local_region_eval", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    Image.new("RGB", (4, 4)).save(tmp_path / "000001.jpg")
    Image.new("RGB", (4, 4)).save(tmp_path / "000002.png")
    (tmp_path / "._000001.jpg").write_text("metadata", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    image_paths = module.collect_images(tmp_path, max_images=None)

    assert [path.name for path in image_paths] == ["000001.jpg", "000002.png"]


def test_local_region_eval_summarizes_records():
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "evaluate_local_region_queries.py"
    )
    spec = importlib.util.spec_from_file_location("local_region_eval", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    summary = module.summarize_records(
        [
            {
                "status": "ok",
                "latency_ms": 2.0,
                "ranker_backend": "heuristic",
                "region": {"region": "pattern", "match_score": 1.5},
            },
            {
                "status": "ok",
                "latency_ms": 4.0,
                "ranker_backend": "heuristic",
                "region": {"region": "left_cuff", "match_score": 2.5},
            },
        ]
    )

    assert summary["num_records"] == 2
    assert summary["status_counts"] == {"ok": 2}
    assert summary["selected_region_counts"] == {"pattern": 1, "left_cuff": 1}
    assert summary["avg_local_region_latency_ms"] == 3.0
