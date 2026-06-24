import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

from fashion_mm.models.instance_segmentation import FashionInstance
from fashion_mm.models.instance_segmentation import SegmentationResult
from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import build_pair_feature
from fashion_mm.models.local_region import candidate_boxes_from_garment
from fashion_mm.models.local_region import HashingTextRegionScorer
from fashion_mm.models.local_region import LearnedRegionRanker
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
    assert proposal.box[3] <= 30
    assert proposal.mask.sum() > 0
    assert np.logical_and(proposal.mask, ~garment_mask).sum() == 0


def test_propose_shoulder_region_uses_shallow_upper_band():
    garment_mask = np.zeros((100, 100), dtype=bool)
    garment_mask[10:90, 10:90] = True

    proposal = propose_local_region(
        garment_mask,
        garment_box=(10, 10, 90, 90),
        region="shoulder",
    )

    assert proposal.status == "ok"
    assert proposal.box is not None
    assert proposal.box[1] == 10
    assert proposal.box[3] <= 30
    assert proposal.box[0] == 10
    assert proposal.box[2] == 90
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


def test_local_region_weak_eval_helpers(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "evaluate_local_region_weak_labels.py"
    )
    spec = importlib.util.spec_from_file_location("local_region_weak_eval", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    anno_dir = tmp_path / "annos"
    image_dir = tmp_path / "images"
    anno_dir.mkdir()
    image_dir.mkdir()
    (anno_dir / "000002.json").write_text("{}", encoding="utf-8")
    (anno_dir / "._000001.json").write_text("{}", encoding="utf-8")
    Image.new("RGB", (10, 10)).save(image_dir / "000002.jpg")

    annotations = module.collect_annotations(anno_dir, max_images=None)
    assert [path.name for path in annotations] == ["000002.json"]

    image_path = module.image_path_for_annotation(
        image_dir,
        anno_dir / "000002.json",
        {},
    )
    assert image_path.name == "000002.jpg"

    mask = module.polygon_to_mask([[1, 1, 5, 1, 5, 5, 1, 5]], (10, 10))
    assert mask.sum() > 0
    assert module.mask_iou(mask, mask) == 1.0


def test_local_region_weak_eval_summarizes_records():
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "evaluate_local_region_weak_labels.py"
    )
    spec = importlib.util.spec_from_file_location("local_region_weak_eval", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    summary = module.summarize_records(
        [
            {
                "status": "ok",
                "parsed_region": "neckline",
                "weak_label_source": "landmark_pseudo_label",
                "garment_iou": 0.8,
                "weak_iou": 0.6,
            },
            {
                "status": "ok",
                "parsed_region": "neckline",
                "weak_label_source": "rule_baseline",
                "garment_iou": 0.6,
                "weak_iou": 0.2,
            },
            {
                "status": "ok",
                "parsed_region": "hem",
                "weak_label_source": "landmark_pseudo_label",
                "garment_iou": 0.7,
                "weak_iou": 0.4,
            },
        ]
    )

    assert summary["num_records"] == 3
    assert summary["status_counts"] == {"ok": 3}
    assert summary["weak_label_source_counts"] == {
        "landmark_pseudo_label": 2,
        "rule_baseline": 1,
    }
    assert summary["avg_garment_iou"] == 0.7
    assert summary["avg_weak_iou"] == 0.4
    assert summary["weak_hit_at"]["0.3"] == 2 / 3
    assert summary["by_region"]["neckline"]["weak_label_source_counts"] == {
        "landmark_pseudo_label": 1,
        "rule_baseline": 1,
    }
    assert summary["by_region"]["neckline"]["avg_garment_iou"] == 0.7
    assert summary["by_region"]["neckline"]["avg_weak_iou"] == 0.4


def test_build_deepfashion2_local_region_query_records(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "data"
        / "build_deepfashion2_local_region_queries.py"
    )
    spec = importlib.util.spec_from_file_location("local_region_query_build", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    image_path = tmp_path / "000001.jpg"
    annotation_path = tmp_path / "000001.json"
    Image.new("RGB", (100, 100)).save(image_path)
    annotation = {
        "item1": {
            "category_id": 1,
            "category_name": "short sleeve top",
            "bounding_box": [20, 10, 80, 90],
            "segmentation": [[20, 10, 80, 10, 80, 90, 20, 90]],
            "landmarks": [
                30,
                20,
                2,
                40,
                20,
                2,
                50,
                20,
                2,
                60,
                20,
                2,
                70,
                20,
                2,
                75,
                20,
                2,
            ],
        }
    }

    records = module.build_records_for_annotation(
        image_path,
        annotation_path,
        annotation,
        ["neckline"],
    )

    assert len(records) == 3
    assert {record["query"] for record in records} == set(
        module.QUERY_TEMPLATES["neckline"]
    )
    assert records[0]["region"] == "neckline"
    assert records[0]["source"] == "landmark_pseudo_label"
    assert records[0]["region_box"]


def test_learned_ranker_candidate_features_and_iou():
    garment_box = (10.0, 20.0, 110.0, 220.0)
    candidates = candidate_boxes_from_garment(garment_box)
    candidate_by_region = {candidate.region: candidate for candidate in candidates}

    assert "neckline" in candidate_by_region
    assert "shoulder" in candidate_by_region
    assert box_iou(
        candidate_by_region["neckline"].box,
        candidate_by_region["neckline"].box,
    ) == 1.0

    feature = build_pair_feature(
        "这件衣服的领口",
        candidate_by_region["neckline"],
        garment_box,
        num_buckets=32,
    )

    assert feature.shape[0] == 32 * 2 + 6
    assert float(feature.sum()) > 0.0


def test_train_local_region_ranker_builds_examples():
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train"
        / "train_local_region_ranker.py"
    )
    spec = importlib.util.spec_from_file_location("local_region_ranker_train", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    record = SimpleNamespace(
        query="这件衣服的领口",
        garment_box=(0.0, 0.0, 100.0, 200.0),
        region_box=(16.0, 0.0, 84.0, 44.0),
    )
    examples = list(module.build_training_examples(record, num_buckets=32))

    assert len(examples) > 1
    assert examples[0][0].shape[0] == 32 * 2 + 6
    assert max(target for _, target in examples) == 1.0


def test_train_local_region_ranker_stream_limit_preserves_train_count():
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train"
        / "train_local_region_ranker.py"
    )
    spec = importlib.util.spec_from_file_location("local_region_ranker_train", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._train_stream_limit(50000, 2000, 0) == 52000
    assert module._train_stream_limit(500000, 10000, 500000) == 500000


def test_localize_region_from_instances_accepts_learned_ranker(tmp_path):
    checkpoint_path = tmp_path / "ranker.pt"
    model = HashingTextRegionScorer(num_buckets=32, hidden_dim=16)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_buckets": 32,
            "hidden_dim": 16,
        },
        checkpoint_path,
    )
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

    ranker = LearnedRegionRanker(checkpoint_path, device="cpu")
    result = localize_region_from_instances(
        segmentation,
        "这件衣服的领口",
        ranker=ranker,
    )

    assert result.status == "ok"
    assert result.ranker_backend == "hybrid_learned_hash_text_geometry_ranker"
    assert result.proposal is not None
    assert result.candidates


def test_learned_ranker_falls_back_for_untrained_open_query(tmp_path):
    checkpoint_path = tmp_path / "ranker.pt"
    model = HashingTextRegionScorer(num_buckets=32, hidden_dim=16)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_buckets": 32,
            "hidden_dim": 16,
        },
        checkpoint_path,
    )
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

    ranker = LearnedRegionRanker(checkpoint_path, device="cpu")
    result = localize_region_from_instances(
        segmentation,
        "右侧的口袋",
        ranker=ranker,
    )

    assert result.status == "ok"
    assert result.ranker_backend == "hybrid_learned_hash_text_geometry_ranker"
    assert result.proposal is not None
    assert result.proposal.proposal.region == "right_pocket"
    assert "heuristic fallback" in result.proposal.reason
