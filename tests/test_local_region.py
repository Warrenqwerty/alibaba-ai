import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace
import json

from fashion_mm.models.instance_segmentation import FashionInstance
from fashion_mm.models.instance_segmentation import SegmentationResult
from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import box_context_features
from fashion_mm.models.local_region import build_pair_feature
from fashion_mm.models.local_region import build_candidate_record_feature
from fashion_mm.models.local_region import candidate_boxes_from_garment
from fashion_mm.models.local_region import CandidateListwiseScorer
from fashion_mm.models.local_region import candidate_prior_features
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


def test_learned_region_ranker_uses_candidate_listwise_checkpoint_for_hem(tmp_path):
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
    checkpoint_path = tmp_path / "candidate_ranker.pt"
    model = CandidateListwiseScorer(num_buckets=16, hidden_dim=32)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_buckets": 16,
            "hidden_dim": 32,
            "loss": "soft",
            "softmax_temperature": 0.08,
        },
        checkpoint_path,
    )

    ranker = LearnedRegionRanker(checkpoint_path, device="cpu")
    hem_result = localize_region_from_instances(
        segmentation,
        "衣服下方的下摆",
        ranker=ranker,
    )
    shoulder_result = localize_region_from_instances(
        segmentation,
        "这件衣服的肩部",
        ranker=ranker,
    )
    pocket_result = localize_region_from_instances(
        segmentation,
        "右侧的口袋",
        ranker=ranker,
    )

    assert ranker.checkpoint_kind == "candidate_listwise"
    assert hem_result.ranker_backend == "hybrid_candidate_listwise_context_ranker"
    assert hem_result.proposal is not None
    assert hem_result.proposal.backend == "hybrid_candidate_listwise_context_ranker"
    assert "listwise candidate" in hem_result.proposal.reason
    assert shoulder_result.proposal is not None
    assert shoulder_result.ranker_backend == "heuristic_text_region_ranker"
    assert shoulder_result.proposal.backend == "heuristic_text_region_ranker"
    assert "heuristic fallback" in shoulder_result.proposal.reason
    assert pocket_result.proposal is not None
    assert pocket_result.ranker_backend == "heuristic_text_region_ranker"
    assert pocket_result.proposal.backend == "heuristic_text_region_ranker"
    assert pocket_result.proposal.proposal.region == "right_pocket"
    assert "heuristic fallback" in pocket_result.proposal.reason


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


def test_build_local_region_candidate_records_exports_iou_labels(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "data"
        / "build_local_region_candidate_records.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_local_region_candidate_records",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    input_path = tmp_path / "queries.jsonl"
    output_path = tmp_path / "candidates.jsonl"
    record = {
        "image": "/data/image/000001.jpg",
        "annotation": "/data/annos/000001.json",
        "item_key": "item1",
        "query": "这件衣服的领口",
        "region": "neckline",
        "garment_box": [10.0, 20.0, 110.0, 220.0],
        "region_box": [26.0, 20.0, 94.0, 64.0],
        "source": "landmark_pseudo_label",
        "confidence": 0.9,
        "category_id": 1,
        "category_name": "top",
    }
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = module.build_candidate_records(
        input_path,
        output_path,
        max_records=None,
        skip_records=0,
        positive_iou_threshold=0.9,
    )
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert summary["num_query_records"] == 1
    assert summary["num_candidate_records"] == len(
        candidate_boxes_from_garment((10.0, 20.0, 110.0, 220.0))
    )
    assert summary["label_counts"]["1"] == 1
    positives = [row for row in rows if row["label"] == 1]
    assert len(positives) == 1
    assert positives[0]["candidate_region"] == "neckline"
    assert positives[0]["iou"] == 1.0


def test_chinese_clip_local_region_ranker_groups_candidates(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "evaluate_chinese_clip_local_region_ranker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_chinese_clip_local_region_ranker",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (20, 20), color=(120, 80, 40)).save(image_path)
    candidates_path = tmp_path / "candidates.jsonl"
    rows = [
        {
            "image": str(image_path),
            "annotation": "/data/annos/000001.json",
            "item_key": "item1",
            "query": "这件衣服的领口",
            "target_region": "neckline",
            "target_region_box": [2.0, 0.0, 18.0, 5.0],
            "garment_box": [0.0, 0.0, 20.0, 20.0],
            "candidate_region": "neckline",
            "candidate_box": [2.0, 0.0, 18.0, 5.0],
            "iou": 1.0,
            "label": 1,
            "weak_label_source": "landmark_pseudo_label",
            "weak_label_confidence": 0.9,
            "category_id": 1,
            "category_name": "top",
        },
        {
            "image": str(image_path),
            "annotation": "/data/annos/000001.json",
            "item_key": "item1",
            "query": "这件衣服的领口",
            "target_region": "neckline",
            "target_region_box": [2.0, 0.0, 18.0, 5.0],
            "garment_box": [0.0, 0.0, 20.0, 20.0],
            "candidate_region": "hem",
            "candidate_box": [0.0, 15.0, 20.0, 20.0],
            "iou": 0.0,
            "label": 0,
            "weak_label_source": "landmark_pseudo_label",
            "weak_label_confidence": 0.9,
            "category_id": 1,
            "category_name": "top",
        },
    ]
    candidates_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    groups = list(module.iter_candidate_groups(candidates_path))
    crop = module.crop_candidate(Image.open(image_path), (-5.0, -2.0, 10.0, 10.0))

    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert groups[0][0].candidate_region == "neckline"
    assert crop.size == (10, 10)


def test_chinese_clip_feature_output_accepts_pooler_output():
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "evaluate_chinese_clip_local_region_ranker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_chinese_clip_local_region_ranker",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    output = SimpleNamespace(pooler_output=torch.ones(2, 4))

    assert torch.equal(module._as_feature_tensor(output), torch.ones(2, 4))


def test_chinese_clip_region_prior_selects_parsed_region():
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "evaluate_chinese_clip_local_region_ranker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_chinese_clip_local_region_ranker",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    scored_group = {
        "group": [
            SimpleNamespace(
                query="这件衣服的领口",
                target_region="neckline",
                candidate_region="right",
                iou=0.1,
            ),
            SimpleNamespace(
                query="这件衣服的领口",
                target_region="neckline",
                candidate_region="neckline",
                iou=0.8,
            ),
        ],
        "base_scores": torch.tensor([0.50, 0.48]),
        "prior_scores": torch.tensor([0.0, 1.0]),
        "parsed_region": "neckline",
    }

    without_prior = module.select_scored_candidate(scored_group, 0.0)
    with_prior = module.select_scored_candidate(scored_group, 0.05)

    assert module.parse_region_prior_weights("0,0.05") == (0.0, 0.05)
    assert without_prior["selected_region"] == "right"
    assert with_prior["selected_region"] == "neckline"
    assert module.candidate_matches_parsed_region("left_cuff", "cuff") is True


def test_candidate_listwise_feature_includes_parser_prior():
    feature = build_candidate_record_feature(
        "这件衣服的领口",
        "neckline",
        (0.0, 0.0, 100.0, 200.0),
        (16.0, 0.0, 84.0, 44.0),
        "neckline",
        category_text="top",
        num_buckets=16,
    )
    model = CandidateListwiseScorer(num_buckets=16, hidden_dim=32)
    logits = model(feature.unsqueeze(0))

    assert feature.shape == (16 * 3 + 6 + 8 + 3,)
    assert len(box_context_features((0.0, 0.0, 100.0, 200.0), (0.0, 0.0, 50.0, 20.0))) == 8
    assert candidate_prior_features("left_cuff", "cuff") == (0.0, 1.0, 0.0)
    assert logits.shape == (1,)


def test_candidate_listwise_trainer_builds_best_iou_target(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train"
        / "train_candidate_local_region_ranker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_candidate_local_region_ranker",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    candidates_path = tmp_path / "candidates.jsonl"
    rows = [
        {
            "image": "/data/image/000001.jpg",
            "annotation": "/data/annos/000001.json",
            "item_key": "item1",
            "query": "这件衣服的领口",
            "target_region": "neckline",
            "target_region_box": [2.0, 0.0, 18.0, 5.0],
            "garment_box": [0.0, 0.0, 20.0, 20.0],
            "candidate_region": "neckline",
            "candidate_box": [2.0, 0.0, 18.0, 5.0],
            "iou": 0.4,
            "label": 0,
            "weak_label_source": "landmark_pseudo_label",
            "weak_label_confidence": 0.9,
        },
        {
            "image": "/data/image/000001.jpg",
            "annotation": "/data/annos/000001.json",
            "item_key": "item1",
            "query": "这件衣服的领口",
            "target_region": "neckline",
            "target_region_box": [2.0, 0.0, 18.0, 5.0],
            "garment_box": [0.0, 0.0, 20.0, 20.0],
            "candidate_region": "whole_garment",
            "candidate_box": [0.0, 0.0, 20.0, 20.0],
            "iou": 0.8,
            "label": 1,
            "weak_label_source": "landmark_pseudo_label",
            "weak_label_confidence": 0.9,
        },
    ]
    candidates_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    group = next(module.iter_candidate_groups(candidates_path))
    features, target_index = module.build_group_training_example(group, num_buckets=16)
    _, soft_target = module.build_group_training_example(
        group,
        num_buckets=16,
        loss_mode="soft",
        softmax_temperature=0.08,
    )

    assert features.shape == (2, 16 * 3 + 6 + 8 + 3)
    assert target_index == 1
    assert isinstance(soft_target, torch.Tensor)
    assert torch.isclose(soft_target.sum(), torch.tensor(1.0))
    assert int(torch.argmax(soft_target)) == 1


def test_candidate_listwise_trainer_eval_only_loads_checkpoint(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train"
        / "train_candidate_local_region_ranker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_candidate_local_region_ranker",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    candidates_path = tmp_path / "candidates.jsonl"
    rows = [
        {
            "image": "/data/image/000001.jpg",
            "annotation": "/data/annos/000001.json",
            "item_key": "item1",
            "query": "这件衣服的领口",
            "target_region": "neckline",
            "target_region_box": [2.0, 0.0, 18.0, 5.0],
            "garment_box": [0.0, 0.0, 20.0, 20.0],
            "candidate_region": "neckline",
            "candidate_box": [2.0, 0.0, 18.0, 5.0],
            "iou": 0.8,
            "label": 1,
            "weak_label_source": "landmark_pseudo_label",
            "weak_label_confidence": 0.9,
        },
        {
            "image": "/data/image/000001.jpg",
            "annotation": "/data/annos/000001.json",
            "item_key": "item1",
            "query": "这件衣服的领口",
            "target_region": "neckline",
            "target_region_box": [2.0, 0.0, 18.0, 5.0],
            "garment_box": [0.0, 0.0, 20.0, 20.0],
            "candidate_region": "whole_garment",
            "candidate_box": [0.0, 0.0, 20.0, 20.0],
            "iou": 0.4,
            "label": 0,
            "weak_label_source": "landmark_pseudo_label",
            "weak_label_confidence": 0.9,
        },
    ]
    candidates_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "ranker.pt"
    metrics_path = tmp_path / "metrics.json"
    model = CandidateListwiseScorer(num_buckets=16, hidden_dim=32)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_buckets": 16,
            "hidden_dim": 32,
        },
        checkpoint_path,
    )
    groups = list(module.iter_candidate_groups(candidates_path))
    loaded = module.load_candidate_ranker(checkpoint_path, torch.device("cpu"))
    metrics = module.evaluate_ranker(loaded[0], groups, loaded[1], torch.device("cpu"))
    module.write_metrics(metrics, metrics_path)

    assert loaded[1] == 16
    assert metrics["num_records"] == 1
    assert metrics_path.exists()


def test_candidate_baseline_evaluator_reports_oracle_and_name_baselines(tmp_path):
    import importlib.util
    from pathlib import Path

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "evaluate_local_region_candidate_baselines.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_local_region_candidate_baselines",
        script_path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    candidates_path = tmp_path / "candidates.jsonl"
    rows = [
        {
            "image": "/data/image/000001.jpg",
            "annotation": "/data/annos/000001.json",
            "item_key": "item1",
            "query": "这件衣服的领口",
            "target_region": "neckline",
            "target_region_box": [2.0, 0.0, 18.0, 5.0],
            "garment_box": [0.0, 0.0, 20.0, 20.0],
            "candidate_region": "neckline",
            "candidate_box": [2.0, 0.0, 18.0, 5.0],
            "iou": 0.8,
            "label": 1,
            "weak_label_source": "landmark_pseudo_label",
            "weak_label_confidence": 0.9,
        },
        {
            "image": "/data/image/000001.jpg",
            "annotation": "/data/annos/000001.json",
            "item_key": "item1",
            "query": "这件衣服的领口",
            "target_region": "neckline",
            "target_region_box": [2.0, 0.0, 18.0, 5.0],
            "garment_box": [0.0, 0.0, 20.0, 20.0],
            "candidate_region": "upper",
            "candidate_box": [0.0, 0.0, 20.0, 10.0],
            "iou": 0.4,
            "label": 0,
            "weak_label_source": "landmark_pseudo_label",
            "weak_label_confidence": 0.9,
        },
    ]
    candidates_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    metrics = module.evaluate_candidate_baselines(
        candidates_path,
        max_groups=None,
        skip_groups=0,
    )

    assert metrics["num_groups"] == 1
    assert metrics["baselines"]["oracle_best_iou"]["avg_top1_iou"] == 0.8
    assert metrics["baselines"]["target_region_name"]["avg_top1_iou"] == 0.8


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
                "ranker_backend": "heuristic",
                "parsed_region": "neckline",
                "weak_label_source": "landmark_pseudo_label",
                "garment_iou": 0.8,
                "weak_iou": 0.6,
            },
            {
                "status": "ok",
                "ranker_backend": "hybrid",
                "parsed_region": "neckline",
                "weak_label_source": "rule_baseline",
                "garment_iou": 0.6,
                "weak_iou": 0.2,
            },
            {
                "status": "ok",
                "ranker_backend": "hybrid",
                "parsed_region": "hem",
                "weak_label_source": "landmark_pseudo_label",
                "garment_iou": 0.7,
                "weak_iou": 0.4,
            },
        ]
    )

    assert summary["num_records"] == 3
    assert summary["status_counts"] == {"ok": 3}
    assert summary["ranker_backend_counts"] == {"heuristic": 1, "hybrid": 2}
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
    assert result.ranker_backend == "heuristic_text_region_ranker"
    assert result.proposal is not None
    assert result.proposal.backend == "heuristic_text_region_ranker"
    assert result.proposal.proposal.region == "right_pocket"
    assert "heuristic fallback" in result.proposal.reason


def test_learned_ranker_falls_back_for_shoulder_after_weak_eval_drop(tmp_path):
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
        "这件衣服的肩部",
        ranker=ranker,
    )

    assert result.status == "ok"
    assert result.proposal is not None
    assert result.ranker_backend == "heuristic_text_region_ranker"
    assert result.proposal.backend == "heuristic_text_region_ranker"
    assert result.proposal.proposal.region == "shoulder"
    assert "heuristic fallback" in result.proposal.reason
