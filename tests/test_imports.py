import json
from pathlib import Path

import numpy as np
from PIL import Image

from fashion_mm.data_loaders import DeepFashion2Dataset
from fashion_mm.data_loaders import iter_local_region_query_records
from fashion_mm.data_loaders import LocalRegionQueryDataset
from fashion_mm.data_loaders import build_balanced_sampler
from fashion_mm.data_loaders.deepfashion2 import DEEPFASHION2_TO_PROJECT_CATEGORY
from fashion_mm.data_loaders.sampling import build_hard_case_weights
from fashion_mm.models.instance_segmentation import FashionInstance, SegmentationResult
from fashion_mm.utils.image_io import load_rgb_image


def test_segmentation_result_serializes_without_masks_by_default():
    instance = FashionInstance(
        mask=np.array([[0, 1], [1, 1]], dtype=bool),
        box=(0.0, 0.0, 2.0, 2.0),
        label_id=1,
        label_name="top",
        score=0.9,
    )
    result = SegmentationResult(
        image_size=(2, 2),
        instances=[instance],
        inference_time_ms=12.5,
    )

    payload = result.to_dict()

    assert payload["instances"][0]["area"] == 3
    assert "mask" not in payload["instances"][0]


def test_numpy_image_is_loaded_as_rgb_pil_image():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    loaded = load_rgb_image(image)

    assert loaded.mode == "RGB"
    assert loaded.size == (4, 4)


def test_deepfashion2_category_mapping_collapses_to_prd_taxonomy():
    assert DEEPFASHION2_TO_PROJECT_CATEGORY[1] == 1
    assert DEEPFASHION2_TO_PROJECT_CATEGORY[3] == 4
    assert DEEPFASHION2_TO_PROJECT_CATEGORY[7] == 2
    assert DEEPFASHION2_TO_PROJECT_CATEGORY[9] == 3
    assert DEEPFASHION2_TO_PROJECT_CATEGORY[13] == 5


def test_local_region_query_dataset_loads_jsonl(tmp_path):
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                '{"image": "/tmp/1.jpg", "annotation": "/tmp/1.json", '
                '"item_key": "item1", "category_id": 1, '
                '"category_name": "short sleeve top", "query": "这件衣服的领口", '
                '"region": "neckline", "garment_box": [0, 0, 10, 20], '
                '"region_box": [2, 0, 8, 5], "source": "landmark_pseudo_label", '
                '"confidence": 0.82}'
            ]
        ),
        encoding="utf-8",
    )

    dataset = LocalRegionQueryDataset(jsonl_path)
    record = dataset[0]

    assert len(dataset) == 1
    assert record.query == "这件衣服的领口"
    assert record.region == "neckline"
    assert record.garment_box == (0.0, 0.0, 10.0, 20.0)
    assert record.region_box == (2.0, 0.0, 8.0, 5.0)
    assert record.category_id == 1


def test_local_region_query_iterator_can_skip_records(tmp_path):
    jsonl_path = tmp_path / "records.jsonl"
    lines = []
    for index in range(3):
        lines.append(
            '{"image": "/tmp/%s.jpg", "annotation": "/tmp/%s.json", '
            '"item_key": "item1", "query": "q%s", "region": "neckline", '
            '"garment_box": [0, 0, 10, 20], "region_box": [2, 0, 8, 5], '
            '"source": "landmark_pseudo_label", "confidence": 0.82}'
            % (index, index, index)
        )
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")

    records = list(
        iter_local_region_query_records(
            jsonl_path,
            max_records=1,
            skip_records=2,
        )
    )

    assert len(records) == 1
    assert records[0].query == "q2"


def test_deepfashion2_dataset_ignores_hidden_annotation_files(tmp_path):
    image_dir = tmp_path / "image"
    anno_dir = tmp_path / "annos"
    image_dir.mkdir()
    anno_dir.mkdir()
    Image.new("RGB", (4, 4)).save(image_dir / "000001.jpg")
    (anno_dir / "000001.json").write_text(
        '{"source": "000001.jpg"}',
        encoding="utf-8",
    )
    (anno_dir / "._000001.json").write_bytes(b"\x00\x05\x16\x07binary")

    dataset = DeepFashion2Dataset(image_dir, anno_dir)

    assert len(dataset) == 1
    assert dataset.annotation_paths[0].name == "000001.json"


def test_deepfashion2_dataset_falls_back_from_invalid_annotation_box(tmp_path):
    image_dir = tmp_path / "image"
    anno_dir = tmp_path / "annos"
    image_dir.mkdir()
    anno_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "000001.jpg")
    (anno_dir / "000001.json").write_text(
        """
        {
          "source": "000001.jpg",
          "item1": {
            "category_id": 1,
            "bounding_box": [0, 0, 0, 0],
            "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]]
          }
        }
        """,
        encoding="utf-8",
    )

    dataset = DeepFashion2Dataset(image_dir, anno_dir)
    _, target = dataset[0]

    assert target["boxes"].shape == (1, 4)
    assert target["boxes"][0, 2] > target["boxes"][0, 0]
    assert target["boxes"][0, 3] > target["boxes"][0, 1]


def test_deepfashion2_get_labels_maps_annotation_categories(tmp_path):
    image_dir = tmp_path / "image"
    anno_dir = tmp_path / "annos"
    image_dir.mkdir()
    anno_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "000001.jpg")
    (anno_dir / "000001.json").write_text(
        """
        {
          "source": "000001.jpg",
          "item1": {"category_id": 1},
          "item2": {"category_id": 9}
        }
        """,
        encoding="utf-8",
    )

    dataset = DeepFashion2Dataset(image_dir, anno_dir)

    assert dataset.get_labels(0) == [1, 3]


def test_balanced_sampler_upweights_rare_class_images(tmp_path):
    image_dir = tmp_path / "image"
    anno_dir = tmp_path / "annos"
    image_dir.mkdir()
    anno_dir.mkdir()
    for index in range(3):
        Image.new("RGB", (8, 8)).save(image_dir / f"{index + 1:06d}.jpg")

    (anno_dir / "000001.json").write_text(
        '{"source": "000001.jpg", "item1": {"category_id": 1}}',
        encoding="utf-8",
    )
    (anno_dir / "000002.json").write_text(
        '{"source": "000002.jpg", "item1": {"category_id": 1}}',
        encoding="utf-8",
    )
    (anno_dir / "000003.json").write_text(
        '{"source": "000003.jpg", "item1": {"category_id": 9}}',
        encoding="utf-8",
    )

    dataset = DeepFashion2Dataset(image_dir, anno_dir)
    sampler = build_balanced_sampler(
        dataset,
        {0: "background", 1: "top", 3: "skirt"},
    )

    weights = sampler.weights.tolist()
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]


def test_hard_case_weights_upweight_selected_failure_images(tmp_path):
    image_dir = tmp_path / "image"
    anno_dir = tmp_path / "annos"
    image_dir.mkdir()
    anno_dir.mkdir()
    for index in range(2):
        Image.new("RGB", (8, 8)).save(image_dir / f"{index + 1:06d}.jpg")
        (anno_dir / f"{index + 1:06d}.json").write_text(
            '{"source": "%06d.jpg", "item1": {"category_id": 1}}' % (index + 1),
            encoding="utf-8",
        )

    hard_case_path = tmp_path / "failure_cases.json"
    hard_case_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "image": "000002.jpg",
                        "reason": "dress_confused_as_top",
                    },
                    {
                        "image": "000001.jpg",
                        "reason": "ignored_reason",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = DeepFashion2Dataset(image_dir, anno_dir)
    weights = build_hard_case_weights(
        dataset,
        {
            "enabled": True,
            "path": str(hard_case_path),
            "reasons": ["dress_confused_as_top"],
            "weight_multiplier": 3.0,
        },
    )

    assert weights == [1.0, 3.0]


def test_deepfashion2_horizontal_flip_augments_boxes_and_masks(tmp_path):
    image_dir = tmp_path / "image"
    anno_dir = tmp_path / "annos"
    image_dir.mkdir()
    anno_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "000001.jpg")
    (anno_dir / "000001.json").write_text(
        """
        {
          "source": "000001.jpg",
          "item1": {
            "category_id": 1,
            "bounding_box": [1, 1, 5, 5],
            "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]]
          }
        }
        """,
        encoding="utf-8",
    )

    dataset = DeepFashion2Dataset(
        image_dir,
        anno_dir,
        augmentation={
            "enabled": True,
            "horizontal_flip_prob": 1.0,
        },
    )
    _, target = dataset[0]

    assert target["boxes"][0].tolist() == [3.0, 1.0, 7.0, 5.0]
    assert target["masks"][0, 1:6, 3:8].sum() > 0


def test_batch_prediction_skips_hidden_image_metadata_files(tmp_path):
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "inference"
        / "batch_predict_instance_segmentation.py"
    )
    spec = importlib.util.spec_from_file_location("batch_predict", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    image_dir = tmp_path / "image"
    image_dir.mkdir()
    Image.new("RGB", (8, 8)).save(image_dir / "000001.jpg")
    (image_dir / "._000001.jpg").write_text("not a real image", encoding="utf-8")

    image_paths = module.collect_images(image_dir, max_images=None)

    assert [path.name for path in image_paths] == ["000001.jpg"]


def test_failure_analysis_classifies_confusion_and_misses():
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval"
        / "analyze_instance_segmentation_failures.py"
    )
    spec = importlib.util.spec_from_file_location("failure_analysis", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert (
        module.classify_failure_reason(
            "dress",
            "top",
            best_any_iou=0.52,
            best_same_iou=0.10,
            low_iou_threshold=0.75,
            miss_iou_threshold=0.3,
        )
        == "dress_confused_as_top"
    )
    assert (
        module.classify_failure_reason(
            "outerwear",
            None,
            best_any_iou=0.05,
            best_same_iou=0.05,
            low_iou_threshold=0.75,
            miss_iou_threshold=0.3,
        )
        == "missed_outerwear"
    )
    assert (
        module.classify_failure_reason(
            "top",
            "top",
            best_any_iou=0.82,
            best_same_iou=0.82,
            low_iou_threshold=0.75,
            miss_iou_threshold=0.3,
        )
        is None
    )
