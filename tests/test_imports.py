from pathlib import Path

import numpy as np
from PIL import Image

from fashion_mm.data_loaders import DeepFashion2Dataset
from fashion_mm.data_loaders import build_balanced_sampler
from fashion_mm.data_loaders.deepfashion2 import DEEPFASHION2_TO_PROJECT_CATEGORY
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
