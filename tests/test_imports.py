import numpy as np
from PIL import Image

from fashion_mm.data_loaders import DeepFashion2Dataset
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
