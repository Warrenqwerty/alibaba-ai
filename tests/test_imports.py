import numpy as np

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
