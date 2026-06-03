from pathlib import Path

from fashion_mm.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_instance_segmentation_config_matches_prd_categories():
    config = load_config(ROOT / "configs/model/instance_segmentation.yaml")

    assert config["model"]["num_classes"] == 9
    assert set(config["categories"].values()) == {
        "background",
        "top",
        "pants",
        "skirt",
        "outerwear",
        "dress",
        "shoes",
        "bag",
        "accessory",
    }
    assert config["performance_target"]["max_latency_ms"] == 50
    assert config["performance_target"]["min_iou"] == 0.85
