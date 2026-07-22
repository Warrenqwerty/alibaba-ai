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


def test_deepfashion2_config_uses_only_available_dataset_classes():
    config = load_config(ROOT / "configs/model/instance_segmentation_deepfashion2.yaml")

    assert config["model"]["num_classes"] == 6
    assert config["categories"] == {
        0: "background",
        1: "top",
        2: "pants",
        3: "skirt",
        4: "outerwear",
        5: "dress",
    }
    assert config["training"]["num_epochs"] == 1
    assert config["training"]["class_balanced_sampling"] is True
    assert config["training"]["hard_mining"] == {
        "enabled": False,
        "path": "outputs/failure_analysis/failure_cases_1000.json",
        "weight_multiplier": 2.0,
        "reasons": [
            "dress_confused_as_top",
            "top_confused_as_dress",
            "low_iou_dress",
            "low_iou_top",
            "low_iou_pants",
        ],
    }
    assert config["inference"]["score_threshold"] == 0.3
    assert config["inference"]["mask_threshold"] == 0.4
    assert config["training"]["augmentation"] == {
        "enabled": True,
        "horizontal_flip_prob": 0.5,
        "scale_jitter": [0.95, 1.05],
        "brightness": 0.08,
        "contrast": 0.08,
        "saturation": 0.05,
    }


def test_fashionai_resnet18_config_changes_only_backbone():
    baseline = load_config(ROOT / "configs/model/fashionai_attributes.yaml")
    candidate = load_config(
        ROOT / "configs/model/fashionai_attributes_resnet18.yaml"
    )

    expected_model = {**baseline["model"], "backbone": "resnet18"}
    assert candidate["model"] == expected_model
    assert candidate["training"] == baseline["training"]
    assert candidate["inference"] == baseline["inference"]
    assert candidate["performance_target"] == baseline["performance_target"]


def test_local_paths_config_points_to_repo_data_dir():
    config = load_config(ROOT / "configs/paths.yaml")

    assert config["data_root"] == "data"
    assert config["deepfashion2"]["root"] == "data/DeepFashion2"
    assert config["deepfashion2"]["train_image_dir"] == (
        "data/DeepFashion2/train/image"
    )
    assert config["deepfashion2"]["train_anno_dir"] == (
        "data/DeepFashion2/train/annos"
    )


def test_autodl_paths_config_points_to_autodl_storage():
    config = load_config(ROOT / "configs/paths.autodl.yaml")

    assert config["data_root"] == "/root/autodl-tmp/datasets"
    assert config["deepfashion2"]["root"] == (
        "/root/autodl-tmp/datasets/DeepFashion2"
    )
    assert config["deepfashion2"]["train_image_dir"] == (
        "/root/autodl-tmp/datasets/DeepFashion2/train/image"
    )
