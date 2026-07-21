from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fashion_mm.data_loaders import build_fashionai_transform
from fashion_mm.data_loaders import collate_fashionai_attributes
from fashion_mm.data_loaders import FashionAIAttributeDataset
from fashion_mm.data_loaders import FashionAIAttributeDefinition
from fashion_mm.data_loaders import FashionAIAttributeSchema
from fashion_mm.data_loaders import infer_fashionai_schema
from fashion_mm.data_loaders import parse_fashionai_label
from fashion_mm.data_loaders import read_fashionai_annotations
from fashion_mm.data_loaders import split_records_by_image
from fashion_mm.models.attributes import FashionAttributeClassifier
from fashion_mm.models.attributes import FashionAttributePredictor
from fashion_mm.models.attributes import prepare_masked_region
from fashion_mm.models.instance_segmentation import FashionInstance
from fashion_mm.models.instance_segmentation import SegmentationResult
from fashion_mm.pipelines import FashionVisualPipeline


def test_parse_fashionai_label_preserves_probable_classes():
    target, probable = parse_fashionai_label("nnymn")

    assert target == 2
    assert probable == (3,)


@pytest.mark.parametrize("label", ["nnnn", "yynn", "nyxn", ""])
def test_parse_fashionai_label_rejects_invalid_vectors(label):
    with pytest.raises(ValueError):
        parse_fashionai_label(label)


def test_fashionai_csv_infers_dynamic_attribute_schema(tmp_path):
    image_root = tmp_path / "dataset"
    image_root.mkdir()
    Image.new("RGB", (12, 10), "white").save(image_root / "a.jpg")
    Image.new("RGB", (12, 10), "gray").save(image_root / "b.jpg")
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "image_path,attribute_name,label\n"
        "a.jpg,collar_design_labels,nyn\n"
        "a.jpg,sleeve_length_labels,nnym\n"
        "b.jpg,collar_design_labels,ynn\n",
        encoding="utf-8",
    )

    records = read_fashionai_annotations(
        csv_path, image_root=image_root, validate_images=True
    )
    schema = infer_fashionai_schema(records)

    assert len(records) == 3
    assert schema.attribute_names == (
        "collar_design_labels",
        "sleeve_length_labels",
    )
    assert schema.definition("collar_design_labels").num_classes == 3
    assert schema.definition("sleeve_length_labels").num_classes == 4
    assert records[1].acceptable_indices == (2, 3)


def test_fashionai_split_keeps_all_heads_for_one_image_together(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for image_index in range(6):
        image_name = f"{image_index}.jpg"
        Image.new("RGB", (8, 8)).save(image_root / image_name)
        rows.extend(
            [
                f"{image_name},a_labels,yn",
                f"{image_name},b_labels,nyn",
            ]
        )
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    records = read_fashionai_annotations(csv_path, image_root=image_root)

    train_records, validation_records = split_records_by_image(
        records, validation_fraction=0.33, seed=42
    )
    train_images = {record.image_path for record in train_records}
    validation_images = {record.image_path for record in validation_records}

    assert train_images
    assert validation_images
    assert train_images.isdisjoint(validation_images)


def test_masked_region_crop_uses_tight_padded_mask_box():
    image = np.full((12, 16, 3), 255, dtype=np.uint8)
    image[4:9, 5:11] = (255, 0, 0)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[4:9, 5:11] = 1

    _, region = prepare_masked_region(image, mask, padding_fraction=0.0)

    assert region.box == (5, 4, 11, 9)
    assert region.image.size == (6, 5)
    assert region.mask_area == 30
    assert region.mask_coverage == 1.0


def test_attribute_dataset_and_collate_support_heterogeneous_heads(tmp_path):
    Image.new("RGB", (16, 16), "white").save(tmp_path / "a.jpg")
    Image.new("RGB", (16, 16), "black").save(tmp_path / "b.jpg")
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "a.jpg,a_labels,yn\n"
        "b.jpg,b_labels,nyn\n",
        encoding="utf-8",
    )
    records = read_fashionai_annotations(csv_path, image_root=tmp_path)
    dataset = FashionAIAttributeDataset(
        records, build_fashionai_transform(32, train=False)
    )

    batch = collate_fashionai_attributes([dataset[0], dataset[1]])

    assert batch["images"].shape == (2, 3, 32, 32)
    assert batch["attribute_names"] == ["a_labels", "b_labels"]
    assert batch["target_indices"].tolist() == [0, 1]


def test_multi_head_attribute_model_outputs_schema_shapes():
    schema = _test_schema()
    model = FashionAttributeClassifier(
        schema, backbone_name="tiny_cnn", pretrained=False
    ).eval()

    outputs = model(torch.zeros((2, 3, 32, 32)))

    assert outputs["collar_design_labels"].shape == (2, 3)
    assert outputs["sleeve_length_labels"].shape == (2, 4)


def test_attribute_predictor_runs_mask_to_structured_predictions(tmp_path):
    schema = _test_schema()
    model = FashionAttributeClassifier(
        schema, backbone_name="tiny_cnn", pretrained=False
    )
    checkpoint_path = tmp_path / "attributes.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "schema": schema.to_dict(),
            "model_config": {
                "backbone": "tiny_cnn",
                "image_size": 32,
                "dropout": 0.2,
                "top_k": 2,
                "confidence_threshold": 0.0,
                "mask_padding_fraction": 0.0,
            },
        },
        checkpoint_path,
    )
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (24, 20), "blue").save(image_path)
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[4:16, 6:19] = 1

    predictor = FashionAttributePredictor(checkpoint_path, device="cpu")
    result = predictor.predict(image_path, mask)
    payload = result.to_dict()

    assert result.status == "ok"
    assert result.region_box == (6, 4, 19, 16)
    assert len(result.predictions) == 2
    assert {item["attribute_name"] for item in payload["attributes"]} == {
        "collar_design_labels",
        "sleeve_length_labels",
    }
    assert all(0.0 <= item["confidence"] <= 1.0 for item in payload["attributes"])


def test_fashion_visual_pipeline_runs_all_three_prd_stages(tmp_path):
    schema = _test_schema()
    model = FashionAttributeClassifier(
        schema, backbone_name="tiny_cnn", pretrained=False
    )
    checkpoint_path = tmp_path / "attributes.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "schema": schema.to_dict(),
            "model_config": {
                "backbone": "tiny_cnn",
                "image_size": 32,
                "dropout": 0.2,
                "top_k": 2,
                "confidence_threshold": 0.0,
                "mask_padding_fraction": 0.0,
            },
        },
        checkpoint_path,
    )
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (40, 48), "white").save(image_path)
    garment_mask = np.zeros((48, 40), dtype=bool)
    garment_mask[5:44, 7:34] = True
    segmentation = SegmentationResult(
        image_size=(40, 48),
        instances=[
            FashionInstance(
                mask=garment_mask,
                box=(7.0, 5.0, 34.0, 44.0),
                label_id=1,
                label_name="top",
                score=0.95,
            )
        ],
        inference_time_ms=1.0,
    )

    class StaticSegmentationPredictor:
        def predict(self, image):
            return segmentation

    pipeline = FashionVisualPipeline(
        StaticSegmentationPredictor(),
        FashionAttributePredictor(checkpoint_path, device="cpu"),
    )
    result = pipeline.predict(image_path, "这件衣服的领口")
    payload = result.to_dict()

    assert result.status == "ok"
    assert payload["local_region"]["query"]["region"] == "neckline"
    assert payload["local_region"]["region"]["box"] is not None
    assert len(payload["attribute_extraction"]["attributes"]) == 2


def _test_schema() -> FashionAIAttributeSchema:
    return FashionAIAttributeSchema(
        (
            FashionAIAttributeDefinition(
                name="collar_design_labels",
                num_classes=3,
                value_names=("round", "v", "square"),
            ),
            FashionAIAttributeDefinition(
                name="sleeve_length_labels",
                num_classes=4,
                value_names=("sleeveless", "short", "long", "extra_long"),
            ),
        )
    )
