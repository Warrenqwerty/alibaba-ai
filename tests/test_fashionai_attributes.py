from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from fashion_mm.data_loaders import build_fashionai_transform
from fashion_mm.data_loaders import collate_fashionai_attributes
from fashion_mm.data_loaders import deduplicate_fashionai_records
from fashion_mm.data_loaders import FashionAIAttributeDataset
from fashion_mm.data_loaders import FashionAIAttributeDefinition
from fashion_mm.data_loaders import FashionAIAttributeSchema
from fashion_mm.data_loaders import infer_fashionai_schema
from fashion_mm.data_loaders import parse_fashionai_label
from fashion_mm.data_loaders import prepare_fashionai_round1_splits
from fashion_mm.data_loaders import read_fashionai_annotations
from fashion_mm.data_loaders import split_records_by_image
from fashion_mm.data_loaders import stratified_split_records
from fashion_mm.models.attributes import FashionAttributeClassifier
from fashion_mm.models.attributes import FashionAttributePredictor
from fashion_mm.models.attributes import prepare_masked_region
from fashion_mm.models.attributes import run_attribute_epoch
from fashion_mm.models.instance_segmentation import FashionInstance
from fashion_mm.models.instance_segmentation import SegmentationResult
from fashion_mm.pipelines import FashionVisualPipeline
from fashion_mm.utils.latency import percentile
from fashion_mm.utils.latency import summarize_timings


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


def test_fashionai_round_sources_deduplicate_by_relative_image_id(tmp_path):
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()
    for root, color in ((source_a, "white"), (source_b, "gray")):
        (root / "Images" / "collar").mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(root / "Images" / "collar" / "shared.jpg")
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    row = "Images/collar/shared.jpg,collar_design_labels,nyn\n"
    csv_a.write_text(row, encoding="utf-8")
    csv_b.write_text(row, encoding="utf-8")

    records_a = read_fashionai_annotations(csv_a, image_root=source_a, source_name="a")
    records_b = read_fashionai_annotations(csv_b, image_root=source_b, source_name="b")
    records, duplicate_count = deduplicate_fashionai_records([*records_a, *records_b])

    assert duplicate_count == 1
    assert len(records) == 1
    assert records[0].source_name == "a"
    assert records[0].split_key == "Images/collar/shared.jpg"


def test_fashionai_round_sources_reject_conflicting_overlap(tmp_path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("same.jpg,a_labels,yn\n", encoding="utf-8")
    csv_b.write_text("same.jpg,a_labels,ny\n", encoding="utf-8")
    records_a = read_fashionai_annotations(csv_a, source_name="a")
    records_b = read_fashionai_annotations(csv_b, source_name="b")

    with pytest.raises(ValueError, match="Conflicting FashionAI labels"):
        deduplicate_fashionai_records([*records_a, *records_b])


def test_fashionai_three_way_split_is_deterministic_and_stratified(tmp_path):
    rows = []
    for attribute_name, labels in {
        "collar_labels": ("yn", "ny"),
        "sleeve_labels": ("ynn", "nyn"),
    }.items():
        for label in labels:
            for index in range(20):
                rows.append(
                    f"{attribute_name}_{label}_{index}.jpg,{attribute_name},{label}"
                )
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    records = read_fashionai_annotations(csv_path)

    first = stratified_split_records(
        records,
        split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        seed=42,
    )
    second = stratified_split_records(
        records,
        split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        seed=42,
    )

    assert {name: len(split) for name, split in first.items()} == {
        "train": 64,
        "validation": 8,
        "test": 8,
    }
    assert {
        name: [record.split_key for record in split] for name, split in first.items()
    } == {
        name: [record.split_key for record in split] for name, split in second.items()
    }
    split_keys = {
        name: {record.split_key for record in split} for name, split in first.items()
    }
    assert split_keys["train"].isdisjoint(split_keys["validation"])
    assert split_keys["train"].isdisjoint(split_keys["test"])
    assert split_keys["validation"].isdisjoint(split_keys["test"])
    for split_name, expected_per_stratum in {
        "train": 16,
        "validation": 2,
        "test": 2,
    }.items():
        counts = {}
        for record in first[split_name]:
            key = record.attribute_name, record.target_index
            counts[key] = counts.get(key, 0) + 1
        assert set(counts.values()) == {expected_per_stratum}


def test_prepare_fashionai_round1_splits_writes_leak_free_manifests(tmp_path):
    source_a = tmp_path / "round_a"
    source_b = tmp_path / "round_b"
    for source in (source_a, source_b):
        (source / "Images" / "collar").mkdir(parents=True)
        (source / "Tests").mkdir()

    rows_a = []
    rows_b = []
    for index in range(30):
        image_name = f"Images/collar/{index:03d}.jpg"
        label = "yn" if index % 2 == 0 else "ny"
        Image.new("RGB", (8, 8), "white").save(source_a / image_name)
        rows_a.append(f"{image_name},collar_labels,{label}")
        if index >= 10:
            Image.new("RGB", (8, 8), "gray").save(source_b / image_name)
            rows_b.append(f"{image_name},collar_labels,{label}")
    for index in range(30, 40):
        image_name = f"Images/collar/{index:03d}.jpg"
        label = "yn" if index % 2 == 0 else "ny"
        Image.new("RGB", (8, 8), "gray").save(source_b / image_name)
        rows_b.append(f"{image_name},collar_labels,{label}")

    answer_a = source_a / "Tests" / "answer_a.csv"
    answer_b = source_b / "Tests" / "answer_b.csv"
    answer_a.write_text("\n".join(rows_a), encoding="utf-8")
    answer_b.write_text("\n".join(rows_b), encoding="utf-8")
    output_dir = tmp_path / "splits"

    payload = prepare_fashionai_round1_splits(
        source_a_root=source_a,
        source_b_root=source_b,
        answer_a=answer_a,
        answer_b=answer_b,
        output_dir=output_dir,
        split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        seed=42,
        label_map=None,
        validate_images=True,
    )

    assert payload["num_records_before_deduplication"] == 60
    assert payload["num_duplicate_records"] == 20
    assert payload["num_unique_records"] == 40
    assert payload["split_overlap_counts"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert {
        name: split["num_records"] for name, split in payload["splits"].items()
    } == {"train": 32, "validation": 4, "test": 4}
    assert payload["stratification_audit"]["stratification_key"] == (
        "attribute_name + strict_y_class"
    )
    assert payload["stratification_audit"]["num_strata"] == 2
    assert payload["stratification_audit"]["strata"]["collar_labels::0"][
        "counts"
    ] == {"train": 16, "validation": 2, "test": 2}
    assert payload["stratification_audit"]["strata"]["collar_labels::1"][
        "counts"
    ] == {"train": 16, "validation": 2, "test": 2}
    assert (output_dir / "train.csv").is_file()
    assert (output_dir / "validation.csv").is_file()
    assert (output_dir / "test.csv").is_file()
    assert (output_dir / "split_summary.json").is_file()


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
        "a.jpg,a_labels,yn\n" "b.jpg,b_labels,nyn\n",
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


def test_attribute_evaluation_reports_per_head_metrics_and_latency(tmp_path):
    Image.new("RGB", (16, 16), "white").save(tmp_path / "a.jpg")
    Image.new("RGB", (16, 16), "black").save(tmp_path / "b.jpg")
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "a.jpg,collar_design_labels,ynn\n" "b.jpg,sleeve_length_labels,nynn\n",
        encoding="utf-8",
    )
    records = read_fashionai_annotations(csv_path, image_root=tmp_path)
    dataset = FashionAIAttributeDataset(
        records, build_fashionai_transform(32, train=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fashionai_attributes,
    )
    model = FashionAttributeClassifier(
        _test_schema(), backbone_name="tiny_cnn", pretrained=False
    )

    metrics = run_attribute_epoch(model, loader, device=torch.device("cpu"))

    assert metrics["num_records"] == 2
    assert set(metrics["by_attribute"]) == {
        "collar_design_labels",
        "sleeve_length_labels",
    }
    assert metrics["model_inference_time_ms"] >= 0.0
    assert metrics["avg_model_inference_time_ms"] >= 0.0


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


def test_attribute_latency_summary_uses_interpolated_p95():
    timings = [1.0, 2.0, 3.0, 4.0]

    assert percentile(timings, 0.0) == 1.0
    assert percentile(timings, 1.0) == 4.0
    assert summarize_timings(timings) == {
        "mean": 2.5,
        "median": 2.5,
        "p95": 3.85,
        "max": 4.0,
    }


def test_attribute_latency_summary_rejects_empty_values():
    with pytest.raises(ValueError, match="empty timing sequence"):
        summarize_timings([])


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
