from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


FASHIONAI_LABEL_STATES = frozenset({"y", "m", "n"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FashionAIAttributeRecord:
    """One FashionAI image/attribute annotation row."""

    image_path: Path
    attribute_name: str
    label: str
    target_index: int
    probable_indices: tuple[int, ...]

    @property
    def num_classes(self) -> int:
        return len(self.label)

    @property
    def acceptable_indices(self) -> tuple[int, ...]:
        return (self.target_index, *self.probable_indices)


@dataclass(frozen=True)
class FashionAIAttributeDefinition:
    """Class layout for one FashionAI attribute dimension."""

    name: str
    num_classes: int
    value_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError(f"Attribute {self.name!r} must have at least two classes.")
        if len(self.value_names) != self.num_classes:
            raise ValueError(
                f"Attribute {self.name!r} has {self.num_classes} classes but "
                f"{len(self.value_names)} value names."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "value_names": list(self.value_names),
        }


@dataclass(frozen=True)
class FashionAIAttributeSchema:
    """All attribute heads inferred from one or more FashionAI CSV files."""

    definitions: tuple[FashionAIAttributeDefinition, ...]

    def __post_init__(self) -> None:
        names = [definition.name for definition in self.definitions]
        if not names:
            raise ValueError("FashionAI schema cannot be empty.")
        if len(names) != len(set(names)):
            raise ValueError("FashionAI schema contains duplicate attribute names.")

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self.definitions)

    def definition(self, attribute_name: str) -> FashionAIAttributeDefinition:
        for definition in self.definitions:
            if definition.name == attribute_name:
                return definition
        raise KeyError(f"Unknown FashionAI attribute: {attribute_name}")

    def to_dict(self) -> dict[str, Any]:
        return {"attributes": [definition.to_dict() for definition in self.definitions]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FashionAIAttributeSchema":
        raw_attributes = payload.get("attributes")
        if not isinstance(raw_attributes, list):
            raise ValueError("FashionAI schema must contain an attributes list.")
        definitions = []
        for raw in raw_attributes:
            if not isinstance(raw, dict):
                raise ValueError("FashionAI attribute definitions must be mappings.")
            definitions.append(
                FashionAIAttributeDefinition(
                    name=str(raw["name"]),
                    num_classes=int(raw["num_classes"]),
                    value_names=tuple(str(value) for value in raw["value_names"]),
                )
            )
        return cls(tuple(definitions))


def parse_fashionai_label(label: str) -> tuple[int, tuple[int, ...]]:
    """Parse FashionAI's y/m/n vector into one target and probable classes."""
    normalized = str(label).strip().lower()
    if not normalized:
        raise ValueError("FashionAI label cannot be empty.")
    invalid_states = set(normalized) - FASHIONAI_LABEL_STATES
    if invalid_states:
        raise ValueError(
            f"FashionAI label {label!r} contains invalid states: {sorted(invalid_states)}"
        )

    target_indices = tuple(index for index, state in enumerate(normalized) if state == "y")
    if len(target_indices) != 1:
        raise ValueError(
            f"FashionAI label {label!r} must contain exactly one 'y', "
            f"found {len(target_indices)}."
        )
    probable_indices = tuple(
        index for index, state in enumerate(normalized) if state == "m"
    )
    return target_indices[0], probable_indices


def read_fashionai_annotations(
    csv_paths: str | Path | Sequence[str | Path],
    *,
    image_root: str | Path | None = None,
    validate_images: bool = False,
    skip_invalid: bool = False,
    max_records: int | None = None,
) -> list[FashionAIAttributeRecord]:
    """Load FashionAI CSV rows without hard-coding attribute dimensions."""
    paths = _normalize_csv_paths(csv_paths)
    root = Path(image_root) if image_root is not None else None
    records: list[FashionAIAttributeRecord] = []

    for csv_path in paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            for row_number, row in enumerate(reader, start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                if row_number == 1 and _looks_like_header(row):
                    continue
                if len(row) < 3:
                    if skip_invalid:
                        continue
                    raise ValueError(
                        f"Expected three columns in {csv_path}:{row_number}, got {row!r}"
                    )

                try:
                    raw_image_path = Path(row[0].strip())
                    attribute_name = row[1].strip()
                    label = row[2].strip().lower()
                    if not attribute_name:
                        raise ValueError("attribute name cannot be empty")
                    target_index, probable_indices = parse_fashionai_label(label)
                    image_path = (
                        raw_image_path
                        if raw_image_path.is_absolute() or root is None
                        else root / raw_image_path
                    )
                    if validate_images and not image_path.is_file():
                        raise FileNotFoundError(f"Image not found: {image_path}")
                except (ValueError, FileNotFoundError):
                    if skip_invalid:
                        continue
                    raise

                records.append(
                    FashionAIAttributeRecord(
                        image_path=image_path,
                        attribute_name=attribute_name,
                        label=label,
                        target_index=target_index,
                        probable_indices=probable_indices,
                    )
                )
                if max_records is not None and len(records) >= max_records:
                    return records

    if not records:
        raise ValueError(f"No valid FashionAI annotations found in: {paths}")
    return records


def infer_fashionai_schema(
    records: Iterable[FashionAIAttributeRecord],
    value_names: dict[str, Sequence[str]] | None = None,
) -> FashionAIAttributeSchema:
    """Infer one classifier head per attribute and validate vector lengths."""
    class_counts: dict[str, int] = {}
    for record in records:
        previous = class_counts.setdefault(record.attribute_name, record.num_classes)
        if previous != record.num_classes:
            raise ValueError(
                f"Attribute {record.attribute_name!r} has inconsistent label lengths: "
                f"{previous} and {record.num_classes}."
            )

    definitions = []
    for name in sorted(class_counts):
        num_classes = class_counts[name]
        names = tuple(value_names.get(name, ())) if value_names else ()
        if not names:
            names = tuple(f"class_{index:03d}" for index in range(num_classes))
        definitions.append(
            FashionAIAttributeDefinition(
                name=name,
                num_classes=num_classes,
                value_names=names,
            )
        )
    return FashionAIAttributeSchema(tuple(definitions))


class FashionAIAttributeDataset(Dataset):
    """Torch dataset for heterogeneous FashionAI attribute heads."""

    def __init__(
        self,
        records: Sequence[FashionAIAttributeRecord],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        if not records:
            raise ValueError("FashionAIAttributeDataset requires at least one record.")
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "attribute_name": record.attribute_name,
            "target_index": record.target_index,
            "acceptable_indices": record.acceptable_indices,
            "image_path": str(record.image_path),
        }


def collate_fashionai_attributes(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate heterogeneous heads while preserving variable acceptable labels."""
    return {
        "images": torch.stack([sample["image"] for sample in samples]),
        "attribute_names": [sample["attribute_name"] for sample in samples],
        "target_indices": torch.tensor(
            [sample["target_index"] for sample in samples], dtype=torch.long
        ),
        "acceptable_indices": [sample["acceptable_indices"] for sample in samples],
        "image_paths": [sample["image_path"] for sample in samples],
    }


def build_fashionai_transform(image_size: int, *, train: bool) -> Callable:
    """Build ImageNet-compatible augmentation for the shared backbone."""
    if train:
        spatial = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        ]
    else:
        resize_size = round(image_size * 256 / 224)
        spatial = [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ]
    return transforms.Compose(
        [
            *spatial,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def split_records_by_image(
    records: Sequence[FashionAIAttributeRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[FashionAIAttributeRecord], list[FashionAIAttributeRecord]]:
    """Split by image path so one product cannot leak across attribute heads."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    unique_images = sorted({str(record.image_path) for record in records})
    if validation_fraction == 0.0 or len(unique_images) < 2:
        return list(records), []

    ordered = sorted(
        unique_images,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    validation_count = round(len(ordered) * validation_fraction)
    validation_count = min(max(validation_count, 1), len(ordered) - 1)
    validation_images = set(ordered[:validation_count])
    train_records = [
        record for record in records if str(record.image_path) not in validation_images
    ]
    validation_records = [
        record for record in records if str(record.image_path) in validation_images
    ]
    return train_records, validation_records


def discover_fashionai_csvs(root: str | Path) -> list[Path]:
    """Return candidate CSV files under a downloaded FashionAI root."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"FashionAI root not found: {root_path}")
    return sorted(path for path in root_path.rglob("*.csv") if not path.name.startswith("._"))


def _normalize_csv_paths(
    csv_paths: str | Path | Sequence[str | Path],
) -> list[Path]:
    raw_paths = [csv_paths] if isinstance(csv_paths, (str, Path)) else list(csv_paths)
    paths = [Path(path) for path in raw_paths]
    if not paths:
        raise ValueError("At least one FashionAI CSV path is required.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FashionAI CSV files not found: {missing}")
    return paths


def _looks_like_header(row: Sequence[str]) -> bool:
    first = row[0].strip().lower() if row else ""
    second = row[1].strip().lower() if len(row) > 1 else ""
    third = row[2].strip().lower() if len(row) > 2 else ""
    return (
        first in {"image", "image_path", "imagename", "image_name"}
        or second in {"attribute", "attribute_name", "attrkey", "attr_key"}
        or third in {"label", "attrvalues", "attr_values", "attribute_value"}
    )
