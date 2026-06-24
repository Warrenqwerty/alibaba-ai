from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalRegionQueryRecord:
    """One weak language-region training record for 3.1.2."""

    image: Path
    annotation: Path
    item_key: str
    query: str
    region: str
    garment_box: tuple[float, float, float, float]
    region_box: tuple[float, float, float, float]
    source: str
    confidence: float
    category_id: int | None = None
    category_name: str | None = None


@dataclass(frozen=True)
class LocalRegionCandidateRecord:
    """One candidate-level weak local-region record."""

    image: Path
    annotation: Path
    item_key: str
    query: str
    target_region: str
    target_region_box: tuple[float, float, float, float]
    garment_box: tuple[float, float, float, float]
    candidate_region: str
    candidate_box: tuple[float, float, float, float]
    iou: float
    label: int
    weak_label_source: str
    weak_label_confidence: float
    category_id: int | None = None
    category_name: str | None = None


class LocalRegionQueryDataset:
    """JSONL dataset for weak DeepFashion2 local-region query records."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Local-region JSONL not found: {self.jsonl_path}")
        self.records = self._load_records(self.jsonl_path)
        if not self.records:
            raise ValueError(f"No local-region records found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LocalRegionQueryRecord:
        return self.records[index]

    @staticmethod
    def _load_records(jsonl_path: Path) -> list[LocalRegionQueryRecord]:
        return list(iter_local_region_query_records(jsonl_path))


def iter_local_region_query_records(
    jsonl_path: str | Path,
    max_records: int | None = None,
    skip_records: int = 0,
):
    """Stream weak local-region JSONL records without loading the full file."""
    path = Path(jsonl_path)
    yielded = 0
    seen = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if seen < skip_records:
                seen += 1
                continue
            try:
                payload = json.loads(stripped)
            except JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            yield _record_from_payload(payload, path, line_number)
            seen += 1
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return


def iter_local_region_candidate_records(
    jsonl_path: str | Path,
    max_records: int | None = None,
    skip_records: int = 0,
):
    """Stream candidate-level local-region JSONL records."""
    path = Path(jsonl_path)
    yielded = 0
    seen = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if seen < skip_records:
                seen += 1
                continue
            try:
                payload = json.loads(stripped)
            except JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            yield _candidate_record_from_payload(payload, path, line_number)
            seen += 1
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return


def _record_from_payload(
    payload: dict[str, Any],
    jsonl_path: Path,
    line_number: int,
) -> LocalRegionQueryRecord:
    required = [
        "image",
        "annotation",
        "item_key",
        "query",
        "region",
        "garment_box",
        "region_box",
        "source",
        "confidence",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"Missing keys at {jsonl_path}:{line_number}: {', '.join(missing)}"
        )

    return LocalRegionQueryRecord(
        image=Path(str(payload["image"])),
        annotation=Path(str(payload["annotation"])),
        item_key=str(payload["item_key"]),
        query=str(payload["query"]),
        region=str(payload["region"]),
        garment_box=_box_from_payload(payload["garment_box"], jsonl_path, line_number),
        region_box=_box_from_payload(payload["region_box"], jsonl_path, line_number),
        source=str(payload["source"]),
        confidence=float(payload["confidence"]),
        category_id=(
            int(payload["category_id"])
            if payload.get("category_id") is not None
            else None
        ),
        category_name=(
            str(payload["category_name"])
            if payload.get("category_name") is not None
            else None
        ),
    )


def _candidate_record_from_payload(
    payload: dict[str, Any],
    jsonl_path: Path,
    line_number: int,
) -> LocalRegionCandidateRecord:
    required = [
        "image",
        "annotation",
        "item_key",
        "query",
        "target_region",
        "target_region_box",
        "garment_box",
        "candidate_region",
        "candidate_box",
        "iou",
        "label",
        "weak_label_source",
        "weak_label_confidence",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"Missing keys at {jsonl_path}:{line_number}: {', '.join(missing)}"
        )

    return LocalRegionCandidateRecord(
        image=Path(str(payload["image"])),
        annotation=Path(str(payload["annotation"])),
        item_key=str(payload["item_key"]),
        query=str(payload["query"]),
        target_region=str(payload["target_region"]),
        target_region_box=_box_from_payload(
            payload["target_region_box"],
            jsonl_path,
            line_number,
        ),
        garment_box=_box_from_payload(payload["garment_box"], jsonl_path, line_number),
        candidate_region=str(payload["candidate_region"]),
        candidate_box=_box_from_payload(
            payload["candidate_box"],
            jsonl_path,
            line_number,
        ),
        iou=float(payload["iou"]),
        label=int(payload["label"]),
        weak_label_source=str(payload["weak_label_source"]),
        weak_label_confidence=float(payload["weak_label_confidence"]),
        category_id=(
            int(payload["category_id"])
            if payload.get("category_id") is not None
            else None
        ),
        category_name=(
            str(payload["category_name"])
            if payload.get("category_name") is not None
            else None
        ),
    )


def _box_from_payload(
    value: Any,
    jsonl_path: Path,
    line_number: int,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Invalid box at {jsonl_path}:{line_number}: {value}")
    box = tuple(float(item) for item in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid box geometry at {jsonl_path}:{line_number}: {box}")
    return box
