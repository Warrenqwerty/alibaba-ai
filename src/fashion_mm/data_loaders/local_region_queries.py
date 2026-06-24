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
):
    """Stream weak local-region JSONL records without loading the full file."""
    path = Path(jsonl_path)
    yielded = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            yield _record_from_payload(payload, path, line_number)
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
