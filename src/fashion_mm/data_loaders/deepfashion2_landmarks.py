from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FashionLandmark:
    """One DeepFashion2 clothing landmark point."""

    index: int
    x: float
    y: float
    visibility: int

    @property
    def is_labeled(self) -> bool:
        return self.visibility > 0

    @property
    def is_visible(self) -> bool:
        return self.visibility == 2

    @property
    def is_occluded(self) -> bool:
        return self.visibility == 1


def parse_landmarks(raw_landmarks: Iterable[float | int]) -> list[FashionLandmark]:
    """Parse DeepFashion2 flat landmark list into indexed point objects."""
    values = list(raw_landmarks)
    if len(values) % 3 != 0:
        raise ValueError(
            "DeepFashion2 landmarks must be flattened [x, y, visibility] triplets."
        )

    landmarks = []
    for offset in range(0, len(values), 3):
        visibility = int(values[offset + 2])
        if visibility not in {0, 1, 2}:
            raise ValueError(f"Unsupported landmark visibility value: {visibility}")
        landmarks.append(
            FashionLandmark(
                index=offset // 3 + 1,
                x=float(values[offset]),
                y=float(values[offset + 1]),
                visibility=visibility,
            )
        )
    return landmarks


def labeled_landmarks(raw_landmarks: Iterable[float | int]) -> list[FashionLandmark]:
    """Return only visible or occluded landmarks, skipping unlabeled points."""
    return [landmark for landmark in parse_landmarks(raw_landmarks) if landmark.is_labeled]
