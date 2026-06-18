import pytest

from fashion_mm.data_loaders.deepfashion2_landmarks import labeled_landmarks
from fashion_mm.data_loaders.deepfashion2_landmarks import parse_landmarks


def test_parse_landmarks_creates_indexed_triplets():
    landmarks = parse_landmarks([10, 20, 2, 30, 40, 1, 0, 0, 0])

    assert len(landmarks) == 3
    assert landmarks[0].index == 1
    assert landmarks[0].x == 10
    assert landmarks[0].y == 20
    assert landmarks[0].is_visible is True
    assert landmarks[1].is_occluded is True
    assert landmarks[2].is_labeled is False


def test_labeled_landmarks_skips_visibility_zero_points():
    landmarks = labeled_landmarks([10, 20, 2, 0, 0, 0, 30, 40, 1])

    assert [landmark.index for landmark in landmarks] == [1, 3]


def test_parse_landmarks_rejects_invalid_flat_list():
    with pytest.raises(ValueError):
        parse_landmarks([10, 20])


def test_parse_landmarks_rejects_unknown_visibility():
    with pytest.raises(ValueError):
        parse_landmarks([10, 20, 9])
