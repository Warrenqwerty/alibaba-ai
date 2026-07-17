import numpy as np

from fashion_mm.models.local_region import propose_region_from_landmarks


def raw_landmarks(
    count: int,
    points: dict[int, tuple[int, int, int]],
) -> list[int]:
    values = [0, 0, 0] * count
    for index, (x, y, visibility) in points.items():
        offset = (index - 1) * 3
        values[offset : offset + 3] = [x, y, visibility]
    return values


def test_landmark_region_proposal_uses_landmark_source_and_clips_mask():
    garment_mask = np.zeros((100, 80), dtype=bool)
    garment_mask[10:90, 20:60] = True
    landmarks = [
        30,
        20,
        2,
        35,
        18,
        2,
        40,
        20,
        2,
        45,
        22,
        1,
        50,
        21,
        2,
        55,
        20,
        2,
    ]

    proposal = propose_region_from_landmarks(
        garment_mask,
        garment_box=(20, 10, 60, 90),
        raw_landmarks=landmarks,
        region="neckline",
    )

    assert proposal.status == "ok"
    assert proposal.source == "landmark_pseudo_label"
    assert proposal.box is not None
    assert proposal.box[1] < 30
    assert proposal.mask.sum() > 0
    assert np.logical_and(proposal.mask, ~garment_mask).sum() == 0


def test_landmark_region_falls_back_when_points_are_missing():
    garment_mask = np.zeros((100, 80), dtype=bool)
    garment_mask[10:90, 20:60] = True
    landmarks = [0, 0, 0] * 30

    proposal = propose_region_from_landmarks(
        garment_mask,
        garment_box=(20, 10, 60, 90),
        raw_landmarks=landmarks,
        region="neckline",
    )

    assert proposal.status == "ok"
    assert proposal.source == "rule_baseline"


def test_category_specific_cuffs_convert_image_side_to_wearer_side():
    garment_mask = np.zeros((120, 120), dtype=bool)
    garment_mask[10:110, 10:110] = True
    landmarks = raw_landmarks(
        25,
        {
            9: (20, 50, 2),
            10: (20, 65, 2),
            22: (100, 50, 2),
            23: (100, 65, 2),
        },
    )

    left = propose_region_from_landmarks(
        garment_mask,
        garment_box=(10, 10, 110, 110),
        raw_landmarks=landmarks,
        region="left_cuff",
        category_id="1",
    )
    right = propose_region_from_landmarks(
        garment_mask,
        garment_box=(10, 10, 110, 110),
        raw_landmarks=landmarks,
        region="right_cuff",
        category_id=1,
    )

    assert left.source == "landmark_pseudo_label"
    assert right.source == "landmark_pseudo_label"
    assert left.box is not None and left.box[0] > 80
    assert right.box is not None and right.box[2] < 40


def test_lower_body_waist_uses_category_specific_landmarks():
    garment_mask = np.zeros((120, 120), dtype=bool)
    garment_mask[10:110, 10:110] = True
    landmarks = raw_landmarks(
        14,
        {
            1: (20, 25, 2),
            2: (60, 20, 2),
            3: (100, 25, 1),
        },
    )

    proposal = propose_region_from_landmarks(
        garment_mask,
        garment_box=(10, 10, 110, 110),
        raw_landmarks=landmarks,
        region="waist",
        category_id=8,
    )

    assert proposal.source == "landmark_pseudo_label"
    assert proposal.box is not None
    assert proposal.box[3] < 40


def test_sleeveless_category_does_not_reuse_sleeve_landmark_indices():
    garment_mask = np.zeros((120, 120), dtype=bool)
    garment_mask[10:110, 10:110] = True
    landmarks = raw_landmarks(15, {9: (20, 50, 2), 10: (20, 65, 2)})

    proposal = propose_region_from_landmarks(
        garment_mask,
        garment_box=(10, 10, 110, 110),
        raw_landmarks=landmarks,
        region="left_cuff",
        category_id=5,
    )

    assert proposal.source == "rule_baseline"
