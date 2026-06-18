import numpy as np

from fashion_mm.models.local_region import propose_region_from_landmarks


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
