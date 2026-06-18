import numpy as np

from fashion_mm.models.local_region import parse_region_query
from fashion_mm.models.local_region import propose_local_region


def test_parse_region_query_extracts_region_and_garment_hint():
    parsed = parse_region_query("这件连衣裙的领口是什么设计")

    assert parsed.region == "neckline"
    assert parsed.garment_hint == "dress"
    assert parsed.is_supported_region is True


def test_parse_region_query_marks_unsupported_region():
    parsed = parse_region_query("这个口袋的设计")

    assert parsed.region == "pocket"
    assert parsed.is_supported_region is False


def test_propose_neckline_region_is_clipped_by_garment_mask():
    garment_mask = np.zeros((100, 80), dtype=bool)
    garment_mask[10:90, 20:60] = True

    proposal = propose_local_region(
        garment_mask,
        garment_box=(20, 10, 60, 90),
        region="neckline",
    )

    assert proposal.status == "ok"
    assert proposal.box is not None
    assert proposal.box[1] < 35
    assert proposal.mask.sum() > 0
    assert np.logical_and(proposal.mask, ~garment_mask).sum() == 0


def test_propose_unsupported_region_returns_structured_fallback():
    garment_mask = np.ones((20, 20), dtype=bool)

    proposal = propose_local_region(
        garment_mask,
        garment_box=(0, 0, 20, 20),
        region="pocket",
    )

    assert proposal.status == "unsupported_region"
    assert proposal.box is None
    assert proposal.mask.sum() == 0
