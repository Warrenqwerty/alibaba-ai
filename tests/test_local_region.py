import numpy as np

from fashion_mm.models.instance_segmentation import FashionInstance
from fashion_mm.models.instance_segmentation import SegmentationResult
from fashion_mm.models.local_region import localize_region_from_instances
from fashion_mm.models.local_region import parse_region_query
from fashion_mm.models.local_region import propose_local_region
from fashion_mm.models.local_region import select_garment_instance


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


def test_select_garment_instance_prefers_query_hint():
    top = FashionInstance(
        mask=np.ones((10, 10), dtype=bool),
        box=(0.0, 0.0, 10.0, 10.0),
        label_id=1,
        label_name="top",
        score=0.99,
    )
    dress = FashionInstance(
        mask=np.ones((10, 10), dtype=bool),
        box=(0.0, 0.0, 10.0, 10.0),
        label_id=5,
        label_name="dress",
        score=0.80,
    )
    segmentation = SegmentationResult(image_size=(10, 10), instances=[top, dress])

    selected = select_garment_instance(
        segmentation,
        parse_region_query("这件连衣裙的领口"),
    )

    assert selected is dress


def test_localize_region_from_instances_returns_region_result():
    mask = np.zeros((100, 80), dtype=bool)
    mask[10:90, 20:60] = True
    instance = FashionInstance(
        mask=mask,
        box=(20.0, 10.0, 60.0, 90.0),
        label_id=1,
        label_name="top",
        score=0.95,
    )
    segmentation = SegmentationResult(image_size=(80, 100), instances=[instance])

    result = localize_region_from_instances(segmentation, "这件上衣的领口")

    assert result.status == "ok"
    assert result.selected_instance is instance
    assert result.proposal is not None
    assert result.proposal.region == "neckline"
    assert result.proposal.mask.sum() > 0
