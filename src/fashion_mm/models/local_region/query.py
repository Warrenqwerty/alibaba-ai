from __future__ import annotations

from dataclasses import dataclass


REGION_KEYWORDS = {
    "neckline": ("领口", "衣领", "领型", "领子"),
    "cuff": ("袖口", "袖子末端", "袖边"),
    "hem": ("下摆", "衣摆", "裙摆", "裤脚"),
    "pocket": ("口袋", "袋口"),
    "shoulder": ("肩部", "肩线", "肩膀"),
    "waist": ("腰部", "腰线", "收腰"),
    "pattern": ("图案", "印花", "花纹", "纹理"),
    "decoration": ("装饰", "纽扣", "扣子", "拉链", "珠片", "亮片"),
}

GARMENT_KEYWORDS = {
    "top": ("上衣", "衣服", "短袖", "长袖", "衬衫", "T恤", "t恤"),
    "pants": ("裤子", "长裤", "短裤", "裤"),
    "skirt": ("裙子", "半身裙"),
    "outerwear": ("外套", "大衣", "夹克"),
    "dress": ("连衣裙", "裙装"),
}


@dataclass(frozen=True)
class ParsedRegionQuery:
    """Structured representation of a language-guided local-region query."""

    query: str
    region: str | None
    garment_hint: str | None
    is_supported_region: bool


def parse_region_query(query: str) -> ParsedRegionQuery:
    """Parse Chinese region and garment hints from a natural-language query."""
    normalized = query.strip()
    region = _first_keyword_match(normalized, REGION_KEYWORDS)
    garment_hint = _first_keyword_match(normalized, GARMENT_KEYWORDS)
    return ParsedRegionQuery(
        query=query,
        region=region,
        garment_hint=garment_hint,
        is_supported_region=region in {"neckline", "cuff", "hem", "shoulder", "waist", "pattern"},
    )


def _first_keyword_match(text: str, keyword_map: dict[str, tuple[str, ...]]) -> str | None:
    for name, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return name
    return None
