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
    "zipper": ("拉链", "拉锁"),
    "button": ("纽扣", "扣子"),
    "decoration": ("装饰", "珠片", "亮片", "刺绣", "蝴蝶结"),
}

GARMENT_KEYWORDS = {
    "top": ("上衣", "衣服", "短袖", "长袖", "衬衫", "T恤", "t恤"),
    "pants": ("裤子", "长裤", "短裤", "裤"),
    "skirt": ("裙子", "半身裙"),
    "outerwear": ("外套", "大衣", "夹克"),
    "dress": ("连衣裙", "裙装"),
}

SPATIAL_KEYWORDS = {
    "left": ("左边", "左侧", "左面", "左"),
    "right": ("右边", "右侧", "右面", "右"),
    "upper": ("上方", "上面", "上半", "顶部"),
    "lower": ("下方", "下面", "下半", "底部"),
    "center": ("中间", "中央", "正面", "前面"),
}

ATTRIBUTE_KEYWORDS = {
    "floral": ("碎花", "花朵", "花卉"),
    "stripe": ("条纹", "竖条", "横条"),
    "plaid": ("格纹", "格子"),
    "shiny": ("亮片", "珠片", "闪光"),
    "embroidered": ("刺绣", "绣花"),
}

RELATION_KEYWORDS = {
    "outer": ("外套", "外层", "外面"),
    "inner": ("内搭", "里面", "内层", "里层"),
}

OPEN_VOCAB_REGIONS = set(REGION_KEYWORDS)


@dataclass(frozen=True)
class ParsedRegionQuery:
    """Structured representation of a language-guided local-region query."""

    query: str
    region: str | None
    garment_hint: str | None
    is_supported_region: bool
    spatial_hints: tuple[str, ...]
    attribute_hints: tuple[str, ...]
    relation_hints: tuple[str, ...]


def parse_region_query(query: str) -> ParsedRegionQuery:
    """Parse Chinese region and garment hints from a natural-language query."""
    normalized = query.strip()
    region = _first_keyword_match(normalized, REGION_KEYWORDS)
    garment_hint = _first_keyword_match(normalized, GARMENT_KEYWORDS)
    return ParsedRegionQuery(
        query=query,
        region=region,
        garment_hint=garment_hint,
        is_supported_region=region in OPEN_VOCAB_REGIONS,
        spatial_hints=_all_keyword_matches(normalized, SPATIAL_KEYWORDS),
        attribute_hints=_all_keyword_matches(normalized, ATTRIBUTE_KEYWORDS),
        relation_hints=_all_keyword_matches(normalized, RELATION_KEYWORDS),
    )


def _first_keyword_match(text: str, keyword_map: dict[str, tuple[str, ...]]) -> str | None:
    for name, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return name
    return None


def _all_keyword_matches(text: str, keyword_map: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(
        name
        for name, keywords in keyword_map.items()
        if any(keyword in text for keyword in keywords)
    )
