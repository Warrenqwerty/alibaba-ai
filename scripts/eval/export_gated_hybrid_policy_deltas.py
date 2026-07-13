from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.compare_local_region_manual_evals import record_key
from scripts.eval.export_local_region_manual_failures import safe_stem


GT_COLOR = (0, 180, 80)
BASELINE_COLOR = (230, 60, 50)
CANDIDATE_COLOR = (40, 100, 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export paired manual-benchmark cases where an experimental gated "
            "policy improves or regresses relative to a baseline policy."
        )
    )
    parser.add_argument(
        "--baseline-eval-json",
        required=True,
        help="Manual-eval JSON for the baseline policy, normally heuristic-only.",
    )
    parser.add_argument(
        "--candidate-eval-json",
        required=True,
        help="Manual-eval JSON for the candidate policy, normally gated hybrid.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/local_region_gated_policy_deltas",
        help="Directory for paired visualizations, summary JSON, and HTML.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Optional target_region filter, e.g. pocket pattern cuff.",
    )
    parser.add_argument(
        "--candidate-routes",
        nargs="+",
        default=["grounding"],
        help="Candidate gated_policy_route values to inspect; use --candidate-routes all for every route.",
    )
    parser.add_argument(
        "--min-abs-delta",
        type=float,
        default=0.05,
        help="Only export records where abs(candidate IoU - baseline IoU) reaches this value.",
    )
    parser.add_argument(
        "--max-cases-per-change",
        type=int,
        default=40,
        help="Maximum exported cases for each improved/regressed group.",
    )
    parser.add_argument(
        "--html-name",
        default="policy_delta_review.html",
        help="Filename for the review page inside output-dir.",
    )
    return parser.parse_args()


def load_eval_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {path}")
    return records


def records_by_key(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key in keyed:
            raise ValueError(f"Duplicate manual-eval record key: {key}")
        keyed[key] = record
    return keyed


def paired_policy_deltas(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    regions: set[str] | None = None,
    candidate_routes: set[str] | None = None,
    min_abs_delta: float = 0.05,
) -> list[dict[str, Any]]:
    """Pair matching records and label material policy improvements/regressions."""
    baseline_by_key = records_by_key(baseline_records)
    candidate_by_key = records_by_key(candidate_records)
    pairs: list[dict[str, Any]] = []
    for key in sorted(set(baseline_by_key) & set(candidate_by_key)):
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]
        region = str(candidate.get("target_region") or "unknown")
        route = str(candidate.get("gated_policy_route") or "unknown")
        if regions is not None and region not in regions:
            continue
        if candidate_routes is not None and route not in candidate_routes:
            continue

        baseline_iou = manual_iou(baseline)
        candidate_iou = manual_iou(candidate)
        delta = candidate_iou - baseline_iou
        if abs(delta) < min_abs_delta:
            continue
        pairs.append(
            {
                "key": key,
                "id": candidate.get("id") or baseline.get("id"),
                "image": candidate.get("image") or baseline.get("image"),
                "query_text": candidate.get("query_text") or baseline.get("query_text"),
                "target_region": region,
                "target_bbox": candidate.get("target_bbox") or baseline.get("target_bbox"),
                "candidate_route": route,
                "change": "improved" if delta > 0 else "regressed",
                "iou_delta": delta,
                "baseline": compact_record(baseline),
                "candidate": compact_record(candidate),
            }
        )
    return sorted(
        pairs,
        key=lambda item: (
            item["change"],
            -abs(float(item["iou_delta"])),
            str(item["target_region"]),
            str(item["image"]),
        ),
    )


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "manual_bbox_iou": manual_iou(record),
        "selected_region": record.get("selected_region"),
        "predicted_bbox": record.get("predicted_bbox"),
        "status": record.get("status"),
        "score": record.get("score"),
        "prompts": record.get("prompts"),
        "ranker_backend": record.get("ranker_backend"),
    }


def manual_iou(record: dict[str, Any]) -> float:
    value = record.get("manual_bbox_iou")
    return float(value) if value is not None else 0.0


def limited_change_groups(
    pairs: list[dict[str, Any]],
    *,
    max_cases_per_change: int | None,
) -> list[dict[str, Any]]:
    """Keep the largest changes from each direction, ordered for visual review."""
    selected: list[dict[str, Any]] = []
    for change in ("improved", "regressed"):
        group = [pair for pair in pairs if pair["change"] == change]
        group.sort(key=lambda pair: abs(float(pair["iou_delta"])), reverse=True)
        if max_cases_per_change is not None:
            group = group[:max_cases_per_change]
        selected.extend(group)
    return selected


def draw_policy_delta(pair: dict[str, Any], output_path: str | Path) -> None:
    """Draw manual GT, baseline prediction, and candidate prediction together."""
    image_path = Path(str(pair["image"]))
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw_box(draw, pair.get("target_bbox"), GT_COLOR, "GT")
    draw_box(draw, pair["baseline"].get("predicted_bbox"), BASELINE_COLOR, "heuristic")
    draw_box(draw, pair["candidate"].get("predicted_bbox"), CANDIDATE_COLOR, "gated")
    title = (
        f"{pair['target_region']} | {pair['change']} | "
        f"heuristic={pair['baseline']['manual_bbox_iou']:.3f} "
        f"gated={pair['candidate']['manual_bbox_iou']:.3f} "
        f"delta={pair['iou_delta']:+.3f}"
    )
    draw.rectangle([0, 0, min(image.width, 900), 25], fill=(255, 255, 255))
    draw.text((6, 5), title, fill=(20, 20, 20))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: list[float] | tuple[float, float, float, float] | None,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [float(value) for value in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1 + 3, max(0, y1 - 14)), label, fill=color)


def export_policy_deltas(
    baseline_eval_json: str | Path,
    candidate_eval_json: str | Path,
    output_dir: str | Path,
    *,
    regions: set[str] | None = None,
    candidate_routes: set[str] | None = None,
    min_abs_delta: float = 0.05,
    max_cases_per_change: int | None = 40,
) -> dict[str, Any]:
    """Create a paired qualitative review for an experimental policy change."""
    baseline_records = load_eval_records(baseline_eval_json)
    candidate_records = load_eval_records(candidate_eval_json)
    pairs = paired_policy_deltas(
        baseline_records,
        candidate_records,
        regions=regions,
        candidate_routes=candidate_routes,
        min_abs_delta=min_abs_delta,
    )
    selected = limited_change_groups(
        pairs,
        max_cases_per_change=max_cases_per_change,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, pair in enumerate(selected):
        filename = (
            f"{index:03d}_{pair['change']}_{pair['target_region']}_"
            f"delta{pair['iou_delta']:+.3f}_{safe_stem(pair.get('id') or pair['key'])}.jpg"
        )
        output_path = output / filename
        draw_policy_delta(pair, output_path)
        cases.append({**pair, "visualization": str(output_path)})

    summary = {
        "baseline_eval_json": str(Path(baseline_eval_json)),
        "candidate_eval_json": str(Path(candidate_eval_json)),
        "output_dir": str(output),
        "regions": sorted(regions) if regions else None,
        "candidate_routes": sorted(candidate_routes) if candidate_routes else "all",
        "min_abs_delta": min_abs_delta,
        "num_baseline_records": len(baseline_records),
        "num_candidate_records": len(candidate_records),
        "num_common_records": len(set(records_by_key(baseline_records)) & set(records_by_key(candidate_records))),
        "num_material_deltas": len(pairs),
        "num_exported_cases": len(cases),
        "change_counts": dict(Counter(pair["change"] for pair in pairs)),
        "by_region": summarize_by_region(pairs),
        "cases": cases,
    }
    (output / "policy_delta_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def summarize_by_region(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(str(pair["target_region"]), []).append(pair)
    return {
        region: {
            "num_material_deltas": len(region_pairs),
            "avg_iou_delta": mean(float(pair["iou_delta"]) for pair in region_pairs),
            "change_counts": dict(Counter(pair["change"] for pair in region_pairs)),
        }
        for region, region_pairs in sorted(grouped.items())
    }


def write_policy_delta_html(summary: dict[str, Any], output_path: str | Path) -> None:
    """Write a compact HTML gallery grouped by improvement and regression."""
    output = Path(output_path)
    sections = []
    for change in ("improved", "regressed"):
        cases = [case for case in summary.get("cases", []) if case["change"] == change]
        cards = "\n".join(render_case_card(case, output.parent) for case in cases)
        sections.append(
            f"""
            <section>
              <h2>{escape(change.title())} <span>{len(cases)} cases</span></h2>
              <div class=\"grid\">{cards or '<p>No material cases exported.</p>'}</div>
            </section>
            """
        )
    page = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <title>Gated Policy Delta Review</title>
  <style>
    body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2328; background: #f6f8fa; }}
    h1 {{ margin-bottom: 4px; font-size: 24px; }}
    .meta {{ margin-bottom: 24px; color: #57606a; }}
    section {{ margin: 28px 0; }}
    h2 {{ font-size: 18px; }} h2 span {{ color: #57606a; font-size: 14px; font-weight: 500; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #d0d7de; border-radius: 8px; background: #fff; overflow: hidden; }}
    .card img {{ display: block; width: 100%; height: auto; background: #fff; }}
    .info {{ padding: 10px 12px 12px; font-size: 13px; line-height: 1.45; }}
    .query {{ font-weight: 600; margin-bottom: 6px; }} .kv {{ color: #57606a; }} code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Gated Policy Delta Review</h1>
  <div class=\"meta\">
    Exported {int(summary.get('num_exported_cases') or 0)} material deltas from
    {int(summary.get('num_common_records') or 0)} paired manual records;
    |IoU delta| &gt;= {float(summary.get('min_abs_delta') or 0.0):.3f}.
    Green = manual bbox, red = heuristic, blue = gated policy.
  </div>
  {''.join(sections)}
</body>
</html>
"""
    output.write_text(page, encoding="utf-8")


def render_case_card(case: dict[str, Any], html_dir: Path) -> str:
    visualization = Path(str(case.get("visualization") or ""))
    try:
        image_src = visualization.relative_to(html_dir).as_posix()
    except ValueError:
        image_src = visualization.as_posix()
    baseline = case["baseline"]
    candidate = case["candidate"]
    prompts = candidate.get("prompts") or []
    return f"""
      <article class=\"card\">
        <img src=\"{escape(image_src)}\" alt=\"{escape(case.get('id') or '')}\">
        <div class=\"info\">
          <div class=\"query\">{escape(case.get('query_text') or '')}</div>
          <div class=\"kv\">target: <code>{escape(case.get('target_region') or '')}</code>; route: <code>{escape(case.get('candidate_route') or '')}</code></div>
          <div class=\"kv\">heuristic IoU: {float(baseline['manual_bbox_iou']):.3f}; gated IoU: {float(candidate['manual_bbox_iou']):.3f}; delta: {float(case['iou_delta']):+.3f}</div>
          <div class=\"kv\">heuristic selected: <code>{escape(baseline.get('selected_region') or '')}</code></div>
          <div class=\"kv\">grounding prompt: <code>{escape(', '.join(str(prompt) for prompt in prompts))}</code></div>
          <div class=\"kv\">grounding score: {format_optional_float(candidate.get('score'))}</div>
          <div class=\"kv\">id: <code>{escape(case.get('id') or '')}</code></div>
        </div>
      </article>
    """


def format_optional_float(value: Any) -> str:
    return f"{float(value):.3f}" if value is not None else "n/a"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def main() -> None:
    args = parse_args()
    candidate_routes = None if args.candidate_routes == ["all"] else set(args.candidate_routes)
    summary = export_policy_deltas(
        args.baseline_eval_json,
        args.candidate_eval_json,
        args.output_dir,
        regions=set(args.regions) if args.regions else None,
        candidate_routes=candidate_routes,
        min_abs_delta=args.min_abs_delta,
        max_cases_per_change=args.max_cases_per_change,
    )
    html_path = Path(args.output_dir) / args.html_name
    write_policy_delta_html(summary, html_path)
    summary["html"] = str(html_path)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
