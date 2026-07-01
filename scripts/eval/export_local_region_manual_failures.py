from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


GT_COLOR = (0, 180, 80)
PRED_COLOR = (230, 60, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export visual failure cases from manual local-region eval JSON."
    )
    parser.add_argument(
        "--eval-json",
        required=True,
        help="Output JSON from evaluate_local_region_manual_labels.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/local_region_manual_failures",
        help="Directory for visualizations and failure_summary.json.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1,
        help="Export records with manual_bbox_iou below this value.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Optional target_region filter, e.g. cuff waist pocket.",
    )
    parser.add_argument("--max-cases", type=int, default=80)
    return parser.parse_args()


def load_eval_records(eval_json_path: str | Path) -> list[dict[str, Any]]:
    """Load per-record manual eval results."""
    payload = json.loads(Path(eval_json_path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"No records list found in {eval_json_path}")
    return records


def select_failure_records(
    records: list[dict[str, Any]],
    *,
    iou_threshold: float,
    regions: set[str] | None = None,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    """Select low-IoU manual eval records for qualitative inspection."""
    selected = []
    for record in records:
        iou = record.get("manual_bbox_iou")
        if iou is None or float(iou) >= iou_threshold:
            continue
        region = str(record.get("target_region") or "unknown")
        if regions is not None and region not in regions:
            continue
        selected.append(record)
    selected.sort(
        key=lambda record: (
            float(record.get("manual_bbox_iou") or 0.0),
            str(record.get("target_region") or ""),
            str(record.get("image") or ""),
        )
    )
    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def draw_failure_record(record: dict[str, Any], output_path: str | Path) -> None:
    """Draw manual target bbox and predicted bbox on the source image."""
    image_path = Path(record["image"])
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    target_bbox = record.get("target_bbox")
    if target_bbox is not None:
        draw_box(draw, target_bbox, GT_COLOR, "GT")

    predicted_bbox = record.get("predicted_bbox")
    if predicted_bbox is not None:
        draw_box(draw, predicted_bbox, PRED_COLOR, "Pred")

    title = (
        f"{record.get('target_region')} | selected={record.get('selected_region')} | "
        f"IoU={float(record.get('manual_bbox_iou') or 0.0):.3f}"
    )
    draw.rectangle([0, 0, min(image.width, 760), 24], fill=(255, 255, 255))
    draw.text((6, 5), title, fill=(20, 20, 20))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: list[float] | tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw one xyxy box."""
    x1, y1, x2, y2 = [float(value) for value in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1 + 3, max(0, y1 - 14)), label, fill=color)


def export_failure_cases(
    eval_json_path: str | Path,
    output_dir: str | Path,
    *,
    iou_threshold: float,
    regions: set[str] | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Export failure visualizations and return a summary."""
    records = load_eval_records(eval_json_path)
    failures = select_failure_records(
        records,
        iou_threshold=iou_threshold,
        regions=regions,
        max_cases=max_cases,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, record in enumerate(failures):
        region = str(record.get("target_region") or "unknown")
        iou = float(record.get("manual_bbox_iou") or 0.0)
        output_path = output / f"{index:03d}_{region}_iou{iou:.3f}_{safe_stem(record.get('id') or Path(record['image']).stem)}.jpg"
        draw_failure_record(record, output_path)
        cases.append(
            {
                "id": record.get("id"),
                "image": record.get("image"),
                "query_text": record.get("query_text"),
                "target_region": region,
                "selected_region": record.get("selected_region"),
                "manual_bbox_iou": iou,
                "target_bbox": record.get("target_bbox"),
                "predicted_bbox": record.get("predicted_bbox"),
                "visualization": str(output_path),
            }
        )

    summary = {
        "eval_json": str(Path(eval_json_path)),
        "output_dir": str(output),
        "iou_threshold": iou_threshold,
        "regions": sorted(regions) if regions else None,
        "num_input_records": len(records),
        "num_exported_cases": len(cases),
        "by_region": dict(Counter(case["target_region"] for case in cases)),
        "cases": cases,
    }
    (output / "failure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def safe_stem(value: Any, max_chars: int = 48) -> str:
    """Return a filesystem-friendly filename stem."""
    text = str(value)[:max_chars]
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return safe.strip("_") or "record"


def main() -> None:
    args = parse_args()
    summary = export_failure_cases(
        args.eval_json,
        args.output_dir,
        iou_threshold=args.iou_threshold,
        regions=set(args.regions) if args.regions else None,
        max_cases=args.max_cases,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
