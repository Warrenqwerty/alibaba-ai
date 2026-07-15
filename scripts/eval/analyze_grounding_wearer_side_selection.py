from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.local_region import box_iou
from scripts.eval.evaluate_local_region_manual_labels import summarize_records


DEFAULT_REGIONS = ("cuff", "pocket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline analysis of garment/wearer-side grounding selection using "
            "detections already saved in a gated manual-evaluation JSON."
        )
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument(
        "--min-score-ratio",
        type=float,
        default=0.5,
        help=(
            "A side-compatible candidate must have at least this fraction of the "
            "highest detection score; otherwise keep the original selection."
        ),
    )
    return parser.parse_args()


def query_wearer_side(query_text: str) -> str | None:
    """Return the garment/wearer side named by a Chinese query."""
    if "左" in query_text:
        return "left"
    if "右" in query_text:
        return "right"
    return None


def desired_image_side(wearer_side: str) -> str:
    """Map garment/wearer side to image side for frontal or front-flat views."""
    if wearer_side == "left":
        return "right"
    if wearer_side == "right":
        return "left"
    raise ValueError(f"Unsupported wearer side: {wearer_side}")


def detection_image_side(detection: dict[str, Any], image_width: int) -> str:
    x1, _, x2, _ = [float(value) for value in detection["bbox"]]
    return "left" if (x1 + x2) * 0.5 < image_width * 0.5 else "right"


def select_wearer_side_detection(
    detections: list[dict[str, Any]],
    *,
    query_text: str,
    image_width: int,
    min_score_ratio: float,
) -> tuple[dict[str, Any] | None, str]:
    """Select the best credible detection on the query's garment/wearer side."""
    if not detections:
        return None, "no_detection"
    baseline = max(detections, key=lambda detection: float(detection["score"]))
    wearer_side = query_wearer_side(query_text)
    if wearer_side is None:
        return baseline, "query_has_no_side"
    if not 0.0 <= min_score_ratio <= 1.0:
        raise ValueError("min_score_ratio must be between 0 and 1")

    target_image_side = desired_image_side(wearer_side)
    minimum_score = float(baseline["score"]) * min_score_ratio
    compatible = [
        detection
        for detection in detections
        if float(detection["score"]) >= minimum_score
        and detection_image_side(detection, image_width) == target_image_side
    ]
    if not compatible:
        return baseline, "no_credible_side_candidate"
    selected = max(compatible, key=lambda detection: float(detection["score"]))
    return selected, "side_candidate"


def grounding_detections(record: dict[str, Any]) -> list[dict[str, Any]]:
    detections = record.get("detections")
    if isinstance(detections, list):
        return detections
    fallback_detections = record.get("grounding_detections")
    return fallback_detections if isinstance(fallback_detections, list) else []


def apply_wearer_side_selection(
    records: list[dict[str, Any]],
    *,
    regions: set[str],
    image_widths: dict[str, int],
    min_score_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a fixed side rule and report its per-record effect."""
    output_records = []
    selection_status_counts: Counter[str] = Counter()
    change_counts: Counter[str] = Counter()
    num_side_records = 0
    num_changed = 0

    for record in records:
        updated = dict(record)
        target_region = str(record.get("target_region") or "")
        wearer_side = query_wearer_side(str(record.get("query_text") or ""))
        if target_region not in regions or wearer_side is None:
            output_records.append(updated)
            continue

        num_side_records += 1
        image_path = str(record["image"])
        selected, selection_status = select_wearer_side_detection(
            grounding_detections(record),
            query_text=str(record.get("query_text") or ""),
            image_width=image_widths[image_path],
            min_score_ratio=min_score_ratio,
        )
        selection_status_counts[selection_status] += 1
        if selected is None:
            output_records.append(updated)
            continue

        selected_bbox = [float(value) for value in selected["bbox"]]
        original_bbox = record.get("predicted_bbox")
        changed = original_bbox is None or any(
            abs(float(left) - float(right)) > 1e-6
            for left, right in zip(original_bbox, selected_bbox, strict=False)
        )
        if not changed:
            output_records.append(updated)
            continue

        num_changed += 1
        old_iou = float(record.get("manual_bbox_iou") or 0.0)
        new_iou = box_iou(selected_bbox, record["target_bbox"])
        delta = new_iou - old_iou
        if delta > 1e-9:
            change_counts["improved"] += 1
        elif delta < -1e-9:
            change_counts["regressed"] += 1
        else:
            change_counts["unchanged_iou"] += 1

        updated.update(
            {
                "predicted_bbox": selected_bbox,
                "manual_bbox_iou": new_iou,
                "selected_region": selected.get("prompt"),
                "score": selected.get("score"),
                "wearer_side_selection_status": selection_status,
                "wearer_side": wearer_side,
                "wearer_side_image_target": desired_image_side(wearer_side),
                "wearer_side_iou_delta": delta,
            }
        )
        output_records.append(updated)

    diagnostics = {
        "num_side_records": num_side_records,
        "num_changed": num_changed,
        "selection_status_counts": dict(selection_status_counts),
        "change_counts": dict(change_counts),
    }
    return output_records, diagnostics


def load_eval(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"No records list found in {path}")
    return payload


def load_image_widths(records: list[dict[str, Any]]) -> dict[str, int]:
    widths = {}
    for record in records:
        image_path = str(record["image"])
        if image_path not in widths:
            with Image.open(image_path) as image:
                widths[image_path] = image.width
    return widths


def main() -> None:
    args = parse_args()
    payload = load_eval(args.eval_json)
    records = payload["records"]
    candidate_records, diagnostics = apply_wearer_side_selection(
        records,
        regions=set(args.regions),
        image_widths=load_image_widths(records),
        min_score_ratio=args.min_score_ratio,
    )
    summary = {
        "eval_json": str(Path(args.eval_json)),
        "regions": args.regions,
        "min_score_ratio": args.min_score_ratio,
        "baseline_summary": summarize_records(records),
        "candidate_summary": summarize_records(candidate_records),
        **diagnostics,
        "records": candidate_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
