from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.local_region import box_iou
from scripts.eval.evaluate_local_region_manual_labels import load_manual_records


MANUAL_IOU_THRESHOLDS = (0.3, 0.5)
DEFAULT_MODEL_NAME = "google/owlvit-base-patch32"
BACKEND_NAMES = ("auto", "owlvit", "owlv2")

REGION_PROMPTS = {
    "neckline": ("neckline", "collar", "clothing collar"),
    "collar": ("collar", "neckline", "clothing collar"),
    "cuff": ("sleeve cuff", "cuff", "end of sleeve"),
    "left_cuff": ("left sleeve cuff", "sleeve cuff", "cuff"),
    "right_cuff": ("right sleeve cuff", "sleeve cuff", "cuff"),
    "hem": ("hem", "bottom hem", "lower edge of clothing"),
    "shoulder": ("shoulder", "clothing shoulder", "shoulder seam"),
    "waist": ("waistband", "waist", "waist area"),
    "pocket": ("pocket", "clothing pocket"),
    "left_pocket": ("left pocket", "pocket", "clothing pocket"),
    "right_pocket": ("right pocket", "pocket", "clothing pocket"),
    "zipper": ("zipper", "clothing zipper"),
    "pattern": ("pattern", "print", "floral pattern", "printed pattern"),
    "button": ("button", "clothing button"),
    "decoration": ("decoration", "ornament", "clothing decoration"),
}

QUERY_REGION_HINTS = (
    ("领口", "neckline"),
    ("衣领", "collar"),
    ("袖口", "cuff"),
    ("下摆", "hem"),
    ("肩部", "shoulder"),
    ("肩膀", "shoulder"),
    ("腰部", "waist"),
    ("腰", "waist"),
    ("口袋", "pocket"),
    ("拉链", "zipper"),
    ("碎花", "pattern"),
    ("图案", "pattern"),
    ("印花", "pattern"),
    ("扣子", "button"),
    ("纽扣", "button"),
    ("装饰", "decoration"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pretrained zero-shot grounding model against the "
            "manual 3.1.2 local-region bbox benchmark."
        )
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Manual JSONL with image, query_text, target_region, and target_bbox.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name or local model directory.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_NAMES,
        default="auto",
        help="Model loader backend. Use owlvit/owlv2 for explicit classes.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("english", "chinese", "both"),
        default="english",
        help="Prompt set used for zero-shot grounding.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/local_region_manual_eval_pretrained_grounding.json",
        help="Path to save summary and per-query records.",
    )
    return parser.parse_args()


class HFZeroShotGrounder:
    """Thin wrapper around HuggingFace zero-shot object detection models."""

    def __init__(
        self,
        model_name: str,
        *,
        backend: str = "auto",
        device: str | None = None,
        score_threshold: float = 0.05,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.device_name = device
        self.score_threshold = score_threshold
        self.processor, self.model, self.torch, self.device = load_hf_grounder(
            model_name,
            backend=backend,
            device=device,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image: Image.Image,
        prompts: list[str],
    ) -> dict[str, Any]:
        if not prompts:
            return {"status": "no_prompt", "detections": [], "best": None}
        inputs = self.processor(
            text=[prompts],
            images=image,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor(
            [[image.height, image.width]],
            dtype=self.torch.float,
            device=self.device,
        )
        try:
            processed = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.score_threshold,
            )[0]
        except TypeError:
            processed = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.score_threshold,
            )[0]
        detections = detections_from_hf_output(processed, prompts)
        best = max(detections, key=lambda item: item["score"]) if detections else None
        return {
            "status": "ok" if best is not None else "no_detection",
            "detections": detections,
            "best": best,
        }


def load_hf_grounder(
    model_name: str,
    *,
    backend: str,
    device: str | None,
):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Pretrained grounding evaluation requires torch on AutoDL."
        ) from exc

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    try:
        if backend == "owlvit":
            from transformers import OwlViTForObjectDetection
            from transformers import OwlViTProcessor

            return (
                OwlViTProcessor.from_pretrained(model_name),
                OwlViTForObjectDetection.from_pretrained(model_name),
                torch,
                selected_device,
            )
        if backend == "owlv2":
            from transformers import Owlv2ForObjectDetection
            from transformers import Owlv2Processor

            return (
                Owlv2Processor.from_pretrained(model_name),
                Owlv2ForObjectDetection.from_pretrained(model_name),
                torch,
                selected_device,
            )
        from transformers import AutoModelForZeroShotObjectDetection
        from transformers import AutoProcessor

        return (
            AutoProcessor.from_pretrained(model_name),
            AutoModelForZeroShotObjectDetection.from_pretrained(model_name),
            torch,
            selected_device,
        )
    except ImportError as exc:
        raise ImportError(
            "Pretrained grounding evaluation requires transformers. Install "
            "with: pip install 'transformers>=4.37.0' sentencepiece"
        ) from exc


def detections_from_hf_output(
    processed: dict[str, Any],
    prompts: list[str],
) -> list[dict[str, Any]]:
    scores = processed.get("scores", [])
    labels = processed.get("labels", [])
    boxes = processed.get("boxes", [])
    detections = []
    for score, label, box in zip(scores, labels, boxes, strict=False):
        label_index = int(label.detach().cpu().item() if hasattr(label, "detach") else label)
        score_value = float(score.detach().cpu().item() if hasattr(score, "detach") else score)
        box_values = (
            box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
        )
        prompt = prompts[label_index] if 0 <= label_index < len(prompts) else str(label_index)
        detections.append(
            {
                "prompt": prompt,
                "prompt_index": label_index,
                "score": score_value,
                "bbox": [float(value) for value in box_values],
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def build_prompts(
    query_text: str,
    target_region: str | None = None,
    *,
    prompt_mode: str = "english",
) -> list[str]:
    """Build zero-shot prompts from Chinese query text and parsed region hints."""
    prompts: list[str] = []
    if prompt_mode in {"chinese", "both"}:
        prompts.append(query_text)

    if prompt_mode in {"english", "both"}:
        region = target_region or infer_region_from_query(query_text)
        region_prompts = REGION_PROMPTS.get(region or "", ())
        prompts.extend(region_prompts)
        if "左" in query_text and region in {"cuff", "pocket"}:
            prompts.insert(0, f"left {region}")
        if "右" in query_text and region in {"cuff", "pocket"}:
            prompts.insert(0, f"right {region}")

    return dedupe_preserve_order(prompts)


def infer_region_from_query(query_text: str) -> str | None:
    for keyword, region in QUERY_REGION_HINTS:
        if keyword in query_text:
            return region
    return None


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def evaluate_pretrained_grounding(
    manual_records: list[dict[str, Any]],
    *,
    model_name: str,
    backend: str,
    prompt_mode: str,
    device: str | None,
    score_threshold: float,
) -> list[dict[str, Any]]:
    grounder = HFZeroShotGrounder(
        model_name,
        backend=backend,
        device=device,
        score_threshold=score_threshold,
    )
    image_cache: dict[str, Image.Image] = {}
    records = []
    for manual_record in manual_records:
        image_path = str(manual_record["image"])
        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path).convert("RGB")
        image = image_cache[image_path]
        prompts = build_prompts(
            manual_record["query_text"],
            manual_record.get("target_region"),
            prompt_mode=prompt_mode,
        )
        prediction = grounder.predict(image, prompts)
        best = prediction["best"]
        predicted_box = tuple(best["bbox"]) if best is not None else None
        manual_iou = (
            box_iou(predicted_box, manual_record["target_bbox"])
            if predicted_box is not None
            else 0.0
        )
        records.append(
            {
                "id": manual_record.get("id"),
                "image": image_path,
                "query_text": manual_record["query_text"],
                "target_region": manual_record.get("target_region"),
                "target_bbox": list(manual_record["target_bbox"]),
                "status": prediction["status"],
                "ranker_backend": f"pretrained_grounding_{grounder.backend}",
                "selected_region": best["prompt"] if best is not None else None,
                "predicted_bbox": list(predicted_box) if predicted_box else None,
                "manual_bbox_iou": manual_iou,
                "score": best["score"] if best is not None else None,
                "prompts": prompts,
                "detections": prediction["detections"][:5],
            }
        )
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(record["status"] for record in records)
    selected_regions = Counter(
        record["selected_region"]
        for record in records
        if record.get("selected_region") is not None
    )
    ious = [
        record["manual_bbox_iou"]
        for record in records
        if record.get("manual_bbox_iou") is not None
    ]
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_region[record.get("target_region") or "unknown"].append(record)

    return {
        "num_records": len(records),
        "status_counts": dict(status_counts),
        "selected_region_counts": dict(selected_regions),
        "avg_manual_bbox_iou": mean(ious) if ious else 0.0,
        "manual_hit_at": {
            str(threshold): hit_rate(ious, threshold)
            for threshold in MANUAL_IOU_THRESHOLDS
        },
        "by_region": {
            region: summarize_region(region_records)
            for region, region_records in sorted(by_region.items())
        },
    }


def summarize_region(records: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [
        record["manual_bbox_iou"]
        for record in records
        if record.get("manual_bbox_iou") is not None
    ]
    return {
        "num_records": len(records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "avg_manual_bbox_iou": mean(ious) if ious else 0.0,
        "manual_hit_at": {
            str(threshold): hit_rate(ious, threshold)
            for threshold in MANUAL_IOU_THRESHOLDS
        },
    }


def hit_rate(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(value >= threshold for value in values) / len(values)


def main() -> None:
    args = parse_args()
    manual_records = load_manual_records(args.annotations, max_records=args.max_records)
    if not manual_records:
        raise ValueError("No labeled manual records found for pretrained grounding eval.")

    records = evaluate_pretrained_grounding(
        manual_records,
        model_name=args.model_name,
        backend=args.backend,
        prompt_mode=args.prompt_mode,
        device=args.device,
        score_threshold=args.score_threshold,
    )
    summary = {
        "annotations": str(Path(args.annotations)),
        "model_name": args.model_name,
        "backend": args.backend,
        "prompt_mode": args.prompt_mode,
        "score_threshold": args.score_threshold,
        "num_labeled_records": len(manual_records),
        **summarize_records(records),
        "records": records,
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
