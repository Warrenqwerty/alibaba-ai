from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mm.models.local_region import box_iou
from fashion_mm.models.local_region import desired_image_side
from fashion_mm.models.local_region import query_wearer_side
from scripts.eval.analyze_grounding_candidate_oracle import build_candidate_oracle
from scripts.eval.analyze_grounding_candidate_oracle import manual_candidates
from scripts.eval.evaluate_local_region_manual_labels import summarize_records


DEFAULT_REGIONS = ("cuff", "pocket", "pattern", "waist", "zipper")
SOURCE_NAMES = ("current", "grounding", "diagnostic_grounding", "heuristic")
MODEL_NAMES = ("heuristic", "grounding_dino_tiny", "grounding_dino_base", "owlv2")
TEXT_BUCKETS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a manual-label candidate selector with image-grouped "
            "cross-validation. Every reported prediction is out-of-fold."
        )
    )
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--selection-policy",
        choices=("listwise", "conservative_pairwise"),
        default="listwise",
    )
    parser.add_argument(
        "--override-threshold",
        type=float,
        default=0.5,
        help="Minimum pairwise recovery probability required to replace current.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


class ManualCandidateSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def canonical_model_name(model_name: str | None, source: str) -> str:
    normalized = (model_name or "").lower()
    if source in {"current", "heuristic"} and not normalized:
        return "heuristic"
    if "grounding-dino-tiny" in normalized:
        return "grounding_dino_tiny"
    if "grounding-dino-base" in normalized:
        return "grounding_dino_base"
    if "owlv2" in normalized:
        return "owlv2"
    return "heuristic" if source == "heuristic" else "grounding_dino_base"


def candidate_model_name(record: dict[str, Any], source: str) -> str:
    if source == "diagnostic_grounding":
        diagnostic = record.get("diagnostic_grounding_candidate")
        model_name = diagnostic.get("grounding_model_name") if isinstance(diagnostic, dict) else None
        return canonical_model_name(model_name, source)
    if source == "grounding":
        return canonical_model_name(record.get("grounding_model_name"), source)
    if source == "current" and "grounding" in str(record.get("gated_policy_route") or ""):
        return canonical_model_name(record.get("grounding_model_name"), source)
    return "heuristic"


def candidate_key(box: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), 3) for value in box)


def selector_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    selected_box = record.get("predicted_bbox")
    if selected_box is not None:
        candidates.append(
            {
                "bbox": [float(value) for value in selected_box],
                "prompt": record.get("selected_region"),
                "score": record.get("score"),
                "candidate_source": "current",
                "candidate_rank": 0,
            }
        )
    candidates.extend(manual_candidates(record))

    deduplicated = []
    seen_boxes = set()
    for candidate in candidates:
        box = candidate.get("bbox")
        if box is None:
            continue
        key = candidate_key(box)
        if key in seen_boxes:
            continue
        seen_boxes.add(key)
        deduplicated.append(candidate)
    return deduplicated


def one_hot(value: str, names: tuple[str, ...]) -> list[float]:
    return [float(value == name) for name in names]


def hash_text(text: str | None, num_buckets: int = TEXT_BUCKETS) -> list[float]:
    vector = [0.0] * num_buckets
    normalized = (text or "").strip().lower()
    if not normalized:
        return vector
    for character in normalized:
        vector[ord(character) % num_buckets] += 1.0
    return [value / len(normalized) for value in vector]


def side_features(
    query_text: str,
    box: list[float] | tuple[float, ...],
    image_width: int,
) -> tuple[float, float]:
    wearer_side = query_wearer_side(query_text)
    if wearer_side is None:
        return 0.0, 0.0
    x1, _, x2, _ = [float(value) for value in box]
    image_side = "left" if (x1 + x2) * 0.5 < image_width * 0.5 else "right"
    return 1.0, float(image_side == desired_image_side(wearer_side))


def candidate_feature(
    record: dict[str, Any],
    candidate: dict[str, Any],
    *,
    image_size: tuple[int, int],
    max_source_score: float,
    current_box: list[float] | None,
    heuristic_box: list[float] | None,
) -> torch.Tensor:
    image_width, image_height = image_size
    x1, y1, x2, y2 = [float(value) for value in candidate["bbox"]]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    source = str(candidate.get("candidate_source") or "current")
    model_name = candidate_model_name(record, source)
    region = str(record.get("target_region") or "")
    score_value = candidate.get("score")
    score = float(score_value) if score_value is not None else 0.0
    rank_value = candidate.get("candidate_rank")
    rank = float(rank_value) if rank_value is not None else 0.0
    has_side, side_matches = side_features(
        str(record.get("query_text") or ""),
        candidate["bbox"],
        image_width,
    )
    source_region = [
        float(source == source_name and region == region_name)
        for source_name in SOURCE_NAMES
        for region_name in DEFAULT_REGIONS
    ]
    geometry = [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height,
        width / image_width,
        height / image_height,
        (width * height) / (image_width * image_height),
        math.log(width / height),
        abs((x1 + x2) * 0.5 / image_width - 0.5),
    ]
    agreement = [
        box_iou(candidate["bbox"], current_box) if current_box is not None else 0.0,
        box_iou(candidate["bbox"], heuristic_box) if heuristic_box is not None else 0.0,
        float(heuristic_box is not None),
    ]
    numeric = [
        score,
        float(score_value is None),
        score / max(max_source_score, 1e-6) if score_value is not None else 0.0,
        rank / 5.0,
        float(rank_value is None),
        has_side,
        side_matches,
        *geometry,
        *agreement,
    ]
    values = [
        *one_hot(region, DEFAULT_REGIONS),
        *one_hot(source, SOURCE_NAMES),
        *one_hot(model_name, MODEL_NAMES),
        *source_region,
        *numeric,
        *hash_text(candidate.get("prompt")),
    ]
    return torch.tensor(values, dtype=torch.float32)


def candidate_examples(
    record: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    candidates = selector_candidates(record)
    if not candidates:
        raise ValueError(f"Record has no selectable candidate: {record.get('id')}")
    source_scores: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        score = candidate.get("score")
        if score is not None:
            source = str(candidate.get("candidate_source") or "current")
            source_scores[source] = max(source_scores[source], float(score))
    heuristic = record.get("heuristic_candidate")
    heuristic_box = (
        heuristic.get("predicted_bbox")
        if isinstance(heuristic, dict)
        else None
    )
    current_box = record.get("predicted_bbox")
    features = torch.stack(
        [
            candidate_feature(
                record,
                candidate,
                image_size=image_size,
                max_source_score=source_scores[
                    str(candidate.get("candidate_source") or "current")
                ],
                current_box=current_box,
                heuristic_box=heuristic_box,
            )
            for candidate in candidates
        ]
    )
    ious = torch.tensor(
        [box_iou(candidate["bbox"], record["target_bbox"]) for candidate in candidates],
        dtype=torch.float32,
    )
    return features, ious, candidates


def image_grouped_folds(
    records: list[dict[str, Any]],
    *,
    num_folds: int,
    seed: int,
) -> list[list[int]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    by_image: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_image[str(record["image"])].append(index)
    if len(by_image) < num_folds:
        raise ValueError("num_folds cannot exceed the number of unique images")

    groups = list(by_image.items())
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    folds: list[list[int]] = [[] for _ in range(num_folds)]
    for _, indices in groups:
        target_fold = min(range(num_folds), key=lambda fold: len(folds[fold]))
        folds[target_fold].extend(indices)
    return folds


def listwise_hit_loss(scores: torch.Tensor, ious: torch.Tensor) -> torch.Tensor:
    hits = ious >= 0.3
    if bool(hits.any()):
        target = hits.float() * (0.5 + ious)
        target = target / target.sum()
    else:
        target = torch.softmax(ious / 0.08, dim=0)
    return -(target * torch.log_softmax(scores, dim=0)).sum()


def train_selector(
    examples: list[tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    train_indices: list[int],
    *,
    hidden_dim: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> ManualCandidateSelector:
    torch.manual_seed(seed)
    input_dim = examples[train_indices[0]][0].shape[1]
    model = ManualCandidateSelector(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = random.Random(seed)
    for _ in range(num_epochs):
        epoch_indices = list(train_indices)
        generator.shuffle(epoch_indices)
        model.train()
        optimizer.zero_grad()
        losses = []
        for index in epoch_indices:
            features, ious, _ = examples[index]
            scores = model(features.to(device))
            losses.append(listwise_hit_loss(scores, ious.to(device)))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def pairwise_recovery_examples(
    examples: list[tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_features = []
    labels = []
    for index in indices:
        features, ious, candidates = examples[index]
        if not candidates or candidates[0].get("candidate_source") != "current":
            continue
        current_feature = features[0]
        current_is_hit = float(ious[0]) >= 0.3
        for candidate_index in range(1, len(candidates)):
            candidate_feature_vector = features[candidate_index]
            pair_features.append(
                torch.cat(
                    [
                        candidate_feature_vector,
                        current_feature,
                        candidate_feature_vector - current_feature,
                    ]
                )
            )
            candidate_is_hit = float(ious[candidate_index]) >= 0.3
            labels.append(float(candidate_is_hit and not current_is_hit))
    if not pair_features:
        raise ValueError("No current-versus-candidate pairs available for training")
    return torch.stack(pair_features), torch.tensor(labels, dtype=torch.float32)


def train_conservative_selector(
    examples: list[tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]],
    train_indices: list[int],
    *,
    hidden_dim: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> ManualCandidateSelector:
    torch.manual_seed(seed)
    features, labels = pairwise_recovery_examples(examples, train_indices)
    model = ManualCandidateSelector(features.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    num_positive = max(float(labels.sum()), 1.0)
    num_negative = max(float(len(labels) - labels.sum()), 1.0)
    positive_weight = torch.tensor(
        math.sqrt(num_negative / num_positive),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    features = features.to(device)
    labels = labels.to(device)
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def select_candidate_record(
    record: dict[str, Any],
    example: tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]],
    model: ManualCandidateSelector,
    device: torch.device,
) -> dict[str, Any]:
    features, ious, candidates = example
    with torch.no_grad():
        scores = model(features.to(device)).cpu()
    selected_index = int(torch.argmax(scores).item())
    candidate = candidates[selected_index]
    selected = dict(record)
    selected.update(
        {
            "predicted_bbox": [float(value) for value in candidate["bbox"]],
            "manual_bbox_iou": float(ious[selected_index].item()),
            "selected_region": candidate.get("prompt"),
            "selector_source": candidate.get("candidate_source"),
            "selector_rank": candidate.get("candidate_rank"),
            "selector_score": float(scores[selected_index].item()),
        }
    )
    return selected


def select_conservative_candidate_record(
    record: dict[str, Any],
    example: tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]],
    model: ManualCandidateSelector,
    device: torch.device,
    *,
    override_threshold: float,
) -> dict[str, Any]:
    if not 0.0 <= override_threshold <= 1.0:
        raise ValueError("override_threshold must be between 0 and 1")
    features, ious, candidates = example
    if not candidates or candidates[0].get("candidate_source") != "current":
        selected = dict(record)
        selected["selector_source"] = "no_current"
        selected["selector_override_probability"] = 0.0
        selected["selector_overrode_current"] = False
        return selected
    if len(candidates) == 1:
        selected = dict(record)
        selected["selector_source"] = "current"
        selected["selector_override_probability"] = 0.0
        selected["selector_overrode_current"] = False
        return selected

    current_feature = features[0]
    pair_features = torch.stack(
        [
            torch.cat(
                [
                    candidate_feature_vector,
                    current_feature,
                    candidate_feature_vector - current_feature,
                ]
            )
            for candidate_feature_vector in features[1:]
        ]
    )
    with torch.no_grad():
        probabilities = torch.sigmoid(model(pair_features.to(device))).cpu()
    best_alternative = int(torch.argmax(probabilities).item()) + 1
    best_probability = float(probabilities[best_alternative - 1].item())
    selected_index = best_alternative if best_probability >= override_threshold else 0
    candidate = candidates[selected_index]
    selected = dict(record)
    selected.update(
        {
            "predicted_bbox": [float(value) for value in candidate["bbox"]],
            "manual_bbox_iou": float(ious[selected_index].item()),
            "selected_region": candidate.get("prompt"),
            "selector_source": candidate.get("candidate_source"),
            "selector_rank": candidate.get("candidate_rank"),
            "selector_override_probability": best_probability,
            "selector_overrode_current": selected_index != 0,
        }
    )
    return selected


def selector_diagnostics(
    baseline_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    source_counts = Counter(record.get("selector_source") for record in selected_records)
    override_counts = Counter(
        "overrode_current" if record.get("selector_overrode_current") else "kept_current"
        for record in selected_records
        if "selector_overrode_current" in record
    )
    transitions: Counter[str] = Counter()
    changes: Counter[str] = Counter()
    by_region: dict[str, Counter[str]] = defaultdict(Counter)
    for baseline, selected in zip(baseline_records, selected_records, strict=True):
        baseline_iou = float(baseline.get("manual_bbox_iou") or 0.0)
        selected_iou = float(selected.get("manual_bbox_iou") or 0.0)
        region_counts = by_region[str(baseline.get("target_region") or "unknown")]
        if baseline_iou < 0.3 <= selected_iou:
            transitions["gained_hit"] += 1
            region_counts["gained_hit"] += 1
        elif selected_iou < 0.3 <= baseline_iou:
            transitions["lost_hit"] += 1
            region_counts["lost_hit"] += 1
        else:
            transitions["same_hit_status"] += 1
            region_counts["same_hit_status"] += 1
        if selected_iou > baseline_iou + 1e-9:
            changes["improved_iou"] += 1
            region_counts["improved_iou"] += 1
        elif selected_iou < baseline_iou - 1e-9:
            changes["regressed_iou"] += 1
            region_counts["regressed_iou"] += 1
        else:
            changes["same_iou"] += 1
            region_counts["same_iou"] += 1
    return {
        "selected_source_counts": dict(source_counts),
        "override_counts": dict(override_counts),
        "hit_transition_counts": dict(transitions),
        "iou_change_counts": dict(changes),
        "by_region": {
            region: dict(counts) for region, counts in sorted(by_region.items())
        },
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    all_records = payload.get("records")
    if not isinstance(all_records, list):
        raise ValueError(f"No records list found in {args.eval_json}")
    regions = set(args.regions)
    selected_indices = [
        index
        for index, record in enumerate(all_records)
        if str(record.get("target_region") or "") in regions
    ]
    selected_records = [all_records[index] for index in selected_indices]
    image_sizes = {}
    for record in selected_records:
        image_path = str(record["image"])
        if image_path not in image_sizes:
            with Image.open(image_path) as image:
                image_sizes[image_path] = image.size
    examples = [
        candidate_examples(record, image_sizes[str(record["image"])])
        for record in selected_records
    ]
    folds = image_grouped_folds(
        selected_records,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    device = torch.device(args.device)
    oof_records: list[dict[str, Any] | None] = [None] * len(selected_records)
    fold_summaries = []
    all_indices = set(range(len(selected_records)))
    for fold_index, test_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(test_indices))
        train_function = (
            train_conservative_selector
            if args.selection_policy == "conservative_pairwise"
            else train_selector
        )
        model = train_function(
            examples,
            train_indices,
            hidden_dim=args.hidden_dim,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + fold_index,
            device=device,
        )
        fold_records = []
        for index in test_indices:
            if args.selection_policy == "conservative_pairwise":
                selected = select_conservative_candidate_record(
                    selected_records[index],
                    examples[index],
                    model,
                    device,
                    override_threshold=args.override_threshold,
                )
            else:
                selected = select_candidate_record(
                    selected_records[index],
                    examples[index],
                    model,
                    device,
                )
            selected["selector_fold"] = fold_index
            oof_records[index] = selected
            fold_records.append(selected)
        fold_baseline_records = [selected_records[index] for index in test_indices]
        fold_summaries.append(
            {
                "fold": fold_index,
                "num_train_records": len(train_indices),
                "num_test_records": len(test_indices),
                "baseline_summary": summarize_records(fold_baseline_records),
                "out_of_fold_summary": summarize_records(fold_records),
            }
        )
    if any(record is None for record in oof_records):
        raise RuntimeError("Cross-validation did not produce every out-of-fold record")
    finalized_oof_records = [record for record in oof_records if record is not None]

    full_oof_records = [dict(record) for record in all_records]
    for source_index, selected in zip(selected_indices, finalized_oof_records, strict=True):
        full_oof_records[source_index] = selected
    oracle_records, oracle_diagnostics = build_candidate_oracle(
        all_records,
        regions=regions,
        hit_threshold=0.3,
    )
    result = {
        "eval_json": str(Path(args.eval_json)),
        "regions": args.regions,
        "num_folds": args.num_folds,
        "num_epochs": args.num_epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "selection_policy": args.selection_policy,
        "override_threshold": args.override_threshold,
        "seed": args.seed,
        "split_policy": "image_grouped_cross_validation",
        "baseline_summary": summarize_records(all_records),
        "candidate_oracle_summary": summarize_records(oracle_records),
        "candidate_oracle_diagnostics": oracle_diagnostics,
        "out_of_fold_summary": summarize_records(full_oof_records),
        "selector_diagnostics": selector_diagnostics(
            selected_records,
            finalized_oof_records,
        ),
        "fold_summaries": fold_summaries,
        "records": full_oof_records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
