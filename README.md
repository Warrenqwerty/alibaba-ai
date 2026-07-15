# Alibaba AI Fashion Multimodal Project

## Project Goal

Fine-grained fashion visual understanding and multimodal reasoning system.

## Current Stage

Stage 1: Fine-grained visual foundation module.

Current target:

- 3.1.1 Fashion instance segmentation
- 3.1.2 Language-guided local-region localization
- Input: RGB fashion image
- Output: clothing instance masks, local-region masks, bounding boxes, category labels
- Classes: top, pants, skirt, outerwear, dress, shoes, bag, accessory
- Target: single-image latency <= 50 ms, mask IoU >= 0.85

## Repository Structure

- `configs/`: configuration files
- `src/`: reusable project source code
- `scripts/`: executable scripts
- `docs/`: setup notes and project documentation
- `docker/`: Docker environment files
- `tests/`: unit tests

## Environment

```bash
pip install -r requirements.txt
pip install -e .
```

## 3.1.1 Usage

Train on AutoDL:

```bash
python scripts/train/train_instance_segmentation.py \
  --model-config configs/model/instance_segmentation_deepfashion2.yaml \
  --paths-config configs/paths.autodl.yaml
```

Current best DeepFashion2 checkpoint:

```bash
/root/autodl-tmp/checkpoints/deepfashion2_6class_soft_aug_epoch2/instance_segmentation/epoch_001.pt
```

Full validation result on DeepFashion2 validation:

- Images: 32,153
- Ground-truth instances: 52,490
- Mean best mask IoU: 0.8547
- Recall@0.75: 0.8937
- Inference thresholds: score >= 0.3, mask >= 0.4

Run inference with a trained checkpoint:

```bash
python scripts/inference/predict_instance_segmentation.py image.jpg \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_soft_aug_epoch2/instance_segmentation/epoch_001.pt \
  --device cuda
```

## 3.1.2 Usage

Run language-guided local-region localization:

```bash
python scripts/inference/predict_local_region.py image.jpg "右侧的口袋" \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output outputs/local_region_sample.json \
  --vis-output outputs/local_region_sample.jpg
```

Run a small AutoDL sanity evaluation:

```bash
python scripts/eval/evaluate_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --max-images 20 \
  --output /root/autodl-tmp/outputs/local_region_query_eval.json \
  --vis-dir /root/autodl-tmp/outputs/local_region_vis
```

### 3.1.2 Current Plan

The PRD direction is language-guided grounding: image plus natural-language
query should return a local-region mask and bbox. DeepFashion2 provides garment
masks, boxes, categories, and landmarks, but it does not provide query-level
human labels such as "右侧口袋" -> bbox/mask. Therefore the current 3.1.2 plan is:

1. Keep the heuristic open-vocabulary pipeline as the online baseline.
2. Use a small manual bbox benchmark as the main evaluation signal.
3. Add pretrained grounding / vision-language baselines next, such as
   GroundingDINO or OWL-ViT, and Chinese/translated CLIP-style reranking.
4. Treat landmark pseudo-label and weak-ranker results as diagnostics only.
   They are useful for exploration, but they are not enough to prove PRD
   language-guided localization accuracy.

Run weak-label evaluation with DeepFashion2 annotations only as a diagnostic:

```bash
python scripts/eval/evaluate_local_region_weak_labels.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/validation/annos \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --max-images 50 \
  --output /root/autodl-tmp/outputs/local_region_weak_eval.json
```

Do not use weak-label IoU as the final PRD metric. It depends on landmark
pseudo-labels plus rule fallback and can reward geometry that does not match
human-labeled query intent.

Build a small manual bbox benchmark for the true 3.1.2 metric. This is not a
full DeepFashion2 relabeling task; label about 100-300 image-query pairs and use
them only for evaluation, not training:

```bash
PYTHONPATH=src python scripts/data/build_local_region_manual_eval_manifest.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/validation/annos \
  --max-images 50 \
  --max-records 150 \
  --shuffle \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_manifest.jsonl
```

When `--anno-dir` is provided, the manifest uses class-aware query templates so
pants receive waist/pant-hem/pocket/zipper queries instead of neckline or
shoulder queries. This reduces skipped records while keeping annotation small.

Start the browser annotator, then drag a bbox for each image-query pair. The
tool writes pixel-coordinate `target_bbox` values into a labeled JSONL file, so
you do not need to calculate coordinates by hand:

```bash
PYTHONPATH=src python scripts/data/annotate_local_region_bboxes.py \
  --manifest /root/autodl-tmp/outputs/local_region_manual_eval_manifest.jsonl \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_labeled.jsonl \
  --host 0.0.0.0 \
  --port 7860
```

Do not use DeepFashion2 landmarks while labeling. Then evaluate the full
pipeline against the manual benchmark:

```bash
PYTHONPATH=src python scripts/eval/evaluate_local_region_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic.json
```

Treat pseudo-label metrics as development diagnostics. The manual bbox
benchmark is the independent check for whether weak-supervised improvements
match real language-guided local-region localization. The initial 55-record
manual benchmark favored the pure heuristic online baseline over the hem-gated
candidate-listwise hybrid (`0.2544` vs `0.2324` average bbox IoU), so the
default online policy is heuristic-only. Keep learned rankers as experimental
branches until they improve this manual benchmark.

Merge multiple manual labeling rounds into one combined benchmark:

```bash
PYTHONPATH=src python scripts/data/merge_local_region_manual_eval_labels.py \
  --inputs \
    /root/autodl-tmp/outputs/local_region_manual_eval_labeled.jsonl \
    /root/autodl-tmp/outputs/local_region_manual_eval_labeled_class_aware.jsonl \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl
```

Then evaluate the combined benchmark with the heuristic default:

```bash
PYTHONPATH=src python scripts/eval/evaluate_local_region_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined.json
```

Combined manual benchmark result so far: `122` labeled records, average bbox IoU
`0.2812`, Hit@0.3 `0.4344`, Hit@0.5 `0.2623`. Shoulder, neckline, and hem are
the strongest regions; cuff, pocket, and waist are the main failure areas.
Export low-IoU examples for review:

```bash
PYTHONPATH=src python scripts/eval/export_local_region_manual_failures.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined.json \
  --output-dir /root/autodl-tmp/outputs/local_region_manual_failures_combined \
  --iou-threshold 0.1 \
  --regions cuff pocket waist \
  --max-cases 80
```

The export directory contains per-case images, `failure_summary.json`, and
`failure_review.html` for a grouped browser review page.

Before using a manual benchmark to select another model, audit its hard cases.
The audit manifest keeps the prior bbox visible, but resets its status so each
case must be explicitly confirmed, adjusted, or skipped. Use garment/wearer
left/right for side queries; skip a record if the named garment or its queried
part is absent, occluded, or ambiguous among multiple garments.

```bash
PYTHONPATH=src python scripts/data/build_local_region_manual_label_audit_manifest.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_four_expert_hybrid_fallback.json \
  --regions cuff pocket zipper waist \
  --iou-threshold 0.3 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_hard_region_audit.jsonl
```

Review the generated file with the existing annotator. Then merge it after the
original labels with `--skip-removes-existing`, so reviewed skips remove invalid
old labels instead of silently retaining them.

Failure review on the 34 exported cases showed three concrete policy issues:
side-specific cuff/pocket queries should follow garment/wearer left-right
convention instead of raw image left-right, cuff candidates should cover the
sleeve end rather than the whole side sleeve strip, and waist/pocket candidates
need category-aware upper-band geometry. Re-run the combined manual benchmark
after any policy refinement before changing the learned ranker.

After the first heuristic refinement, the same 122-record manual benchmark
improved to average bbox IoU `0.3064`, Hit@0.3 `0.4754`, and Hit@0.5 `0.2787`.
The targeted failure regions also improved: cuff `0.0190 -> 0.0592`, pocket
`0.0000 -> 0.1337`, and waist `0.0961 -> 0.2306`. Cuff remains the main
bottleneck and should be reviewed again before introducing more training.
The next cuff-only refinement emits both upper-sleeve and lower-terminal cuff
candidates, because the remaining manual failures mix short-sleeve/armhole
cases with long-sleeve terminal cases.

The cuff-variant policy improved the 122-record benchmark again to average bbox
IoU `0.3123`, Hit@0.3 `0.4836`, and Hit@0.5 `0.2705`; cuff improved from
`0.0592` to `0.0904`. This confirms the visual diagnosis, but cuff remains a
low-confidence region where pure geometry is near its limit.

### 3.1.2 Next Experiments

The next implementation direction should return to the PRD's pretrained
visual-text matching route instead of expanding pseudo-label ranker training:

1. Add an offline pretrained grounding evaluator.
   - Candidate models: GroundingDINO, OWL-ViT/OWL-V2, or Chinese-CLIP/CLIP crop
     reranking with SAM/3.1.1 masks as candidate regions.
   - If the model is English-centric, map Chinese query words to English
     prompts, e.g. `领口 -> neckline`, `袖口 -> cuff`, `口袋 -> pocket`,
     `拉链 -> zipper`, `下摆 -> hem`.
   - Evaluate only against
     `/root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl`.
   - First AutoDL command:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_pretrained_grounding_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --backend owlvit \
  --model-name google/owlvit-base-patch32 \
  --prompt-mode english \
  --device cuda \
  --score-threshold 0.05 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_owlvit.json
```

   - OWL-ViT base result on the 122-record manual benchmark was very weak:
     average bbox IoU `0.0305`, Hit@0.3 `0.0410`, Hit@0.5 `0.0000`, with
     `101/122` records returning no detection. Treat this as a negative
     generic open-vocabulary detector baseline, not as the final pretrained
     grounding route.
   - Next model to test:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_pretrained_grounding_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --backend auto \
  --model-name IDEA-Research/grounding-dino-tiny \
  --prompt-mode english \
  --device cuda \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_grounding_dino_tiny.json
```

   - GroundingDINO tiny result on the 122-record manual benchmark: average bbox
     IoU `0.2225`, Hit@0.3 `0.2295`, Hit@0.5 `0.1639`. It is still below the
     heuristic control overall, but it is much stronger on visual semantic
     regions: pattern `0.8262`, zipper `0.8233`, neckline `0.3843`. It is weak
     on geometry/structural regions such as hem, shoulder, cuff, and pocket.
   - Compare heuristic and GroundingDINO by region:

```bash
PYTHONPATH=src python scripts/eval/compare_local_region_manual_evals.py \
  --eval-json \
    /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_cuff_variants.json \
    /root/autodl-tmp/outputs/local_region_manual_eval_grounding_dino_tiny.json \
  --names heuristic grounding_dino_tiny \
  --default-eval heuristic \
  --region-policy pattern=grounding_dino_tiny zipper=grounding_dino_tiny \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_vs_grounding_dino.json
```

   - The fixed semantic-region hybrid (`pattern/zipper -> GroundingDINO`,
     others -> heuristic) reaches average bbox IoU `0.3465`, Hit@0.3 `0.5246`,
     Hit@0.5 `0.3197` on the 122-record benchmark. This is better than the
     heuristic-only control, but it is still a small benchmark and should be
     validated on a larger manual split before changing the default online path.
   - Next validation split: generate a targeted semantic/detail manifest that
     skips already labeled records and balances `pattern`, `zipper`, and
     `pocket`:

```bash
PYTHONPATH=src python scripts/data/build_local_region_manual_eval_manifest.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/validation/annos \
  --max-images 300 \
  --max-records 150 \
  --shuffle \
  --target-regions pattern zipper pocket \
  --balance-target-regions \
  --exclude-existing /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_manifest_semantic_150.jsonl
```

   - Targeted semantic split result: `49` labeled records and `101` skipped
     records. GroundingDINO beats heuristic on this split overall (`0.2133` vs
     `0.1296` avg IoU), especially on `pattern` (`0.5591` vs `0.3046`) and
     `pocket` (`0.1162` vs `0.0096`). `zipper` is not stable here, where
     heuristic is slightly better (`0.1637` vs `0.1334`). The fixed policy for
     this split is therefore `pattern/pocket -> GroundingDINO`, all other
     regions -> heuristic, reaching avg IoU `0.2250`.
   - Final validation should merge the original 122 labels and the new 49
     semantic labels, then re-run heuristic, GroundingDINO, and the fixed
     `pattern/pocket` hybrid on the combined manual benchmark.
   - Merged 171-record validation result: heuristic-only avg IoU `0.2599`,
     GroundingDINO-only `0.2199`, and fixed `pattern/pocket` hybrid `0.3060`.
     Hit@0.3 improves from `0.3918` to `0.4503`; Hit@0.5 improves from
     `0.2047` to `0.2749`. The fixed policy is effectively equal to the
     per-region oracle (`0.3060` avg IoU), so the current PRD-aligned direction
     is a gated hybrid rather than a full detector replacement.
   - The explicit experimental evaluator is
     `scripts/eval/evaluate_gated_hybrid_manual_labels.py`. It routes
     `pattern/pocket` to GroundingDINO and all other regions to the current
     heuristic path without changing default online inference. On the merged
     171-record benchmark, this executable gated path matches the fixed
     comparison result exactly: avg IoU `0.3060`, Hit@0.3 `0.4503`, Hit@0.5
     `0.2749`, with `41` grounding-routed records and `130` heuristic-routed
     records.
   - The matching single-image experimental script is
     `scripts/inference/predict_gated_hybrid_local_region.py`. Keep
     `scripts/inference/predict_local_region.py` as the default heuristic-only
     online path; use the gated script only when explicitly testing the
     `pattern/pocket -> GroundingDINO` policy.

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/inference/predict_gated_hybrid_local_region.py \
  /root/autodl-tmp/datasets/DeepFashion2/validation/image/000001.jpg \
  "这件衣服上的碎花图案" \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --grounding-regions pattern pocket \
  --grounding-backend auto \
  --grounding-model-name IDEA-Research/grounding-dino-tiny \
  --prompt-mode english \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_gated_single.json \
  --vis-output /root/autodl-tmp/outputs/local_region_gated_single.jpg
```

Run a small batch gated-hybrid demo with route counts, latency stats, records,
and visualizations:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --max-images 20 \
  --queries \
    "这件衣服的领口" \
    "衣服下方的下摆" \
    "这件衣服的肩部" \
    "这件衣服上的碎花图案" \
    "右侧的口袋" \
  --grounding-regions pattern pocket \
  --grounding-backend auto \
  --grounding-model-name IDEA-Research/grounding-dino-tiny \
  --prompt-mode english \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_gated_query_eval_20.json \
  --vis-dir /root/autodl-tmp/outputs/local_region_gated_query_vis \
  --vis-count 40
```

For visual review, use a per-record JSONL manifest when queries are not valid
for every image. Each line needs `image` and `query_text`; optional metadata is
copied into the output record:

```json
{"image": "/root/autodl-tmp/datasets/DeepFashion2/validation/image/000003.jpg", "query_text": "这件衣服上的碎花图案"}
{"image": "/root/autodl-tmp/datasets/DeepFashion2/validation/image/000010.jpg", "query_text": "右侧的口袋"}
```

Build the qualitative manifest from the completed gated manual evaluation
instead of choosing image ids by hand. The builder selects successful records
by manual IoU within each requested region and writes its selection provenance
and reference bbox into the JSONL. The visualization draws this manual reference
in green as `GT`; orange remains the predicted local region. This is a visual
sanity set, not an aggregate performance result.

```bash
PYTHONPATH=src python scripts/data/build_gated_hybrid_demo_manifest.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --target-regions pattern neckline hem shoulder \
  --per-region 2 \
  --min-iou 0.3 \
  --require-full-quota \
  --output /root/autodl-tmp/outputs/local_region_gated_demo_manifest.jsonl
```

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_queries.py \
  --manifest /root/autodl-tmp/outputs/local_region_gated_demo_manifest.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --grounding-regions pattern pocket \
  --grounding-backend auto \
  --grounding-model-name IDEA-Research/grounding-dino-tiny \
  --prompt-mode english \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_gated_demo_manifest_eval.json \
  --vis-dir /root/autodl-tmp/outputs/local_region_gated_demo_manifest_vis \
  --vis-count 80
```

2. Keep the online policy heuristic-only until a pretrained grounding baseline
   is wired behind an explicit experimental flag. The validated gated policy is:
   - `pattern` -> GroundingDINO
   - `pocket` -> GroundingDINO
   - all other regions -> heuristic

3. Use failure review to decide whether a model improves the hard cases:
   - cuff: needs real visual evidence for sleeve ends and armholes
   - pocket: needs side-aware small-object grounding
   - zipper/pattern/decoration: needs visual-text matching more than geometry

4. Only consider fine-tuning after the pretrained baseline is measured. If more
   training data is needed, add a small targeted calibration set instead of
   relabeling all of DeepFashion2.

Before changing the fixed `pattern/pocket` gate, analyze whether low-confidence
GroundingDINO detections should fall back to the heuristic. This command uses
completed JSON outputs only, splits by image into calibration and holdout sets,
and does not rerun either model. It is exploratory: only a holdout improvement
should justify a new online policy experiment.

```bash
PYTHONPATH=src python scripts/eval/analyze_gated_hybrid_confidence.py \
  --gated-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --heuristic-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --grounding-regions pattern pocket \
  --thresholds 0.0 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 \
  --holdout-fraction 0.3 \
  --output /root/autodl-tmp/outputs/local_region_gated_confidence_analysis.json
```

Compare `holdout_results` at threshold `0.0` (the current fixed gate) with the
selected threshold's `semantic_summary`; do not treat calibration gain alone as
evidence for a policy change.

The current confidence split did not establish a stable fallback gain. Before
changing the fixed gate, run a no-training prompt ablation on the same semantic
manual records. The model is loaded once and each profile is evaluated fairly:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_grounding_prompt_profiles.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --model-name IDEA-Research/grounding-dino-tiny \
  --backend auto \
  --prompt-mode english \
  --prompt-profiles ensemble precise fashion \
  --target-regions pattern pocket \
  --device cuda \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_grounding_prompt_profiles_pattern_pocket.json
```

`ensemble` is the validated current set of English synonyms; `precise` uses one
direct phrase; `fashion` adds an explicit clothing context. After selecting a
candidate only from this result, inspect its real improvements and regressions
against heuristic-only output:

```bash
PYTHONPATH=src python scripts/eval/export_gated_hybrid_policy_deltas.py \
  --baseline-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --candidate-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --regions pattern pocket \
  --candidate-routes grounding \
  --min-abs-delta 0.05 \
  --output-dir /root/autodl-tmp/outputs/local_region_gated_pattern_pocket_deltas
```

The paired `policy_delta_review.html` shows manual GT in green, heuristic in
red, and gated grounding in blue. It is offline analysis only; any prompt or
gate revision still needs a new full 171-record manual evaluation before it can
affect the default heuristic-only online path.

The paired review also exposes occasional background-object detections. An
explicit manual-evaluation experiment can reject GroundingDINO boxes that do
not overlap the frozen 3.1.1 selected garment mask, then fall back to heuristic
when no valid grounding detection remains:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --grounding-regions pattern pocket \
  --grounding-backend auto \
  --grounding-model-name IDEA-Research/grounding-dino-tiny \
  --prompt-profile ensemble \
  --constrain-grounding-to-garment \
  --grounding-min-mask-coverage 0.2 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_garment_constrained.json
```

This gate is experimental; retain it only if the complete manual benchmark
improves, and keep the default online path unchanged because it adds mask
inference for semantic queries.

On the 171-record benchmark, the garment constraint reduced Hit@0.3 from
`0.4503` to `0.4386`, so it is not adopted. Measure the theoretical per-record
best-of-two upper bound before investing in a router:

```bash
PYTHONPATH=src python scripts/eval/analyze_local_region_routing_oracle.py \
  --baseline-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --candidate-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --output /root/autodl-tmp/outputs/local_region_routing_oracle_heuristic_vs_gated.json
```

The result is an analysis-only ceiling: if its Hit@0.3 stays below 60%, routing
these two experts cannot meet the weekly target and one of the experts needs a
new capability.

Observed routing-oracle result on the 171 manually labeled records: best-of-two
reaches only Hit@0.3 `0.4561` (heuristic selected for 148 records, gated
GroundingDINO for 23). Therefore, do not spend another iteration on routing or
threshold tuning. Evaluate a new visual-text expert directly on the same manual
benchmark instead.

The next offline PRD-aligned baseline is frozen Chinese-CLIP crop reranking.
It uses the original Chinese query and candidate crops generated inside the
frozen 3.1.1 garment instance; it does not use landmarks, pseudo labels, or
training. The small region-prior sweep is diagnostic only, not an online
policy change:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_chinese_clip_manual_local_regions.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --device cuda \
  --region-prior-weights 0.0,0.05,0.1,0.2 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_chinese_clip_candidates.json
```

Compare each run with the heuristic and gated 171-record results. Keep the
heuristic-only online default unless a Chinese-CLIP configuration improves the
full manual benchmark and produces credible gains on cuff, waist, pocket, or
zipper after visual review.

Observed result: the best Chinese-CLIP settings (`0.1` and `0.2`) reached only
Hit@0.3 `0.3860`, below heuristic-only (`0.3918`) and far below the gated
GroundingDINO policy (`0.4503`). The visual score did not localize cuff, pocket,
or zipper reliably; the prior mainly restored rule-derived candidate names.
Do not integrate Chinese-CLIP crop reranking into the online policy. The next
pretrained comparison is GroundingDINO-base, evaluated offline before deciding
which hard regions, if any, it should replace:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_pretrained_grounding_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --model-name IDEA-Research/grounding-dino-base \
  --backend auto \
  --prompt-mode english \
  --prompt-profile ensemble \
  --device cuda \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_grounding_dino_base.json
```

Observed GroundingDINO-base result: it improves pocket Hit@0.3 to `0.2083`
(tiny/heuristic: `0.1250`) and cuff to `0.1304` (heuristic: `0.0870`), but it
is worse than tiny on pattern and worse than heuristic on structural regions.
The next reproducible policy test therefore uses tiny only for pattern, base
only for pocket/cuff, and heuristic for every other region:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --grounding-routes \
    pattern=IDEA-Research/grounding-dino-tiny \
    pocket=IDEA-Research/grounding-dino-base \
    cuff=IDEA-Research/grounding-dino-base \
  --grounding-backend auto \
  --prompt-mode english \
  --prompt-profile ensemble \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_multi_expert_pattern_tiny_pocket_cuff_base.json
```

This is still a same-benchmark exploratory policy: it must be evaluated as a
real pipeline run and visually reviewed before being treated as evidence. It
is not expected to reach 60% Hit@0.3 by itself, because zipper and most cuff
cases remain unresolved.

Observed pipeline result: the fixed multi-expert policy reproduces the expected
gain, reaching average manual IoU `0.3082`, Hit@0.3 `0.4678`, and Hit@0.5
`0.2924`. It is the current best experimental result, but still needs 23 more
Hit@0.3 successes to reach the 60% weekly target. Do not tune this route
further before testing a different pretrained grounding family.

The next diagnostic is OWLv2-large on the 79 hard cuff/pocket/zipper/waist
records. Run all three prompt profiles with one model load, then compare their
per-region results before launching a complete 171-record evaluation:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_grounding_prompt_profiles.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --model-name google/owlv2-large-patch14-ensemble \
  --backend owlv2 \
  --prompt-mode english \
  --prompt-profiles ensemble precise fashion \
  --target-regions cuff pocket zipper waist \
  --device cuda \
  --score-threshold 0.05 \
  --output /root/autodl-tmp/outputs/local_region_owlv2_large_hard_region_profiles.json
```

This is a diagnostic comparison only. A profile must improve a hard region
over the current multi-expert policy before it is evaluated on all 171 records
and considered for routing.

Observed OWLv2 diagnostic result: `precise` improves cuff Hit@0.3 to `0.2174`
(current base route: `0.1304`), and `ensemble` improves waist to `0.5000`
(heuristic: `0.3333`). Pocket only ties base at `0.2083`; zipper remains lower
than heuristic. Verify the following fixed four-expert policy on all 171
records. Per-region prompt and threshold overrides preserve each model's
validated setting:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --grounding-routes \
    pattern=IDEA-Research/grounding-dino-tiny \
    pocket=IDEA-Research/grounding-dino-base \
    cuff=google/owlv2-large-patch14-ensemble \
    waist=google/owlv2-large-patch14-ensemble \
  --grounding-route-profiles cuff=precise waist=ensemble \
  --grounding-route-thresholds cuff=0.05 waist=0.05 \
  --grounding-backend auto \
  --prompt-mode english \
  --prompt-profile ensemble \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_four_expert_hybrid.json
```

The pipeline loads each `(model, threshold)` pair once. Zipper deliberately
stays on heuristic. The expected same-benchmark gain is roughly four Hit@0.3
successes over the three-expert run; only the real 171-record output can
confirm it.

After the hard-region label audit, the decision benchmark contains 161 valid
records. Heuristic-only reaches Hit@0.3 `0.4099` (66/161), while the four-expert
policy reaches `0.5217` (84/161). Offline Top-5 analysis found that enforcing
garment/wearer-side consistency improves cuff from 5/18 to 8/18 Hit@0.3. The
same rule does not improve pocket Hit@0.3 and reduces pocket Hit@0.5, so it is
enabled only for cuff:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_audited.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --grounding-routes \
    pattern=IDEA-Research/grounding-dino-tiny \
    pocket=IDEA-Research/grounding-dino-base \
    cuff=google/owlv2-large-patch14-ensemble \
    waist=google/owlv2-large-patch14-ensemble \
  --grounding-route-profiles cuff=precise waist=ensemble \
  --grounding-route-thresholds cuff=0.05 waist=0.05 \
  --grounding-backend auto \
  --prompt-mode english \
  --prompt-profile ensemble \
  --score-threshold 0.15 \
  --fallback-on-no-detection \
  --wearer-side-regions cuff \
  --wearer-side-min-score-ratio 0.5 \
  --record-heuristic-candidates-for-grounding \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_four_expert_side_cuff_audited.json
```

Before tuning another selector, measure whether the saved Top-5 grounding boxes
or the diagnostic heuristic candidate can recover the remaining failures. This
oracle is diagnostic only and never uses manual boxes in online inference:

```bash
PYTHONPATH=src python scripts/eval/analyze_grounding_candidate_oracle.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_four_expert_side_cuff_audited.json \
  --regions cuff pocket pattern waist \
  --hit-threshold 0.3 \
  --output /root/autodl-tmp/outputs/local_region_grounding_candidate_oracle_audited.json
```

The 60% target is 97/161 hits. The side-aware cuff result is expected to reach
87/161 if the online rerun reproduces the offline analysis, leaving 10 hits.
Use `recoverable_failures` to decide whether the next step is candidate
selection or new candidate generation.

The expanded grounding-plus-heuristic oracle remains at 98/161 Hit@0.3;
heuristic candidates improve IoU and Hit@0.5 but add no new Hit@0.3 success.
Generate diagnostic zipper candidates with the already loaded
GroundingDINO-base model while preserving heuristic as the selected zipper
route:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_manual_labels.py \
  ...same audited four-expert arguments... \
  --diagnostic-grounding-routes zipper=IDEA-Research/grounding-dino-base \
  --record-heuristic-candidates-for-grounding \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_zipper_candidates_audited.json
```

The diagnostic route saves Top-K boxes but cannot change the 87/161 selected
policy result. Include `zipper` in the candidate-oracle regions for the next
ceiling measurement.

The zipper candidate raises the oracle to 101/161 Hit@0.3 (`0.6273`), adding
three recoverable zipper failures. To widen the four-hit margin without loading
another model, cross the two already loaded GroundingDINO experts: use base as
the diagnostic model for pattern/cuff/waist and tiny for pocket, while retaining
base for zipper. Selected online routes remain unchanged.

The completed cross-model oracle reaches 107/161 Hit@0.3 (`0.6646`), providing
a ten-hit margin above the 97/161 target. Candidate-selector development must
not train and report on the same manual records. Run the image-grouped 5-fold
selector evaluation; every reported prediction is produced by a model trained
without any label from that image:

```bash
PYTHONPATH=src python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_cross_model_candidates_audited.json \
  --regions cuff pocket pattern waist zipper \
  --num-folds 5 \
  --num-epochs 120 \
  --device cpu \
  --output /root/autodl-tmp/outputs/local_region_candidate_selector_5fold_audited.json
```

Use `out_of_fold_summary.manual_hit_at["0.3"]` as the decision metric, not an
in-sample score or the oracle ceiling.

### Archived Weak-Supervision Experiments

These commands are kept for reproducibility, but they are no longer the main
3.1.2 plan after the manual benchmark and mentor review.

Build weak query-region records for a learned 3.1.2 ranker:

```bash
python scripts/data/build_deepfashion2_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/train/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/train/annos \
  --output /root/autodl-tmp/outputs/local_region_train_queries.jsonl
```

Train the lightweight learned ranker:

```bash
python scripts/train/train_local_region_ranker.py \
  --records /root/autodl-tmp/outputs/local_region_train_queries.jsonl \
  --output /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry.pt \
  --device cuda \
  --max-records 50000 \
  --val-records 2000 \
  --num-epochs 1
```

Use `--val-offset` to evaluate on a later JSONL slice during larger runs.

Export candidate-level records for the next vision-language local-region ranker:

```bash
python scripts/data/build_local_region_candidate_records.py \
  --records /root/autodl-tmp/outputs/local_region_train_queries.jsonl \
  --output /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --max-records 500000
```

Install and evaluate Chinese-CLIP candidate reranking on weak candidates:

```bash
pip install "transformers>=4.37.0" sentencepiece
HF_ENDPOINT=https://hf-mirror.com \
python scripts/eval/evaluate_chinese_clip_local_region_ranker.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --device cuda \
  --max-groups 2000 \
  --region-prior-weights 0,0.01,0.02,0.05,0.1,0.2 \
  --output /root/autodl-tmp/outputs/local_region_chinese_clip_eval_2k.json
```

Run candidate diagnostics:

```bash
python scripts/eval/evaluate_local_region_candidate_baselines.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --max-groups 2000 \
  --output /root/autodl-tmp/outputs/local_region_candidate_baselines_2k.json
```

Train a listwise candidate ranker from weak IoU labels:

```bash
python scripts/train/train_candidate_local_region_ranker.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --output /root/autodl-tmp/checkpoints/local_region_ranker/candidate_listwise_context_50k.pt \
  --device cuda \
  --max-groups 50000 \
  --val-groups 2000 \
  --loss soft \
  --softmax-temperature 0.08 \
  --metrics-output /root/autodl-tmp/outputs/local_region_candidate_listwise_context_50k_metrics.json \
  --num-epochs 1
```

Validate the saved candidate ranker on a later slice:

```bash
python scripts/train/train_candidate_local_region_ranker.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/local_region_ranker/candidate_listwise_context_50k.pt \
  --device cuda \
  --val-offset 50000 \
  --val-groups 5000 \
  --eval-only \
  --metrics-output /root/autodl-tmp/outputs/local_region_candidate_listwise_context_eval_offset50k.json
```

The context-feature candidate ranker is strong offline, but manual bbox
evaluation did not confirm an online gain. On the initial 55-record manual
benchmark, pure heuristic reached average bbox IoU `0.2544` while the hem-gated
candidate-listwise hybrid reached `0.2324`. Candidate-listwise checkpoints are
therefore disabled in online inference by default and should be treated as an
experimental weak-supervision branch, not the deployed 3.1.2 baseline.

Optionally compare an experimental learned ranker checkpoint against the
heuristic default:

```bash
python scripts/eval/evaluate_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --ranker-checkpoint /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry_500k.pt \
  --device cuda \
  --max-images 20 \
  --output /root/autodl-tmp/outputs/local_region_query_eval_learned.json
```
