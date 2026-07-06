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

2. Keep the online policy heuristic-only until a pretrained grounding baseline
   beats the 122-record manual benchmark:
   - current heuristic avg bbox IoU: `0.3123`
   - Hit@0.3: `0.4836`
   - Hit@0.5: `0.2705`

3. Use failure review to decide whether a model improves the hard cases:
   - cuff: needs real visual evidence for sleeve ends and armholes
   - pocket: needs side-aware small-object grounding
   - zipper/pattern/decoration: needs visual-text matching more than geometry

4. Only consider fine-tuning after the pretrained baseline is measured. If more
   training data is needed, add a small targeted calibration set instead of
   relabeling all of DeepFashion2.

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
