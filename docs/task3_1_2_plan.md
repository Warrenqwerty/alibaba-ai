## 3.1.2 Language-Guided Local Region Localization

This document starts the 3.1.2 module after the 3.1.1 instance segmentation
checkpoint reached the PRD target on full DeepFashion2 validation.

### PRD Scope

Task: language-guided local region localization.

Input:

- RGB fashion image.
- Natural-language query, such as "这件衣服的领口" or "袖口的设计".
- 3.1.1 instance segmentation output should be reused as visual grounding
  context whenever available.

Output:

- Target local-region mask.
- Target local-region bounding box.
- Region label and confidence score.

Supported regions:

- neckline / collar
- cuff
- hem
- pocket
- shoulder
- waist
- pattern / print
- decoration
- other clothing local regions described by natural language

Target metrics:

- Region localization accuracy >= 92%.
- Localization latency <= 30 ms.

### Current Starting Point

3.1.1 is frozen as the upstream visual foundation module.

Current best checkpoint:

```bash
/root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt
```

Full validation result:

- Mean best mask IoU: 0.8585.
- Recall@0.75: 0.8967.
- Inference thresholds: score >= 0.3, mask >= 0.4.

3.1.2 should not redo whole-garment segmentation from scratch. It should consume
the image plus 3.1.1 garment instances, then localize the requested part inside
the relevant garment instance.

### Revised Direction After Data Review

The original PRD points to a pretrained visual-text grounding route: extract
region features, compare them with text features, and return the region that
best matches the natural-language query. That means 3.1.2 should be framed as
open-vocabulary language-guided localization, not fixed part segmentation and
not a pure DeepFashion2 pseudo-label training task.

The current project data makes this distinction important:

DeepFashion2 provides:

- garment instance masks
- garment bounding boxes
- garment category labels
- landmarks / keypoints

This is useful for garment-level grounding and weak diagnostics, but it does
not provide query-level human annotations such as "左边的袖口" or "衣服上的拉链"
mapped to a bbox or mask. The current JSON category metadata names landmarks
only as numeric ids (`1` to `30`). It does not directly name keypoints as
"collar", "cuff", "hem", etc. Therefore DeepFashion2 local regions should be
treated as weak supervision or candidate-generation hints, not as perfect
semantic language-region labels.

The rule-based rectangles in the first demo are not suitable as training labels.
They are only sanity-check proposals and online fallback candidates. The
previous pseudo-label ranker experiments are useful engineering exploration, but
they cannot be the main evidence for PRD 3.1.2 because their labels come from
landmark pseudo-labels and rule fallback.

FashionAI-Attributes is still needed for the broader attribute pipeline in
3.1.3, and may help define part-related query words such as collar and sleeve.
However, FashionAI attributes are image/attribute labels rather than pixel-level
local-region masks, so it is not enough by itself for 3.1.2 mask supervision.

### Final Baseline Direction

Use a PRD-aligned open-vocabulary grounding baseline. The baseline has two
tracks: a safe heuristic online track and an offline pretrained grounding track.

1. Whole-garment grounding from 3.1.1:
   - Run the frozen instance segmentation model.
   - Select the relevant garment instance using category words in the query
     when possible. Example: "这条裙子的下摆" should prefer `skirt`.
   - If no garment class is mentioned, use the highest-confidence clothing
     instance or all candidate instances.

2. Open candidate generation inside the garment:
   - Generate multiple generic candidate masks, not only training-time fixed
     parts.
   - Current candidates include whole garment, upper/lower/left/right/center
     spatial regions, neckline, hem, shoulder, waist, left/right cuff,
     left/right pocket, zipper, button, decoration, and pattern-like
     full-garment candidates.
   - Landmarks and fixed part rules are helper candidate generators, not the
     final task definition.

3. Text-region matching:
   - Current online version: lightweight heuristic ranker that scores candidates from
     raw Chinese query text, spatial words, attribute words, relation words,
     and part words.
   - Target offline version: pretrained grounding / visual-text matching,
     following the PRD direction of "区域特征与文本特征相似度匹配".
     Candidate models include GroundingDINO, OWL-ViT/OWL-V2, Chinese-CLIP,
     CLIP with Chinese-to-English prompt mapping, or DINOv2 region features
     paired with a separate text encoder.
   - This should support open descriptions such as "左边的袖口", "碎花图案",
     "衣服上的拉链", and "外套里面的内搭" better than fixed-part
     classification.

This means the old fixed-region parser is no longer the main decision maker.
It is only a helper for candidate scoring.

The default online path remains heuristic-only until a pretrained grounding
model improves the manual benchmark. Candidate-listwise weak rankers are
disabled in online inference because manual evaluation did not confirm their
offline pseudo-label gains.

### Suggested Output Schema

```json
{
  "image": "000001.jpg",
  "query": {
    "text": "这件衣服的领口",
    "region": "neckline",
    "garment_hint": "top",
    "spatial_hints": [],
    "attribute_hints": [],
    "relation_hints": []
  },
  "selected_instance": {
    "label": "top",
    "score": 0.94,
    "box": [x1, y1, x2, y2]
  },
  "region": {
    "name": "neckline",
    "source": "rule_baseline",
    "confidence": 0.70,
    "box": [x1, y1, x2, y2],
    "mask": "optional serialized mask or saved mask path"
  },
  "latency_ms": 12.3
}
```

### Evaluation Plan

Because there is no ready-made query-level local-region ground truth,
evaluation should be staged and manual-first:

1. Manual bbox benchmark:
   - Use this as the main 3.1.2 decision metric.
   - Target size: 100-300 image-query pairs, covering neckline, hem, shoulder,
     cuff, pocket, zipper, pattern, decoration, and waist where applicable.
   - Label only `target_bbox` in xyxy image pixels, do not use landmarks, and
     do not use this file for training.
   - Use `--anno-dir` when building the manifest to enable class-aware query
     templates and avoid impossible pairs such as pants + neckline.
   - Current combined benchmark: 122 labeled records.
   - Current heuristic-only result after cuff-variant refinement: average bbox
     IoU 0.3123, Hit@0.3 0.4836, Hit@0.5 0.2705.
   - Strong regions: shoulder, neckline, hem.
   - Weak regions: cuff, pocket, waist, zipper/pattern when target is small or
     visually ambiguous.

2. Functional sanity test:
   - Given fixed images and queries, verify the pipeline returns non-empty masks
     within the selected garment instance.

3. Pretrained grounding evaluation:
   - Add an offline evaluator for GroundingDINO / OWL-ViT / Chinese-CLIP or
     CLIP-style crop reranking.
   - For English-centric grounding models, map Chinese query words to English
     prompts, e.g. `领口 -> neckline`, `袖口 -> cuff`, `口袋 -> pocket`,
     `拉链 -> zipper`, `下摆 -> hem`.
   - Compare every model against the same manual bbox benchmark before changing
     the online policy.
   - If a pretrained model beats the heuristic baseline on manual IoU and hard
     regions, integrate it as an optional backend; otherwise keep it as an
     offline experiment.

4. Weak automatic evaluation:
   - Use DeepFashion2 landmarks and garment masks to approximate regions such as
     neckline, hem, shoulder, waist.
   - Measure whether the returned region overlaps the weakly derived target.
   - Current 50-image AutoDL weak-label baseline:
     - Average weak IoU: 0.2884.
     - Hit@0.3: 0.3733.
     - Hit@0.5: 0.1867.
     - By region: hem 0.3753, neckline 0.2895, shoulder 0.2003.
   - The next heuristic iteration should focus on shoulder geometry first,
     because it is the weakest landmark-aligned region.
   - After shoulder/neckline geometry tuning:
     - Average weak IoU: 0.3217 (+0.0333).
     - Hit@0.3: 0.4333 (+0.0600).
     - Hit@0.5: 0.1933 (+0.0066).
     - By region: hem 0.3753, neckline 0.3138, shoulder 0.2759.
     - Diagnostics: weak labels are mostly landmark-derived (130/150), and
       average garment IoU is 0.8329, so the remaining gap is mainly local
       region localization rather than failed garment handoff.
   - 200-image stability check after tuning:
     - Average weak IoU: 0.2818.
     - Hit@0.3: 0.3933.
     - Hit@0.5: 0.1383.
     - By region: hem 0.2788, neckline 0.3000, shoulder 0.2665.
     - Diagnostics: weak labels are mostly landmark-derived (518/600), and
       average garment IoU is 0.7843.
     - This larger run confirms the heuristic baseline is stable but not strong
       enough; the next useful baseline should add learned text-region
       similarity instead of more geometry tuning.
   - Full train weak query export:
     - Annotations: 191,961.
     - JSONL records: 2,808,252.
     - Region counts: neckline 936,336; hem 935,478; shoulder 936,438.
     - Source counts: landmark pseudo labels 2,231,694; rule fallback 576,558.
   - Lightweight learned hash text-geometry ranker:
     - 50k-record smoke result: validation top-1 box IoU 0.3540.
     - 500k-record offset-validation result: validation top-1 box IoU 0.3560.
     - This is stable enough to integrate as an optional inference backend,
       while the next model upgrade should use image-region embeddings.
     - Because weak training currently covers neckline, hem, and shoulder only,
       production inference should use a hybrid backend: learned scorer for
       trained regions and heuristic fallback for open queries.
     - 20-image hybrid sanity result: 140/140 ok, diverse selected regions
       restored for cuff, pattern, zipper, and pocket, average local-region
       latency 16.93 ms.
     - 200-image hybrid weak-label result: average weak IoU 0.2759, below the
       tuned heuristic 0.2818 because shoulder dropped to 0.2477. Therefore the
       current learned scorer should be used only for neckline/hem, with
       heuristic fallback for shoulder until image-region features are added.
     - 200-image neckline/hem-only hybrid result: average weak IoU 0.2822,
       neckline 0.3013, hem 0.2788, shoulder 0.2665. This recovers the tuned
       heuristic baseline and gives only a very small gain, so the lightweight
       text-geometry scorer should be treated as a bridge rather than the final
       learned localizer.
   - Context-feature listwise candidate ranker:
     - 50k-query training result: validation top-1 candidate IoU 0.5113 on the
       first 2k groups, close to the candidate-set oracle 0.5704 and clearly
       above the target-region-name baseline 0.3589.
     - Later-slice evaluation result: average top-1 candidate IoU 0.4456 on 5k
       groups, with oracle 0.5193. This shows the ranker learns useful
       query/candidate/context signals instead of only memorizing the first
       validation slice.
     - Full 3.1.2 weak-label pipeline result without gating: average weak IoU
       0.2732 on 200 validation images, below the tuned heuristic baseline.
       Neckline and shoulder degrade after the candidate score is transferred
       back through the predicted instance mask.
     - Weak-label online attempt: gating the listwise context ranker to hem
       recovered the 200-image weak metric to average weak IoU 0.2818, Hit@0.3
       0.4050, Hit@0.5 0.1333; by region: hem 0.2789, neckline 0.3000,
       shoulder 0.2665.
   - Manual benchmark result: on the initial 55 manually labeled bbox
       records, pure heuristic outperformed the hem-gated candidate-listwise
       hybrid (average bbox IoU 0.2544 vs 0.2324; Hit@0.3 0.4000 vs 0.3455).
       Hem dropped from 0.3077 to 0.1982 when the listwise branch was used.
     - Current online policy: use the pure heuristic open-vocabulary pipeline
       by default. Keep candidate-listwise rankers as weak-supervised
       experimental branches until they improve the manual benchmark.
     - Next annotation step: expand to about 120-150 labeled records using
       class-aware query sampling so pants/skirt/dress/top images receive
       compatible local-region queries and the skip rate is lower.
     - Combined manual benchmark after two annotation rounds: 122 labeled
       records, average bbox IoU 0.2812, Hit@0.3 0.4344, Hit@0.5 0.2623.
       Shoulder, neckline, and hem are relatively stable; cuff, pocket, and
       waist are the main failure regions.
     - Use `scripts/eval/export_local_region_manual_failures.py` to export
       low-IoU visual cases plus `failure_review.html` for qualitative review
       before changing model direction.
     - First failure review found that many cuff/pocket zero-IoU cases were
       caused by image-left/image-right vs garment-left/garment-right mismatch,
       and that cuff/waist/pocket windows were too coarse. The online heuristic
       policy was therefore refined before returning to any learned ranker work.
     - Refined heuristic manual benchmark: 122 labeled records, average bbox
       IoU 0.3064, Hit@0.3 0.4754, Hit@0.5 0.2787. Targeted failure regions
       improved, especially waist (0.0961 -> 0.2306) and pocket (0.0000 ->
       0.1337), while cuff remains weak (0.0190 -> 0.0592).
     - Second cuff review separated short-sleeve/armhole failures from
       long-sleeve terminal failures. The online policy now emits upper-sleeve
       and lower-terminal cuff candidates so the heuristic can choose between
       them without adding learned-ranker dependence.
     - Cuff-variant manual benchmark: 122 labeled records, average bbox IoU
       0.3123, Hit@0.3 0.4836, Hit@0.5 0.2705. Cuff improved again (0.0592 ->
       0.0904), but remains a low-confidence region where geometry-only rules
       are likely close to their practical ceiling.
   - Metric caveat after review:
     - The weak-label train/eval loop uses landmark pseudo-labels plus rule
       fallback, so it can be biased toward the pseudo-label geometry instead
       of the real region a user would mark.
     - Candidate-level experiments are optimistic because the records are
       generated from DeepFashion2 GT masks and landmark-derived weak targets.
       Full pipeline evaluation uses predicted 3.1.1 masks and no landmark
       access, so it is the more realistic online metric.
     - Therefore pseudo-label metrics should be treated as development
       diagnostics, not final PRD accuracy.
5. Latency evaluation:
   - Measure 3.1.2 region localization time excluding or including 3.1.1,
     depending on final system definition.

### Initial Engineering Tasks

1. Add query parser:
   - input: natural-language query
   - output: canonical region name and optional garment class hint

2. Add local region proposal module:
   - input: image size, garment mask, garment bbox, canonical region
   - output: region mask and bbox

3. Add inference script:
   - input: image, query, 3.1.1 checkpoint
   - output: JSON + visualization

4. Add validation / sanity script:
   - fixed query templates
   - representative validation images
   - visualization grid for quick manual review

5. Add a pretrained grounding baseline evaluator:
   - first target: run offline against the 122-record manual benchmark
   - output: selected bbox, confidence, IoU, and visualization
   - gate: do not change online inference until it beats heuristic-only

6. Keep weak-label ranker scripts as archived experiments:
   - useful for reproducibility and ablation
   - not the default route for PRD 3.1.2

### Current Baseline Usage

The first rule-based 3.1.2 inference script is:

```bash
python scripts/inference/predict_local_region.py \
  /path/to/image.jpg \
  "这件衣服的领口" \
  --model-config configs/model/instance_segmentation_deepfashion2.yaml \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output outputs/local_region_sample.json \
  --vis-output outputs/local_region_sample.jpg
```

The visualization uses blue for the selected whole garment instance and orange
for the localized query region.

### Landmark Inspection Utility

Before training a learned 3.1.2 model, inspect DeepFashion2 landmarks and map
numeric landmark ids to semantic parts. Use:

```bash
python scripts/data/visualize_deepfashion2_landmarks.py \
  --image /root/autodl-tmp/datasets/DeepFashion2/validation/image/000003.jpg \
  --annotation /root/autodl-tmp/datasets/DeepFashion2/validation/annos/000003.json \
  --output outputs/deepfashion2_landmarks_000003.jpg
```

Visible landmarks are drawn in green, occluded landmarks in orange, and each
point is labeled with its DeepFashion2 landmark index.

### Risks

- DeepFashion2 landmarks are not directly named by semantic region in the local
  metadata, so weak labels may be noisy.
- The current heuristic ranker is only a prototype, but it is the safest online
  baseline after the first manual bbox benchmark. True PRD performance still
  requires learned visual-text similarity using DINOv2/CLIP-style features or a
  stronger grounding model.
- Rule-based candidates can cover neckline/hem/waist, but pocket, pattern,
  decoration, zipper, button, and relation queries need visual feature matching.
- PRD latency target is tight; a heavy grounding model may exceed 30 ms unless
  cached or optimized.
- Query ambiguity is common. Example: "这件衣服的设计" is too broad and should
  ask for clarification or return the full garment region.

### Recommended Immediate Next Step

Implement the pretrained grounding benchmark while keeping the current
heuristic baseline fixed:

- keep selected-garment instance handoff from 3.1.1
- keep the current heuristic-only policy as the control baseline
- use `scripts/eval/evaluate_pretrained_grounding_manual_labels.py` to evaluate
  OWL-ViT/OWL-V2-style pretrained grounding models on the manual JSONL
- support Chinese-to-English prompt templates for English-centric models
- visualize selected region plus top candidate scores
- compare the pretrained model against the 122-record manual benchmark before
  enabling it in the online path

First offline baseline to run on AutoDL:

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

OWL-ViT base result on the 122-record manual benchmark: average bbox IoU
`0.0305`, Hit@0.3 `0.0410`, Hit@0.5 `0.0000`, with `101/122` records returning
`no_detection`. This confirms that a generic OWL-ViT detector is not enough for
fine-grained fashion local regions. The next offline baseline should be
GroundingDINO tiny:

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

GroundingDINO tiny result on the same benchmark: average bbox IoU `0.2225`,
Hit@0.3 `0.2295`, Hit@0.5 `0.1639`. This still does not replace the heuristic
control overall, but it changes the direction: visual grounding is clearly
useful for semantic appearance regions, especially pattern (`0.8262`) and
zipper (`0.8233`), while structural geometry regions still favor the heuristic.

Use the comparison utility before changing online behavior:

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

The fixed semantic-region hybrid reaches average bbox IoU `0.3465`, Hit@0.3
`0.5246`, and Hit@0.5 `0.3197`, compared with heuristic-only `0.3123` /
`0.4836` / `0.2705`. This supports a gated design rather than a full detector
replacement: use GroundingDINO for appearance semantics (`pattern`, `zipper`) and
keep heuristic geometry for neckline, hem, shoulder, cuff, pocket, and waist.

Before wiring this into online inference, build a larger targeted manual split
for semantic/detail regions:

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

The targeted semantic split produced `49` labeled records. GroundingDINO beats
heuristic on this split overall (`0.2133` vs `0.1296` average bbox IoU), with a
clear gain on pattern (`0.5591` vs `0.3046`) and pocket (`0.1162` vs `0.0096`).
Zipper is not stable on this split; heuristic is slightly better (`0.1637` vs
`0.1334`). The currently supported fixed policy is therefore:

- `pattern` -> GroundingDINO
- `pocket` -> GroundingDINO
- all other regions -> heuristic

Next validation step: merge the original 122-record benchmark with the 49
semantic labels and evaluate this fixed `pattern/pocket` policy on the combined
manual benchmark before changing online inference.

Merged 171-record validation result:

- heuristic-only: avg bbox IoU `0.2599`, Hit@0.3 `0.3918`, Hit@0.5 `0.2047`
- GroundingDINO-only: avg bbox IoU `0.2199`, Hit@0.3 `0.2456`, Hit@0.5 `0.1813`
- fixed `pattern/pocket` hybrid: avg bbox IoU `0.3060`, Hit@0.3 `0.4503`,
  Hit@0.5 `0.2749`

This validates a gated hybrid direction:

- route `pattern` and `pocket` queries to GroundingDINO
- keep heuristic geometry for neckline, hem, shoulder, cuff, waist, and zipper
- expose this as an explicit experimental path first, not as a silent default
  change, because GroundingDINO requires the original image and adds substantial
  inference cost

The experimental path is implemented as
`scripts/eval/evaluate_gated_hybrid_manual_labels.py`. It should be used to
validate policy behavior and latency before adding an inference-facing flag.
It has been validated on the merged 171-record benchmark and matches the fixed
hybrid comparison: avg bbox IoU `0.3060`, Hit@0.3 `0.4503`, Hit@0.5 `0.2749`,
with `41` GroundingDINO-routed records and `130` heuristic-routed records.

The matching inference-facing experiment is
`scripts/inference/predict_gated_hybrid_local_region.py`. Use it only when
explicitly testing the gated policy. The default
`scripts/inference/predict_local_region.py` entry remains heuristic-only.

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

For a deliverable-style 3.1.2 demo, run the batch gated-hybrid query evaluator:

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

This produces per-query records, route counts, latency by route, and
visualizations. Use it as the current inference demo while keeping the default
online path unchanged.

For the final visual demo, use `--manifest` with valid image-query pairs instead
of applying the same query list to every image. This is important for optional
local regions such as pockets, zippers, and patterns, which may not exist or may
not be visible in every DeepFashion2 validation image.

```json
{"image": "/root/autodl-tmp/datasets/DeepFashion2/validation/image/000003.jpg", "query_text": "这件衣服上的碎花图案"}
{"image": "/root/autodl-tmp/datasets/DeepFashion2/validation/image/000010.jpg", "query_text": "右侧的口袋"}
```

Do not choose final demo examples by browsing image ids. Build the manifest
from the completed gated manual evaluation: it selects successful records by
manual IoU in each requested region and includes selection provenance in every
JSONL line. This is a reproducible qualitative check, not a substitute for the
`171`-record manual benchmark. The generated visualization overlays the manual
reference bbox in green as `GT`; orange is the predicted local region.

```bash
PYTHONPATH=src python scripts/data/build_gated_hybrid_demo_manifest.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --target-regions pattern neckline hem shoulder \
  --per-region 2 \
  --min-iou 0.3 \
  --require-full-quota \
  --output /root/autodl-tmp/outputs/local_region_gated_demo_manifest.jsonl
```

The next controlled improvement is confidence fallback for semantic grounding:
if GroundingDINO is not confident for `pattern` or `pocket`, compare its result
with the existing heuristic result instead of assuming the detector should
always win. Use an image-held-out calibration analysis, not the visual demo, to
choose a candidate threshold:

```bash
PYTHONPATH=src python scripts/eval/analyze_gated_hybrid_confidence.py \
  --gated-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --heuristic-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --grounding-regions pattern pocket \
  --thresholds 0.0 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 \
  --holdout-fraction 0.3 \
  --output /root/autodl-tmp/outputs/local_region_gated_confidence_analysis.json
```

This is offline analysis only. Integrate a threshold into the experimental
inference path only after it improves the image-held-out semantic result and a
fresh full manual evaluation confirms the gain. Compare the candidate threshold
with `holdout_results` at `0.0`, which represents the current fixed gate.

The completed threshold analysis selected `0.0`; the small holdout fluctuation
at higher thresholds is not enough to change the policy. Do not continue
threshold tuning. The next controlled experiment is prompt wording, evaluated
without training and without reloading the grounding model for each variant:

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

`ensemble` preserves the existing validated English synonym set. `precise`
uses one region phrase, which may reduce prompt competition, and `fashion`
adds clothing context for small garments parts. Select a profile only if it
improves the semantic manual metrics and then rerun the full gated evaluator.

Before any implementation change, export paired material deltas from the
current policy for visual review:

```bash
PYTHONPATH=src python scripts/eval/export_gated_hybrid_policy_deltas.py \
  --baseline-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --candidate-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --regions pattern pocket \
  --candidate-routes grounding \
  --min-abs-delta 0.05 \
  --output-dir /root/autodl-tmp/outputs/local_region_gated_pattern_pocket_deltas
```

The `policy_delta_review.html` output overlays manual GT (green), heuristic
prediction (red), and gated prediction (blue). Use it to distinguish prompt
wording failures from false detections and left/right ambiguity. The default
online path remains heuristic-only.

Paired visual review found a concrete, testable failure mode: several gated
regressions select a bag or a background pattern outside the relevant garment.
The next experiment constrains GroundingDINO detections to the frozen 3.1.1
selected garment mask. A detection must have at least `0.2` of its box area in
the mask; otherwise the evaluator uses the existing heuristic result. This
does not solve every within-garment ambiguity such as a whole trouser leg, so
the full manual benchmark remains the decision criterion.

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --grounding-regions pattern pocket \
  --grounding-backend auto \
  --grounding-model-name IDEA-Research/grounding-dino-tiny \
  --prompt-mode english \
  --prompt-profile ensemble \
  --score-threshold 0.15 \
  --constrain-grounding-to-garment \
  --grounding-min-mask-coverage 0.2 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_garment_constrained.json
```

Report the complete 171-record `manual_hit_at["0.3"]`, not only selected
examples. The constraint is experimental and adds segmentation latency to the
semantic route; do not enable it in the default online path unless the manual
gain is clear.

Observed result: the garment-mask constraint reduced full-benchmark Hit@0.3
from `0.4503` to `0.4386` and reduced pocket Hit@0.3 from `0.1250` to
`0.0417`. Six grounding cases fell back to heuristic, but the selected garment
mask is not reliable enough for this filter. Do not adopt this constraint.

Before adding another router, measure the best possible per-record choice
between the original heuristic and original gated outputs. This is an oracle
upper bound, not a reportable model metric: it tells whether a learned router
could plausibly reach the weekly Hit@0.3 target using these two experts.

```bash
PYTHONPATH=src python scripts/eval/analyze_local_region_routing_oracle.py \
  --baseline-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --candidate-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --output /root/autodl-tmp/outputs/local_region_routing_oracle_heuristic_vs_gated.json
```

If this oracle is below `0.60` Hit@0.3, routing alone cannot meet the target;
the next iteration must improve an expert, especially cuff/waist/pocket. If it
reaches or exceeds `0.60`, inspect the per-region `source_counts` and design a
small, independently validated router rather than using the oracle directly.

The completed oracle reached only Hit@0.3 `0.4561` on 171 manually labeled
records (heuristic selected for 148 records, gated GroundingDINO for 23).
This is an upper bound, so routing the current two experts cannot reach the
weekly 60% target. Stop routing and threshold tuning; the next experiment must
test a genuinely new visual-text expert.

Run frozen Chinese-CLIP crop reranking directly against the same manual
benchmark. It uses the Chinese query as text input and ranks open-vocabulary
crop candidates generated inside the frozen 3.1.1 selected garment instance.
It does not train, use landmarks, or use pseudo labels as its evaluation
target:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_chinese_clip_manual_local_regions.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --device cuda \
  --region-prior-weights 0.0,0.05,0.1,0.2 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_chinese_clip_candidates.json
```

The small prior-weight sweep tests whether visual similarity adds value beyond
the parser, but it is not an online policy. Compare every `runs` result with
the heuristic and gated 171-record metrics. Only a clear full-benchmark gain,
including review of cuff/waist/pocket/zipper cases, can justify integration.

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

This matches the PRD more closely than fixed-part segmentation, while keeping
the current code measurable and easy to debug.
