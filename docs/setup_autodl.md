## AutoDL 5090 Setup

This project is designed so Mac-side work can focus on coding and tests, while GPU
training runs on AutoDL.

```bash
cd /root/projects/alibaba-ai
pip install -r requirements.txt
pip install -e .
python scripts/setup/check_gpu.py
```

Expected training command for `3.1.1`:

```bash
python scripts/train/train_instance_segmentation.py \
  --model-config configs/model/instance_segmentation_deepfashion2.yaml \
  --paths-config configs/paths.autodl.yaml
```

Sanity evaluation command for `3.1.2`:

```bash
cd /root/projects/alibaba-ai
git pull
python scripts/eval/evaluate_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --max-images 20 \
  --output /root/autodl-tmp/outputs/local_region_query_eval.json \
  --vis-dir /root/autodl-tmp/outputs/local_region_vis \
  --vis-count 20
```

The `3.1.2` command runs the frozen `3.1.1` garment instance model first, then
uses the open-vocabulary local-region candidate ranker for queries such as
`左边的袖口`, `右侧的口袋`, `衣服上的拉链`, and `这件衣服上的碎花图案`.
Review the JSON summary and visualization directory, but do not treat this
sanity run as the PRD metric.

### Current 3.1.2 Direction

After the manual benchmark and mentor feedback, the recommended direction is:

1. Keep heuristic-only local-region inference as the online control baseline.
2. Use the manual bbox benchmark as the main decision metric.
3. Add an offline pretrained grounding baseline next, such as GroundingDINO,
   OWL-ViT/OWL-V2, or CLIP/Chinese-CLIP crop reranking with Chinese-to-English
   query templates where needed.
4. Keep DeepFashion2 landmark pseudo-label and candidate-listwise ranker work as
   historical weak-supervision experiments. They are useful for debugging, but
   they are not enough to prove language-guided localization accuracy.

Weak-label evaluation command for `3.1.2`:

```bash
cd /root/projects/alibaba-ai
git pull
python scripts/eval/evaluate_local_region_weak_labels.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/validation/annos \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --max-images 50 \
  --output /root/autodl-tmp/outputs/local_region_weak_eval.json
```

Weak-label evaluation with the hybrid learned ranker:

```bash
python scripts/eval/evaluate_local_region_weak_labels.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/validation/annos \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --ranker-checkpoint /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry_500k.pt \
  --device cuda \
  --max-images 200 \
  --output /root/autodl-tmp/outputs/local_region_weak_eval_hybrid_200.json
```

This weak evaluation compares predicted local-region masks with approximate
DeepFashion2 landmark-derived labels for queries such as neckline, hem, and
shoulder. Treat it as a debugging metric, not the final PRD accuracy number.

Build a small manual bbox benchmark to avoid optimizing only against noisy
pseudo-labels. This should be small, e.g. 100-300 image-query pairs, and should
be used only for evaluation:

```bash
cd /root/projects/alibaba-ai
PYTHONPATH=src python scripts/data/build_local_region_manual_eval_manifest.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/validation/annos \
  --max-images 50 \
  --max-records 150 \
  --shuffle \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_manifest.jsonl
```

With `--anno-dir`, the manifest uses category-aware query templates. For
example, pants receive waist/pant-hem/pocket/zipper queries instead of neckline
or shoulder queries. This should reduce the skip rate in the next annotation
round.

Start the browser annotator and drag one bbox for each image-query pair. The
tool automatically writes `[x1, y1, x2, y2]` pixel coordinates, so there is no
need to calculate bbox numbers manually:

```bash
PYTHONPATH=src python scripts/data/annotate_local_region_bboxes.py \
  --manifest /root/autodl-tmp/outputs/local_region_manual_eval_manifest.jsonl \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_labeled.jsonl \
  --host 0.0.0.0 \
  --port 7860
```

Open the printed URL through the AutoDL port/proxy UI or by SSH port
forwarding. Do not use landmarks or the weak-label generator while labeling
this file.

Evaluate the full 3.1.2 pipeline against the manual benchmark:

```bash
PYTHONPATH=src python scripts/eval/evaluate_local_region_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic.json
```

Use this manual result as the independent benchmark. The landmark/rule
weak-label metric remains useful for debugging and large-scale development, but
it is not a replacement for human-localized query regions. The first 55-record
manual benchmark favored the pure heuristic baseline over the hem-gated
candidate-listwise hybrid (`0.2544` vs `0.2324` average bbox IoU), so the
current online policy is heuristic-only.

After multiple annotation rounds, merge the labeled files and run the combined
benchmark:

```bash
PYTHONPATH=src python scripts/data/merge_local_region_manual_eval_labels.py \
  --inputs \
    /root/autodl-tmp/outputs/local_region_manual_eval_labeled.jsonl \
    /root/autodl-tmp/outputs/local_region_manual_eval_labeled_class_aware.jsonl \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl

PYTHONPATH=src python scripts/eval/evaluate_local_region_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined.json
```

Current combined result: `122` labeled records, average bbox IoU `0.2812`,
Hit@0.3 `0.4344`, Hit@0.5 `0.2623`. Export low-IoU manual failure cases for
qualitative analysis:

```bash
PYTHONPATH=src python scripts/eval/export_local_region_manual_failures.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined.json \
  --output-dir /root/autodl-tmp/outputs/local_region_manual_failures_combined \
  --iou-threshold 0.1 \
  --regions cuff pocket waist \
  --max-cases 80
```

The export directory contains per-case images, `failure_summary.json`, and
`failure_review.html` for grouped visual inspection.

After the first failure review, the online heuristic policy was refined for
cuff, pocket, and waist: side-specific cuff/pocket queries use garment/wearer
left-right convention, cuff candidates are narrowed to sleeve ends, and waist
uses category-aware vertical bands. Re-run the combined manual benchmark after
pulling the change:

```bash
PYTHONPATH=src python scripts/eval/evaluate_local_region_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_refined.json
```

Current refined result: `122` labeled records, average bbox IoU `0.3064`,
Hit@0.3 `0.4754`, Hit@0.5 `0.2787`. Compared with the pre-refinement manual
benchmark, cuff improved from `0.0190` to `0.0592`, pocket from `0.0000` to
`0.1337`, and waist from `0.0961` to `0.2306`. Export the remaining cuff
failures for the next review pass:

```bash
PYTHONPATH=src python scripts/eval/export_local_region_manual_failures.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_refined.json \
  --output-dir /root/autodl-tmp/outputs/local_region_manual_failures_refined_cuff \
  --iou-threshold 0.3 \
  --regions cuff \
  --max-cases 80
```

The second cuff review showed that short-sleeve/armhole examples often need an
upper-side sleeve candidate, while long sleeves still need a lower terminal
candidate. The online policy therefore generates both cuff variants and lets the
heuristic ranker choose by confidence. Re-run the manual benchmark after pulling
that change:

```bash
PYTHONPATH=src python scripts/eval/evaluate_local_region_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_cuff_variants.json
```

Current cuff-variant result: `122` labeled records, average bbox IoU `0.3123`,
Hit@0.3 `0.4836`, Hit@0.5 `0.2705`. Cuff improved from `0.0592` to `0.0904`,
so the variant policy is better than the previous heuristic, but cuff is still
the main residual weakness.

Next recommended AutoDL work is to evaluate a pretrained grounding baseline on
`/root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl`.
Do not run more weak-label ranker training as the main path unless the
pretrained grounding baseline has already been measured.

First pretrained grounding baseline command:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_pretrained_grounding_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --backend owlvit \
  --model-name google/owlvit-base-patch32 \
  --prompt-mode english \
  --device cuda \
  --score-threshold 0.05 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_owlvit.json
```

Observed OWL-ViT base result on the 122-record manual benchmark: average bbox
IoU `0.0305`, Hit@0.3 `0.0410`, Hit@0.5 `0.0000`, with `101/122` records
returning `no_detection`. This is a negative baseline and is far below the
heuristic control result (`0.3123` average bbox IoU).

Run GroundingDINO tiny next:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_pretrained_grounding_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
  --backend auto \
  --model-name IDEA-Research/grounding-dino-tiny \
  --prompt-mode english \
  --device cuda \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_grounding_dino_tiny.json
```

Observed GroundingDINO tiny result on the 122-record manual benchmark: average
bbox IoU `0.2225`, Hit@0.3 `0.2295`, Hit@0.5 `0.1639`. This is below the
heuristic control overall, but it is much better for visual semantic regions:
pattern `0.8262`, zipper `0.8233`, neckline `0.3843`. It remains weak for
structural geometry regions: cuff `0.0698`, hem `0.1345`, shoulder `0.1895`,
pocket `0.0332`.

Compare the heuristic and GroundingDINO outputs by region:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src python scripts/eval/compare_local_region_manual_evals.py \
  --eval-json \
    /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_cuff_variants.json \
    /root/autodl-tmp/outputs/local_region_manual_eval_grounding_dino_tiny.json \
  --names heuristic grounding_dino_tiny \
  --default-eval heuristic \
  --region-policy pattern=grounding_dino_tiny zipper=grounding_dino_tiny \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_vs_grounding_dino.json
```

The fixed semantic-region hybrid (`pattern/zipper -> GroundingDINO`, all other
regions -> heuristic) reaches average bbox IoU `0.3465`, Hit@0.3 `0.5246`, and
Hit@0.5 `0.3197` on the 122-record manual benchmark. The practical next design
is therefore a gated hybrid: keep heuristic geometry for structural regions and
use visual grounding only for semantic appearance details such as pattern and
zipper. Do not switch the full online path until this holds on a larger manual
split.

Build the next targeted manual benchmark for semantic/detail regions:

```bash
cd /root/projects/alibaba-ai
git pull
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

Then annotate it with the existing bbox tool:

```bash
PYTHONPATH=src python scripts/data/annotate_local_region_bboxes.py \
  --manifest /root/autodl-tmp/outputs/local_region_manual_eval_manifest_semantic_150.jsonl \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_labeled_semantic_150.jsonl \
  --host 0.0.0.0 \
  --port 7860
```

Targeted semantic split result: `49` labeled records and `101` skipped records.
GroundingDINO is better than heuristic on this semantic-only split overall:
average bbox IoU `0.2133` vs `0.1296`, Hit@0.3 `0.2857` vs `0.1633`, Hit@0.5
`0.2245` vs `0.0408`. Per-region, GroundingDINO is stronger on pattern
(`0.5591` vs `0.3046`) and pocket (`0.1162` vs `0.0096`), while zipper is still
slightly better with the heuristic (`0.1637` vs `0.1334`). For this split, use
the fixed policy `pattern/pocket -> GroundingDINO`, all other regions ->
heuristic.

Merge the original and semantic manual labels. The merge script deduplicates by
annotation `id` when present, so same-image multi-item semantic records are kept:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src python scripts/data/merge_local_region_manual_eval_labels.py \
  --inputs \
    /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined.jsonl \
    /root/autodl-tmp/outputs/local_region_manual_eval_labeled_semantic_150.jsonl \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl
```

Re-evaluate heuristic and GroundingDINO on the merged manual benchmark:

```bash
PYTHONPATH=src python scripts/eval/evaluate_local_region_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json

PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_pretrained_grounding_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --backend auto \
  --model-name IDEA-Research/grounding-dino-tiny \
  --prompt-mode english \
  --device cuda \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_grounding_dino_combined_plus_semantic.json
```

Compare the fixed `pattern/pocket` hybrid on the merged manual benchmark:

```bash
PYTHONPATH=src python scripts/eval/compare_local_region_manual_evals.py \
  --eval-json \
    /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
    /root/autodl-tmp/outputs/local_region_manual_eval_grounding_dino_combined_plus_semantic.json \
  --names heuristic grounding_dino_tiny \
  --default-eval heuristic \
  --region-policy pattern=grounding_dino_tiny pocket=grounding_dino_tiny \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_fixed_pattern_pocket_combined_plus_semantic.json
```

Final merged benchmark result (`122 + 49 = 171` labeled records):

- heuristic-only: avg bbox IoU `0.2599`, Hit@0.3 `0.3918`, Hit@0.5 `0.2047`
- GroundingDINO-only: avg bbox IoU `0.2199`, Hit@0.3 `0.2456`, Hit@0.5 `0.1813`
- fixed `pattern/pocket` hybrid: avg bbox IoU `0.3060`, Hit@0.3 `0.4503`,
  Hit@0.5 `0.2749`

Per-region evidence on the merged benchmark:

- `pattern`: GroundingDINO `0.6691` vs heuristic `0.3080`
- `pocket`: GroundingDINO `0.1024` vs heuristic `0.0303`
- `zipper`: roughly tied, so keep heuristic by default
- structural regions (`hem`, `shoulder`, `neckline`, `cuff`, `waist`): keep
  heuristic

Next implementation step: add an explicit experimental gated hybrid path. Do not
silently change the default `localize_region_from_instances` behavior, because
GroundingDINO requires original image pixels and is much heavier than the
heuristic geometry path.

Run the explicit experimental gated-hybrid evaluator:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_gated_hybrid_manual_labels.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --grounding-regions pattern pocket \
  --grounding-backend auto \
  --grounding-model-name IDEA-Research/grounding-dino-tiny \
  --prompt-mode english \
  --score-threshold 0.15 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json
```

This command executes the policy directly instead of stitching two completed
eval JSON files together. It should be close to the fixed hybrid comparison
result (`0.3060` average bbox IoU) if the gated path is implemented correctly.
Observed result: avg bbox IoU `0.3060`, Hit@0.3 `0.4503`, Hit@0.5 `0.2749`,
with `41` records routed to GroundingDINO and `130` records routed to the
heuristic path.

Run the same policy on a single image only when explicitly testing the gated
experimental path. The default `predict_local_region.py` command remains
heuristic-only:

```bash
cd /root/projects/alibaba-ai
git pull
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

For `pattern` and `pocket` queries this script loads GroundingDINO and does not
run the 3.1.1 segmentation model. For all other parsed regions, it runs the
existing segmentation plus heuristic local-region path and writes the same
local-region JSON shape as the default command, with `gated_policy_route`
showing which branch was used.

Run a small batch gated-hybrid query demo after the two single-image routes are
verified:

```bash
cd /root/projects/alibaba-ai
git pull
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

Check the printed summary and JSON for `gated_policy_route_counts`,
`ranker_backend_counts`, and `avg_local_region_latency_by_route_ms`. Then review
the visualization folder to decide whether the gated hybrid is good enough as a
3.1.2 demo path.

For a fair visual demo, prefer a per-record manifest instead of running every
query on every image. This avoids impossible prompts such as asking for a
visible pocket on images with no pocket:

```json
{"image": "/root/autodl-tmp/datasets/DeepFashion2/validation/image/000003.jpg", "query_text": "这件衣服上的碎花图案"}
{"image": "/root/autodl-tmp/datasets/DeepFashion2/validation/image/000010.jpg", "query_text": "右侧的口袋"}
{"image": "/root/autodl-tmp/datasets/DeepFashion2/validation/image/000012.jpg", "query_text": "衣服下方的下摆"}
```

Build this manifest from the completed gated manual evaluation so the selection
rule is reproducible. It selects the highest-IoU successful records for each
requested region and writes the route and selection IoU into the JSONL. These
are qualitative examples only; report the `171`-record manual benchmark for
performance. The resulting visualization overlays the manual reference bbox in
green as `GT`; the predicted local region remains orange.

```bash
PYTHONPATH=src python scripts/data/build_gated_hybrid_demo_manifest.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --target-regions pattern neckline hem shoulder \
  --per-region 2 \
  --min-iou 0.3 \
  --require-full-quota \
  --output /root/autodl-tmp/outputs/local_region_gated_demo_manifest.jsonl
```

Before revising the fixed `pattern/pocket` gate, analyze whether low-confidence
GroundingDINO cases should fall back to heuristic geometry. This reuses the two
completed manual-eval JSON files, splits by image for a small held-out check,
and does not run any model:

```bash
PYTHONPATH=src python scripts/eval/analyze_gated_hybrid_confidence.py \
  --gated-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --heuristic-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --grounding-regions pattern pocket \
  --thresholds 0.0 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 \
  --holdout-fraction 0.3 \
  --output /root/autodl-tmp/outputs/local_region_gated_confidence_analysis.json
```

Treat this as an offline calibration analysis. Keep the existing policy unless
the selected threshold improves the image-held-out semantic summary, then rerun
the full gated manual evaluator before reporting an improvement. In the output,
compare `holdout_results` at threshold `0.0` with the selected threshold.

The current threshold analysis does not justify score-based fallback. Test
English prompt wording before changing the gate; this command loads the model
once and evaluates the current synonym ensemble, a single direct phrase, and a
clothing-context phrase on only the manually labeled semantic records:

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

Then produce a paired qualitative review of the current gated policy. It does
not run a model and can use CPU:

```bash
PYTHONPATH=src python scripts/eval/export_gated_hybrid_policy_deltas.py \
  --baseline-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --candidate-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --regions pattern pocket \
  --candidate-routes grounding \
  --min-abs-delta 0.05 \
  --output-dir /root/autodl-tmp/outputs/local_region_gated_pattern_pocket_deltas
```

Open `policy_delta_review.html` after downloading the output directory. Green
is the manual bbox, red heuristic-only, and blue the gated prediction. Use the
review to identify prompt competition, small-object false positives, and
left/right ambiguity; rerun the full manual benchmark after any change.

The paired review showed that a small number of GroundingDINO regressions are
background objects (for example, a bag) or background patterns. Test the
experimental garment-mask constraint next. It uses the frozen 3.1.1 mask only
to reject grounding boxes that do not sufficiently overlap the selected garment;
when all grounding detections are rejected, it falls back to the existing
heuristic result. This is a manual-evaluation experiment, not the default
online policy:

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

Compare its full-benchmark `manual_hit_at["0.3"]` with the current gated value
`0.4503`. Also inspect `ranker_backend_counts`: `heuristic_fallback` records
mean a grounding box was rejected by the garment mask. Keep the constraint only
if it improves the full manual result; it adds segmentation work to semantic
queries and is therefore unsuitable for the latency-sensitive default path
until validated.

Observed result: the garment constraint reached `0.4386` Hit@0.3, below the
unconstrained gated policy (`0.4503`), so do not use it further. Measure the
per-record routing ceiling before introducing another router:

```bash
PYTHONPATH=src python scripts/eval/analyze_local_region_routing_oracle.py \
  --baseline-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_heuristic_combined_plus_semantic.json \
  --candidate-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_gated_pattern_pocket_combined_plus_semantic.json \
  --output /root/autodl-tmp/outputs/local_region_routing_oracle_heuristic_vs_gated.json
```

This is a theoretical best-of-two upper bound. It must not be reported as a
model result or directly implemented. If it is below the 60% Hit@0.3 target,
router tuning cannot meet the target and the next effort must improve cuff,
waist, or pocket localization itself.

The completed 171-record oracle is Hit@0.3 `0.4561`: it selects heuristic for
148 records and the gated GroundingDINO result for 23. This rules out further
routing or threshold tuning as a path to the 60% target. Evaluate a new
visual-text expert directly on the manual benchmark instead.

Chinese-CLIP crop reranking is the next frozen pretrained baseline. Unlike the
historical weak-candidate experiment below, this command uses only manual
evaluation labels for scoring and never consumes landmark pseudo labels for the
decision metric. It encodes the original Chinese query and crop candidates
inside the frozen 3.1.1 selected garment instance:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python scripts/eval/evaluate_chinese_clip_manual_local_regions.py \
  --annotations /root/autodl-tmp/outputs/local_region_manual_eval_labeled_combined_plus_semantic.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --device cuda \
  --region-prior-weights 0.0,0.05,0.1,0.2 \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_chinese_clip_candidates.json
```

The output contains one `runs` entry per prior weight. Compare each full
171-record result against heuristic (`Hit@0.3 0.3918`) and gated
GroundingDINO (`0.4503`) before any online integration. The current
heuristic-only default remains unchanged.

Observed result: the best Chinese-CLIP prior settings (`0.1`, `0.2`) reached
Hit@0.3 `0.3860`, below heuristic-only (`0.3918`). Cuff, pocket, and zipper
remain unreliable, so the visual score is not a useful new expert; the prior
mostly restores the rule-derived candidate region. Do not wire this reranker
into the online path.

The next controlled pretrained comparison is the larger GroundingDINO-base.
Run it as an offline full manual evaluation first; do not add new gated regions
until its per-region results improve on both the current heuristic and tiny
GroundingDINO outputs:

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

Observed base result: pocket Hit@0.3 improves to `0.2083` from `0.1250`, and
cuff improves to `0.1304` from `0.0870`. Pattern is still stronger with tiny;
hem, shoulder, and neckline remain better with the heuristic. Use the new
explicit multi-expert route capability to verify this fixed, limited policy as
a real pipeline run:

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

`--grounding-routes` is authoritative: it replaces the single-model
`--grounding-regions` configuration. The evaluator loads each unique model
once, records its name per grounding prediction, and routes every other target
region through the heuristic. This is an exploratory same-benchmark policy,
not the default online path or an independent final result.

Observed real-pipeline result: multi-expert routing reaches average manual IoU
`0.3082`, Hit@0.3 `0.4678`, and Hit@0.5 `0.2924`. This confirms the pocket and
cuff gain, but it remains 23 Hit@0.3 successes below the weekly 60% target.
Do not keep tuning the same tiny/base routing combination.

Use a different visual-text detector family next. Run OWLv2-large as a prompt
profile ablation on only the 79 weak cuff/pocket/zipper/waist records first;
the model is loaded once and reused for all profiles:

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

This result is not comparable to the overall benchmark because it contains
only hard regions. Use it only to select a promising OWLv2 prompt profile;
then run the selected profile on all 171 records before changing a route.

Observed OWLv2 result: `precise` raises cuff Hit@0.3 to `0.2174` from the
base-route `0.1304`, while `ensemble` raises waist to `0.5000` from heuristic
`0.3333`. Pocket only ties the base route and zipper remains below heuristic,
so route only cuff and waist to OWLv2. The evaluator supports independent
per-region prompt profiles and thresholds, which prevents the OWLv2 `0.05`
threshold from changing the validated GroundingDINO `0.15` routes:

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

This full output is the decision metric. It will load the two GroundingDINO
models and OWLv2 once each; zipper remains heuristic by design. Do not claim
the expected gain until the complete manual benchmark has run.

Run the same evaluator with `--manifest`:

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

### Archived Weak-Supervision Commands

Build weak query-region records for the learned `3.1.2` ranker:

```bash
cd /root/projects/alibaba-ai
git pull
python scripts/data/build_deepfashion2_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/train/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/train/annos \
  --output /root/autodl-tmp/outputs/local_region_train_queries.jsonl
```

The JSONL records contain image paths, item keys, Chinese query templates,
garment boxes, weak local-region boxes, and whether the region came from
landmarks or rule fallback. Use this as the first weak supervision source for
the archived learned text-region matching baseline.

Train a lightweight learned text-region ranker:

```bash
cd /root/projects/alibaba-ai
git pull
python scripts/train/train_local_region_ranker.py \
  --records /root/autodl-tmp/outputs/local_region_train_queries.jsonl \
  --output /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry.pt \
  --device cuda \
  --max-records 50000 \
  --val-records 2000 \
  --num-epochs 1
```

This first learned baseline uses hashed Chinese query text plus normalized
candidate geometry. It is not the final DINOv2/CLIP-style ranker, but it gives
a reproducible checkpoint and top-1 weak IoU metric. It should not be expanded
as the main 3.1.2 plan after the manual benchmark showed no online gain.

Initial 50k-record smoke result: loss `0.4699`, validation top-1 box IoU
`0.3540`.

500k-record offset-validation result: loss `0.4465`, validation top-1 box IoU
`0.3560`.

For a larger run with a later validation slice:

```bash
python scripts/train/train_local_region_ranker.py \
  --records /root/autodl-tmp/outputs/local_region_train_queries.jsonl \
  --output /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry_500k.pt \
  --device cuda \
  --max-records 500000 \
  --val-records 10000 \
  --val-offset 500000 \
  --num-epochs 1
```

Run sanity evaluation with the hybrid learned ranker checkpoint:

```bash
python scripts/eval/evaluate_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --ranker-checkpoint /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry_500k.pt \
  --device cuda \
  --max-images 20 \
  --output /root/autodl-tmp/outputs/local_region_query_eval_hybrid.json \
  --vis-dir /root/autodl-tmp/outputs/local_region_vis_hybrid
```

The hash checkpoint remains an experimental branch. After the manual bbox
benchmark, the default online policy is to omit `--ranker-checkpoint` and use
the pure heuristic open-vocabulary pipeline unless a learned branch improves
the manual benchmark.

20-image hybrid sanity result: 140/140 ok, average local-region latency
`16.93 ms`, and open-query outputs remain diverse instead of collapsing to the
whole garment.

200-image neckline/hem-only hybrid weak-label result: average weak IoU `0.2822`.
This recovers the tuned heuristic baseline, but the gain is too small to treat
the hash text-geometry scorer as the final model.

Export candidate-level records for historical CLIP/OpenCLIP or DINOv2
text-region experiments:

```bash
python scripts/data/build_local_region_candidate_records.py \
  --records /root/autodl-tmp/outputs/local_region_train_queries.jsonl \
  --output /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --max-records 500000
```

Each input query record is expanded into candidate boxes with IoU labels against
the weak region box. This keeps image paths and candidate boxes together, so the
experimental scripts can crop candidate regions and learn image-text matching
instead of relying only on geometry. Because the target boxes are still
weak-label boxes, validate any gain on the manual benchmark before drawing a
3.1.2 conclusion.

Install the Chinese-CLIP dependencies:

```bash
pip install "transformers>=4.37.0" sentencepiece
```

Evaluate frozen Chinese-CLIP candidate reranking:

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/eval/evaluate_chinese_clip_local_region_ranker.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --device cuda \
  --max-groups 2000 \
  --region-prior-weights 0,0.01,0.02,0.05,0.1,0.2 \
  --output /root/autodl-tmp/outputs/local_region_chinese_clip_eval_2k.json
```

This baseline uses the Chinese query directly, crops each candidate box, and
ranks candidates by Chinese-CLIP image-text cosine similarity. It is a stronger
fit than OpenCLIP here because the 3.1.2 queries are Chinese.
The optional region-prior sweep adds a small score bonus when the candidate name
matches the region parsed from the query, which helps measure whether CLIP
features are useful beyond the existing query parser.

If the mirror is unavailable, download `OFA-Sys/chinese-clip-vit-base-patch16`
to an AutoDL-local directory and pass that directory with `--model-name`.

When Chinese-CLIP selection is worse than the heuristic baseline, run candidate
diagnostics to separate candidate quality from scorer quality:

```bash
python scripts/eval/evaluate_local_region_candidate_baselines.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --max-groups 2000 \
  --output /root/autodl-tmp/outputs/local_region_candidate_baselines_2k.json
```

Use `oracle_best_iou` as the candidate-set upper bound and
`target_region_name` as the label-name baseline before training a CLIP-feature
ranker.

Train a supervised listwise candidate ranker from the candidate JSONL:

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

This trains against either the best-IoU candidate (`--loss hard`) or a soft IoU
distribution (`--loss soft`) in each query group. It uses query text, candidate
region, normalized geometry, absolute box context, and garment category text.
Compare its `val_top1_iou` with the `target_region_name` baseline and the
oracle upper bound before wiring it into the online 3.1.2 predictor.

50k context-feature result: validation top-1 IoU `0.5113`, compared with
`0.3589` for the target-region-name baseline and `0.5704` for the candidate-set
oracle on the same 2k slice.

Validate the saved checkpoint on a later candidate slice:

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

Historical full weak-label result on 200 images without gating: average weak
IoU `0.2732`, below the safer tuned baseline. A hem-only gate recovered the
weak metric, but the manual bbox benchmark showed the pure heuristic baseline
is better online than the hem-gated candidate-listwise hybrid.
Candidate-listwise checkpoints are therefore disabled in online inference by
default and should remain an experimental branch until they improve the manual
benchmark.

AutoDL dataset and checkpoint paths are configured in `configs/paths.autodl.yaml`.

## Audited Side-Aware Cuff Evaluation

The audited benchmark has 161 valid records. Re-run the fixed four-expert
policy with wearer-side selection enabled only for cuff:

```bash
cd /root/projects/alibaba-ai
git pull

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

Expected from the fixed offline analysis: Hit@0.3 improves from 84/161 to
87/161. Confirm the actual JSON before reporting it. Then run the CPU-only
candidate ceiling analysis:

```bash
PYTHONPATH=src python scripts/eval/analyze_grounding_candidate_oracle.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_four_expert_side_cuff_audited.json \
  --regions cuff pocket pattern waist \
  --hit-threshold 0.3 \
  --output /root/autodl-tmp/outputs/local_region_grounding_candidate_oracle_audited.json
```

The grounding-only Top-5 oracle reached 98/161 Hit@0.3, only one hit above the
97/161 target. The command above also records a heuristic candidate for each
grounding-routed record without selecting it. Report the expanded
`candidate_oracle_summary.manual_hit_at`, each region's `recoverable_failures`,
`oracle_source_counts`, and `oracle_rank_counts` before choosing the next model
experiment.

Add GroundingDINO-base zipper detections without changing the selected policy
by repeating the side-aware evaluation command above with:

```bash
  --diagnostic-grounding-routes zipper=IDEA-Research/grounding-dino-base \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_zipper_candidates_audited.json
```

Keep `--record-heuristic-candidates-for-grounding` enabled. The selected
Hit@0.3 must remain 87/161. Then run:

```bash
PYTHONPATH=src python scripts/eval/analyze_grounding_candidate_oracle.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_zipper_candidates_audited.json \
  --regions cuff pocket pattern waist zipper \
  --hit-threshold 0.3 \
  --output /root/autodl-tmp/outputs/local_region_zipper_candidate_oracle_audited.json
```

The zipper oracle reaches 101/161 Hit@0.3. Run the next cross-model candidate
experiment by replacing the single diagnostic route with:

```bash
  --diagnostic-grounding-routes \
    pattern=IDEA-Research/grounding-dino-base \
    pocket=IDEA-Research/grounding-dino-tiny \
    cuff=IDEA-Research/grounding-dino-base \
    waist=IDEA-Research/grounding-dino-base \
    zipper=IDEA-Research/grounding-dino-base \
  --output /root/autodl-tmp/outputs/local_region_manual_eval_cross_model_candidates_audited.json
```

No additional model is loaded: tiny and base are already used by the selected
pattern and pocket routes. Re-run the candidate oracle on the new output with
`--regions cuff pocket pattern waist zipper`.

The cross-model oracle reaches 107/161 Hit@0.3. Evaluate a selector without
manual-label leakage on CPU:

```bash
PYTHONPATH=src python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_cross_model_candidates_audited.json \
  --regions cuff pocket pattern waist zipper \
  --num-folds 5 \
  --num-epochs 120 \
  --hidden-dim 48 \
  --learning-rate 0.003 \
  --weight-decay 0.01 \
  --seed 42 \
  --device cpu \
  --output /root/autodl-tmp/outputs/local_region_candidate_selector_5fold_audited.json
```

Report `out_of_fold_summary`, `selector_diagnostics`, and every fold summary.
Do not use `candidate_oracle_summary` as the achieved model result.

The listwise run reaches only 85/161 Hit@0.3 (`0.5280`), versus the current
policy's 87/161 (`0.5404`): seven hits are gained but nine are lost. Do not use
this selector online. Run the conservative current-versus-alternative selector:

```bash
PYTHONPATH=src python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_cross_model_candidates_audited.json \
  --regions cuff pocket pattern waist zipper \
  --num-folds 5 \
  --num-epochs 120 \
  --hidden-dim 48 \
  --learning-rate 0.003 \
  --weight-decay 0.01 \
  --selection-policy conservative_pairwise \
  --override-threshold 0.5 \
  --seed 42 \
  --device cpu \
  --output /root/autodl-tmp/outputs/local_region_candidate_selector_conservative_5fold_audited.json
```

Keep `--override-threshold 0.5` fixed for this OOF run. The relevant result is
still `out_of_fold_summary`, while `override_counts` shows how often current
was replaced.

Observed conservative result: 20 overrides produce three gained hits and five
lost hits. Full OOF Hit@0.3 remains 85/161 (`0.5280`), so this selector is also
disabled online. Add frozen Chinese-CLIP evidence to the existing candidates:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com \
python scripts/eval/enrich_grounding_candidates_with_chinese_clip.py \
  --eval-json /root/autodl-tmp/outputs/local_region_manual_eval_cross_model_candidates_audited.json \
  --regions cuff pocket pattern waist zipper \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --prompt-profile region_ensemble \
  --context-scale 1.6 \
  --image-batch-size 32 \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_cross_model_candidates_chinese_clip_audited.json
```

This pass needs the GPU but does not load the segmentation or grounding models.
It records tight-crop and 1.6x-context similarities plus within-query ranks,
without reading `target_bbox`. Run selector training on CPU afterward:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_cross_model_candidates_chinese_clip_audited.json \
  --regions cuff pocket pattern waist zipper \
  --num-folds 5 \
  --num-epochs 120 \
  --hidden-dim 48 \
  --learning-rate 0.003 \
  --weight-decay 0.01 \
  --selection-policy conservative_pairwise \
  --override-threshold 0.5 \
  --seed 42 \
  --device cpu \
  --output /root/autodl-tmp/outputs/local_region_candidate_selector_clip_conservative_5fold_audited.json
```

Before interpreting the metric, verify `num_records_with_visual_scores` is 86.

Observed result with the fixed-threshold visual MLP: full OOF Hit@0.3 remains
85/161 (`0.5280`). Its 24 overrides gain four hits and lose six. Cuff gains
four and loses two, while pocket, waist, and zipper produce no gains. Reject
this run and do not tune the threshold from its outer OOF records.

Run the nested, region-gated linear selector on the existing enriched JSON; no
GPU inference is repeated:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_cross_model_candidates_chinese_clip_audited.json \
  --regions cuff pocket pattern waist zipper \
  --num-folds 5 \
  --inner-folds 3 \
  --num-epochs 200 \
  --selector-architecture linear \
  --selection-policy conservative_pairwise \
  --threshold-policy nested_region \
  --nested-thresholds 0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --nested-max-lost-hits 0 \
  --nested-min-net-gain 1 \
  --learning-rate 0.01 \
  --weight-decay 0.01 \
  --seed 42 \
  --device cpu \
  --output /root/autodl-tmp/outputs/local_region_candidate_selector_clip_nested_linear_5fold_audited.json
```

For every outer fold, threshold selection sees only three-fold inner OOF
predictions from the outer training images. A region remains on the current
policy unless inner OOF has at least one net gain and zero lost hits. Inspect
both `nested_region_activation_counts` and final `selector_diagnostics`.

Observed nested result: OOF Hit@0.3 is `85/161` (`0.5280`) versus the current
policy's `87/161` (`0.5404`). It performs five overrides, gains zero hits, and
loses two. Do not integrate or continue tuning this selector on the manual set.

### Independent Weak-Train Selector

Build a randomized landmark-only train set on CPU. This command excludes all
rule fallback targets and exports one visual per image/item/region for review:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src python scripts/data/build_deepfashion2_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/train/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/train/annos \
  --regions left_cuff right_cuff waist \
  --landmark-only \
  --shuffle \
  --seed 42 \
  --max-images 5000 \
  --vis-dir /root/autodl-tmp/outputs/local_region_train_landmark_vis \
  --vis-count 40 \
  --output /root/autodl-tmp/outputs/local_region_train_landmark_cuff_waist.jsonl
```

Before using a GPU, inspect the 40 images. Green must be a tight cuff/waist
weak target, blue the source garment. Sleeve-less items must not appear as cuff
records, and the reported `source_counts` must contain only
`landmark_pseudo_label`. DeepFashion2 names contours by image side, while this
project uses garment/wearer side. The builder therefore swaps cuff pairs for
frontal/flat-lay images and reports `cuff_side_convention` plus the known back
view limitation in its summary.

Run a 100-record online candidate smoke test on the 5090. The selected OWLv2
route, GroundingDINO-base diagnostic route, and frozen heuristic candidate all
run before the landmark target is used for IoU scoring:

```bash
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python \
  scripts/data/build_online_local_region_weak_candidates.py \
  --queries /root/autodl-tmp/outputs/local_region_train_landmark_cuff_waist.jsonl \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --regions cuff waist \
  --max-records 100 \
  --grounding-routes \
    cuff=google/owlv2-large-patch14-ensemble \
    waist=google/owlv2-large-patch14-ensemble \
  --grounding-route-profiles cuff=precise waist=ensemble \
  --grounding-route-thresholds cuff=0.05 waist=0.05 \
  --diagnostic-grounding-routes \
    cuff=IDEA-Research/grounding-dino-base \
    waist=IDEA-Research/grounding-dino-base \
  --grounding-backend auto \
  --prompt-mode english \
  --prompt-profile ensemble \
  --score-threshold 0.15 \
  --wearer-side-regions cuff \
  --wearer-side-min-score-ratio 0.5 \
  --output /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_100.json
```

Check that `supervision_type` is `landmark_pseudo_label_only`,
`candidate_generation_uses_target_bbox` is `false`, and both cuff variants are
present. If correct, repeat with `--max-records 2000` and change the output
suffix to `_2000`.

Train and calibrate on those external weak records, then evaluate once on the
existing frozen manual candidate artifact. The selector training supports CUDA:

```bash
PYTHONPATH=src python \
  scripts/eval/train_external_grounding_candidate_selector.py \
  --train-eval-json /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_2000.json \
  --test-eval-json /root/autodl-tmp/outputs/local_region_manual_eval_cross_model_candidates_audited.json \
  --regions cuff waist \
  --calibration-folds 5 \
  --selector-architecture linear \
  --num-epochs 200 \
  --learning-rate 0.01 \
  --weight-decay 0.01 \
  --thresholds 0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --max-calibration-lost-hits 0 \
  --min-calibration-net-gain 1 \
  --seed 42 \
  --device cuda \
  --model-output /root/autodl-tmp/checkpoints/local_region_ranker/external_weak_cuff_waist.pt \
  --output /root/autodl-tmp/outputs/local_region_external_weak_selector_audited.json
```

The output must report `test_labels_used_for_training_or_calibration: false`
and `train_test_image_overlap: 0`. The achieved result is only
`frozen_test_summary`; weak-train and oracle summaries are diagnostics.

Before opening the frozen manual artifact, enrich the independent weak
candidate pool with frozen DINOv2 region embeddings. This command preserves
existing Chinese-CLIP scalar scores and never reads the target bbox:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python \
  scripts/eval/enrich_grounding_candidates_with_dinov2.py \
  --eval-json /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_chinese_clip_2338_v2.json \
  --regions cuff waist \
  --model-name facebook/dinov2-base \
  --context-scale 1.6 \
  --image-batch-size 32 \
  --projection-seed 42 \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_chinese_clip_dinov2_2338_v2.json
```

Require `num_scored_records == 2338`, `projection_dim == 64`, a non-empty
`projection_fingerprint`, and `target_bbox_used_for_features == false`.

The completed region-conditioned diagnostics show that soft-target linear
reaches Hit@0.3 `0.4778` (1,117/2,338), multi-positive linear also reaches
`0.4778`, and multi-positive MLP falls to `0.4542`. Do not rerun the MLP or tune
the loss. Add cuff-only patch spatial descriptors to the existing enriched
artifact instead:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src HF_ENDPOINT=https://hf-mirror.com python \
  scripts/eval/enrich_grounding_candidates_with_dinov2.py \
  --eval-json /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_chinese_clip_dinov2_2338_v2.json \
  --regions cuff \
  --model-name facebook/dinov2-base \
  --feature-mode spatial_pyramid \
  --context-scale 1.6 \
  --image-batch-size 32 \
  --projection-seed 42 \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_clip_dinov2_spatial_cuff_2338_v2.json
```

Require `num_scored_records == 1996`, `projection_dim == 128`, eight
`spatial_components`, a non-empty `projection_fingerprint`, and
`target_bbox_used_for_features == false`. Then run the same image-grouped OOF
diagnostic with the stronger linear/soft-target configuration:

```bash
PYTHONPATH=src python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_clip_dinov2_spatial_cuff_2338_v2.json \
  --regions cuff waist \
  --num-folds 5 \
  --num-epochs 200 \
  --selector-architecture linear \
  --selection-policy listwise \
  --listwise-loss soft_target \
  --threshold-policy fixed \
  --learning-rate 0.01 \
  --weight-decay 0.01 \
  --seed 42 \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_clip_dinov2_spatial_cuff_listwise_oof_2338_v2.json
```

Observed spatial result: Hit@0.3 is `0.5021` (1,174/2,338), cuff is `0.4624`,
and waist is `0.7339`. This is 57 hits above the previous listwise run but 229
below the 60% gate. Add online predicted-garment geometry to the same artifact.
This runs only the existing 3.1.1 segmentation checkpoint and does not rerun
GroundingDINO, OWLv2, Chinese-CLIP, or DINOv2:

```bash
cd /root/projects/alibaba-ai
git pull
PYTHONPATH=src python \
  scripts/eval/enrich_grounding_candidates_with_garment_geometry.py \
  --eval-json /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_clip_dinov2_spatial_cuff_2338_v2.json \
  --model-config configs/model/instance_segmentation_deepfashion2.yaml \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --regions cuff waist \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_clip_dinov2_spatial_cuff_garment_geometry_2338_v2.json \
  > /root/autodl-tmp/outputs/garment_geometry_enrichment.log 2>&1
```

Require `num_scored_records == 2338`,
`num_records_with_online_garment_instance == 2338`, and
`target_bbox_used_for_features == false`. Then run the controlled OOF
comparison, keeping every training setting unchanged:

```bash
PYTHONPATH=src python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_clip_dinov2_spatial_cuff_garment_geometry_2338_v2.json \
  --regions cuff waist \
  --num-folds 5 \
  --num-epochs 200 \
  --selector-architecture linear \
  --selection-policy listwise \
  --listwise-loss soft_target \
  --threshold-policy fixed \
  --learning-rate 0.01 \
  --weight-decay 0.01 \
  --seed 42 \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_clip_dinov2_spatial_cuff_garment_geometry_listwise_oof_2338_v2.json \
  > /root/autodl-tmp/outputs/garment_geometry_listwise_run.log 2>&1
```

Observed garment-geometry result: Hit@0.3 is `0.5141` (1,202/2,338), cuff is
`0.4714`, waist is `0.7632`, and Hit@0.5 is `0.1719`. This adds 28 hits over
the spatial-only run but remains 201 below the 60% gate.

The next v5 experiment reuses that exact enriched JSON and adds only
target-independent candidate-pool consensus and expert-interaction features.
It does not rerun segmentation, GroundingDINO, OWLv2, Chinese-CLIP, or DINOv2.
Keep the linear model and every training setting fixed:

```bash
PYTHONPATH=src python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_clip_dinov2_spatial_cuff_garment_geometry_2338_v2.json \
  --regions cuff waist \
  --num-folds 5 \
  --num-epochs 200 \
  --selector-architecture linear \
  --selection-policy listwise \
  --listwise-loss soft_target \
  --threshold-policy fixed \
  --learning-rate 0.01 \
  --weight-decay 0.01 \
  --seed 42 \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_candidate_consensus_v5_listwise_oof_2338_v2.json \
  > /root/autodl-tmp/outputs/candidate_consensus_v5_run.log 2>&1
```

Print only the comparison fields after the run:

```bash
python - <<'PY'
import json

path = "/root/autodl-tmp/outputs/local_region_candidate_consensus_v5_listwise_oof_2338_v2.json"
p = json.load(open(path))
oof = p["out_of_fold_summary"]
print(json.dumps({
    "feature_schema": p["candidate_feature_schema"],
    "garment_geometry_records": p["num_records_with_online_garment_geometry"],
    "baseline_hit": p["baseline_summary"]["manual_hit_at"],
    "oof_hit": oof["manual_hit_at"],
    "hit30_count": round(oof["manual_hit_at"]["0.3"] * oof["num_records"]),
    "oof_by_region": {
        region: values["manual_hit_at"]
        for region, values in oof["by_region"].items()
    },
    "transitions": p["selector_diagnostics"]["hit_transition_counts"],
}, ensure_ascii=False, indent=2))
PY
```

Observed v5 result: Hit@0.3 remains `0.5141` (1,202/2,338), cuff is `0.4709`,
waist is `0.7661`, and Hit@0.5 is `0.1728`. The transition balance changes to
477 gains and 122 losses but has no net improvement over v4. Do not tune more
candidate-consensus features. Run the cuff-side/pair diagnostic on the saved
v5 output instead; this is JSON-only analysis and performs no model inference:

```bash
PYTHONPATH=src python scripts/eval/analyze_cuff_pair_constraints.py \
  --eval-json /root/autodl-tmp/outputs/local_region_candidate_consensus_v5_listwise_oof_2338_v2.json \
  --hit-threshold 0.3 \
  --pair-max-iou 0.5 \
  --output /root/autodl-tmp/outputs/local_region_cuff_pair_constraint_diagnostic_v5.json
```

The output is intentionally short. Compare
`wrong_side_misses_recoverable_on_compatible_side` against
`selected_hits_on_incompatible_side`, then compare selected pair collisions and
both-hit pairs against `side_distinct_oracle_pairs_with_both_hits`. Do not add a
hard side or paired decoder until these counts show positive headroom.

Observed diagnostic: only 9 wrong-side misses are recoverable, while 19 current
hits violate the simple side rule; do not enable hard side filtering. There are
52 collisions among 809 complete pairs. Learned pair decoding has a larger
diagnostic ceiling: current complete pairs contain 766 record hits and the
side-compatible distinct-pair oracle contains 953.

Run the training-fold-only linear pair reranker on CUDA. This reuses every
frozen candidate feature and performs no additional model inference:

```bash
PYTHONPATH=src python scripts/eval/cross_validate_grounding_candidate_selector.py \
  --eval-json /root/autodl-tmp/outputs/local_region_train_online_candidates_cuff_waist_clip_dinov2_spatial_cuff_garment_geometry_2338_v2.json \
  --regions cuff waist \
  --num-folds 5 \
  --num-epochs 200 \
  --selector-architecture linear \
  --selection-policy listwise \
  --listwise-loss soft_target \
  --cuff-pair-decoding \
  --threshold-policy fixed \
  --learning-rate 0.01 \
  --weight-decay 0.01 \
  --seed 42 \
  --device cuda \
  --output /root/autodl-tmp/outputs/local_region_cuff_pair_reranker_v1_oof_2338_v2.json \
  > /root/autodl-tmp/outputs/cuff_pair_reranker_v1_run.log 2>&1
```

Print only the required comparison:

```bash
python - <<'PY'
import json

path = "/root/autodl-tmp/outputs/local_region_cuff_pair_reranker_v1_oof_2338_v2.json"
p = json.load(open(path))
oof = p["out_of_fold_summary"]
print(json.dumps({
    "pair_schema": p["cuff_pair_feature_schema"],
    "pair_decoded_records": p["num_pair_decoded_records"],
    "baseline_hit": p["baseline_summary"]["manual_hit_at"],
    "oof_hit": oof["manual_hit_at"],
    "hit30_count": round(oof["manual_hit_at"]["0.3"] * oof["num_records"]),
    "oof_by_region": {
        region: values["manual_hit_at"]
        for region, values in oof["by_region"].items()
    },
    "transitions": p["selector_diagnostics"]["hit_transition_counts"],
}, ensure_ascii=False, indent=2))
PY
```

The listwise output must report `num_records_with_dinov2_embeddings == 2338`,
`num_records_with_dinov2_spatial_embeddings == 1996`, and
`num_records_with_online_garment_geometry == 2338`. When train and
frozen-manual artifacts are later used together, the external selector rejects mismatched
DINOv2 model, enriched region set, feature mode, context, projection seed,
dimension, spatial components, or projection fingerprint. Legacy global
artifacts without `feature_mode` are treated as `global`.
The baseline contains 847 Hit@0.3 cases; the 60% target requires at least 1,403
and the candidate oracle contains about 1,595. The completed spatial result is
1,174 hits and online garment geometry reaches 1,202, leaving 201. With the
current waist result retained, cuff still needs roughly Hit@0.3 `0.572`. If v5
remains below 60%, inspect gained/lost hits and per-source selection before
changing model capacity. Do not open the manual benchmark or tune the MLP on
this weak split.
