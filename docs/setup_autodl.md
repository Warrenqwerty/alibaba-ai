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
