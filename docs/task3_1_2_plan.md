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

### Important Data Observation

DeepFashion2 provides:

- garment instance masks
- garment bounding boxes
- garment category labels
- landmarks / keypoints

This is useful for weakly deriving local regions, but the current JSON category
metadata names landmarks only as numeric ids (`1` to `30`). It does not directly
name keypoints as "collar", "cuff", "hem", etc. Therefore the first baseline
should treat DeepFashion2 local regions as weak supervision derived from spatial
rules and landmarks, not as perfect semantic part labels.

The rule-based rectangles in the first demo are not suitable as training labels.
They are only sanity-check proposals. The real 3.1.2 training signal should come
from DeepFashion2 masks and landmarks, then be converted into weak local-region
pseudo labels.

FashionAI-Attributes is still needed for the broader attribute pipeline in
3.1.3, and may help define part-related query words such as collar and sleeve.
However, FashionAI attributes are image/attribute labels rather than pixel-level
local-region masks, so it is not enough by itself for 3.1.2 mask supervision.

### Final Baseline Direction

Use a PRD-aligned open-vocabulary grounding baseline:

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
   - First version: lightweight heuristic ranker that scores candidates from
     raw Chinese query text, spatial words, attribute words, relation words,
     and part words.
   - Target version: DINOv2 region features plus a text encoder, following the
     PRD direction of "区域特征与文本特征相似度匹配".
   - This should support open descriptions such as "左边的袖口", "碎花图案",
     "衣服上的拉链", and "外套里面的内搭" better than fixed-part
     classification.

This means the old fixed-region parser is no longer the main decision maker.
It is only a helper for candidate scoring.

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

Because there is no ready-made local-region ground truth yet, evaluation should
be staged:

1. Functional sanity test:
   - Given fixed images and queries, verify the pipeline returns non-empty masks
     within the selected garment instance.

2. Weak automatic evaluation:
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
     - Online policy: gate the listwise context ranker to hem only, and keep
       heuristic fallback for neckline, shoulder, and open-vocabulary regions.
       The 200-image hem-gated result is average weak IoU 0.2818, Hit@0.3
       0.4050, Hit@0.5 0.1333; by region: hem 0.2789, neckline 0.3000,
       shoulder 0.2665.
   - This confirms the weak metric is sensitive enough for heuristic iteration,
     but the remaining gap should be handled by learned text-region matching.

3. Human-labeled evaluation set:
   - Manually label a small set, e.g. 100-300 image-query pairs.
   - Include common queries: collar, cuff, hem, shoulder, waist, print.
   - Use this as the true metric set for localization accuracy.

4. Latency evaluation:
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

5. Decide whether to build a small human-labeled validation set before training
   any learned region localizer.

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
- The current heuristic ranker is only a prototype. It makes the interface
  open-vocabulary, but true PRD performance requires learned visual-text
  similarity using DINOv2/CLIP-style features or a stronger grounding model.
- Rule-based candidates can cover neckline/hem/waist, but pocket, pattern,
  decoration, zipper, button, and relation queries need visual feature matching.
- PRD latency target is tight; a heavy grounding model may exceed 30 ms unless
  cached or optimized.
- Query ambiguity is common. Example: "这件衣服的设计" is too broad and should
  ask for clarification or return the full garment region.

### Recommended Immediate Next Step

Implement the open-vocabulary baseline first:

- keep selected-garment instance handoff from 3.1.1
- generate multiple region candidates inside the garment
- rank candidates against raw natural-language query text
- visualize selected region plus top candidate scores
- then replace the heuristic ranker with DINOv2/text-feature similarity when
  model weights and runtime dependencies are available

This matches the PRD more closely than fixed-part segmentation, while keeping
the current code measurable and easy to debug.
