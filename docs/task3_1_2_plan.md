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

### Proposed Baseline Direction

Use a three-stage baseline:

1. Whole-garment grounding from 3.1.1:
   - Run the frozen instance segmentation model.
   - Select the relevant garment instance using category words in the query
     when possible. Example: "这条裙子的下摆" should prefer `skirt`.
   - If no garment class is mentioned, use the highest-confidence clothing
     instance or all candidate instances.

2. Query-to-region parsing:
   - Map Chinese query keywords to canonical region names.
   - Examples:
     - "领口", "衣领", "领型" -> `neckline`
     - "袖口", "袖子末端" -> `cuff`
     - "下摆", "裙摆" -> `hem`
     - "口袋" -> `pocket`
     - "肩部", "肩线" -> `shoulder`
     - "腰部", "腰线" -> `waist`
     - "图案", "印花", "花纹" -> `pattern`
     - "装饰", "纽扣", "拉链", "珠片" -> `decoration`

3. Region proposal inside garment mask:
   - First version: deterministic geometric proposals clipped by the garment
     mask.
   - Later version: learned text-region matching using DINOv2/CLIP-style
     features or a phrase grounding model.

### First Baseline Region Rules

The goal of the first baseline is to create a measurable, debuggable pipeline
before training a heavier language-grounded model.

For one selected garment mask and bounding box:

| Region | Initial proposal rule |
| --- | --- |
| neckline | upper 20-30% of garment mask, centered around top-middle |
| cuff | left/right side regions around sleeve ends, when garment is top/dress/outerwear |
| hem | lower 20-25% of garment mask |
| shoulder | upper-left and upper-right mask regions |
| waist | horizontal band around 45-60% height of garment box |
| pocket | no reliable deterministic rule; mark as unsupported or use model-based proposal |
| pattern | visible full garment mask by default, later refined by texture/CLIP matching |
| decoration | unsupported in rule baseline unless detected by later attribute/local feature model |

For unsupported or ambiguous regions, return a structured fallback:

```json
{
  "status": "unsupported_region",
  "reason": "pocket/decoration requires learned local detector or extra labels"
}
```

This is better than returning misleading masks.

### Suggested Output Schema

```json
{
  "image": "000001.jpg",
  "query": "这件衣服的领口",
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
- Rule-based geometry can work for neckline/hem/waist but is weak for pocket,
  pattern, and decoration.
- PRD latency target is tight; a heavy grounding model may exceed 30 ms unless
  cached or optimized.
- Query ambiguity is common. Example: "这件衣服的设计" is too broad and should
  ask for clarification or return the full garment region.

### Recommended Immediate Next Step

Implement the rule-based 3.1.2 baseline first:

- query parser
- selected-garment instance handoff from 3.1.1
- geometry-based local region proposal
- visualization script

This creates a working pipeline and makes failure cases visible. After that, we
can decide whether to train a learned local-region model or move directly toward
3.1.3 attribute extraction for supported regions.
