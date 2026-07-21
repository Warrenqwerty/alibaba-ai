# Task 3.1.3 Plan: Fine-Grained Attribute Extraction

## PRD Contract

- Input: RGB product image plus a target-region mask.
- Output: fine-grained attribute labels and confidence scores.
- Scope: 14 broad groups and 200+ values in the final product taxonomy.
- Target: attribute accuracy at least 88% and extraction latency at most 20 ms.

The PRD target is broader than any schema assumed by this repository. The
downloaded FashionAI annotation files and their README are therefore the source
of truth for the first training schema. Code must not claim 14 groups or 200+
semantic values when those names are not present in the local data.

## Baseline Architecture

The working baseline has four explicit boundaries:

1. Parse FashionAI `image_path, AttrKey, AttrValues` rows and infer one head per
   `AttrKey` from the y/m/n vector length.
2. Train one lightweight shared MobileNetV3-Small image encoder with a separate
   softmax head for each attribute dimension.
3. At inference, apply the supplied mask, fill pixels outside the mask, crop a
   padded target bbox, and classify the crop.
4. Compose the predictor after the frozen 3.1.1 and 3.1.2 modules for a single
   JSON result.

Exactly one `y` is the strict class. `m` classes remain ambiguity-aware
alternatives for evaluation; they are not silently converted into hard labels.
All random splits are grouped by image path.

## Implemented Interfaces

- `FashionAIAttributeSchema`: dynamic heads and optional semantic value names.
- `FashionAIAttributeDataset`: heterogeneous attribute records with grouped
  collation.
- `FashionAttributeClassifier`: MobileNetV3/ResNet/tiny smoke backbone plus
  dynamic heads.
- `FashionAttributePredictor`: checkpoint loading, masked crop, confidence, and
  alternatives.
- `FashionVisualPipeline`: 3.1.1 -> frozen 3.1.2 -> 3.1.3 composition.
- Data inspection, CUDA training, standalone inference, and full-pipeline CLIs.

## Validation Order

### Gate A: Dataset integrity

- Discover the actual CSV files under the AutoDL FashionAI root.
- Validate image paths and annotation vectors.
- Compare inferred heads/value counts with the dataset README.
- Add a semantic label map only from the README or another authoritative file.

### Gate B: Working CUDA baseline

- Run a small `--max-records` training smoke test on CUDA.
- Verify `best.pt` can load without downloading another backbone.
- Run standalone mask inference and the complete visual pipeline on one image.
- Confirm JSON, mask, visualization, and latency fields are present.

### Gate C: Model quality

- Train on the complete train split with image-grouped validation.
- Report strict top-1, ambiguity-aware top-1, per-head accuracy, and latency.
- Do not collapse all heads into one overall number when class balance differs.
- Keep the official FashionAI evaluation convention separate from the PRD's
  simple accuracy target.

### Gate D: Region-domain alignment

FashionAI supervision is image-level, but 3.1.3 inference is mask-conditioned.
If the full-image baseline is weak on local crops, compare these controlled
options:

- region-aware crops derived from attribute dimension;
- 3.1.1 predicted garment masks generated offline for FashionAI images;
- a crop/full-image dual-view encoder;
- fine-tuning on a small manually checked masked-region set.

Only one change should be introduced per comparison, and the validation images
must stay fixed.

## Current Definition of Done

The first milestone is complete when a real AutoDL command produces:

1. a validated FashionAI schema report;
2. a trained `best.pt` checkpoint;
3. standalone image+mask attribute JSON;
4. complete 3.1.1 -> 3.1.2 -> 3.1.3 JSON plus saved region mask;
5. measured accuracy and CUDA latency without silent CPU fallback.

The 88% PRD metric is a later quality gate, not a prerequisite for calling the
software path operational.
