# Task 3.1.3 Plan: Fine-Grained Attribute Extraction

## PRD Contract

- Input: RGB product image plus a target-region mask.
- Output: fine-grained attribute labels and confidence scores.
- Scope: 14 broad groups and 200+ values in the final product taxonomy.
- Target: attribute accuracy at least 88% and extraction latency at most 20 ms.

The PRD target is broader than the available Round1 FashionAI files. After
review with the mentor, the working dataset is limited to the labeled
`round1_fashionAI_attributes_test_a` and `test_b` releases. Their README and
answer CSVs define 8 attribute groups and 54 values. Results from this baseline
must not be described as a 14-group/200-value benchmark.

## Round1 Dataset Policy

- Test A contains 10,080 labeled records and test B contains 15,042.
- The two answer files share 5,206 relative image paths. All overlapping labels
  were verified to match, so duplicates are removed before any split.
- The resulting corpus has 19,916 unique human-labeled image/attribute records.
- B's apparent 30,084 JPEG count includes 15,042 macOS `._*.jpg` sidecars. Only
  paths referenced by the answer CSV are loaded.
- The unique corpus is split 80/10/10 into train, validation, and held-out test.
  Sampling is deterministic and stratified by `(attribute group, strict y
  class)` with seed 42.
- Image identity is based on the original relative path, not the extracted
  absolute path. This keeps A/B copies together and guarantees zero image
  overlap between splits.
- Validation may guide model selection. The test split is used only for final
  reporting with `evaluate_fashionai_attributes.py`.

## Baseline Architecture

The working baseline has four explicit boundaries:

1. Merge and deduplicate the two Round1 answer files, then parse FashionAI
   `image_path, AttrKey, AttrValues` rows and infer one head per `AttrKey` from
   the y/m/n vector length.
2. Train one lightweight shared MobileNetV3-Small image encoder with a separate
   softmax head for each attribute dimension.
3. At inference, apply the supplied mask, fill pixels outside the mask, crop a
   padded target bbox, and classify the crop.
4. Compose the predictor after the frozen 3.1.1 and 3.1.2 modules for a single
   JSON result.

Exactly one `y` is the strict class. `m` classes remain ambiguity-aware
alternatives for evaluation; they are not silently converted into hard labels.
All random splits are grouped by stable image ID and stratified by attribute
group plus strict class.

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
- Round1 A/B deduplication, stratified split manifests, and held-out CUDA test
  evaluation.

## Validation Order

### Gate A: Dataset integrity (complete)

- Validate all A/B answer paths and annotation vectors.
- Reject conflicting labels before deduplication.
- Generate leak-free stratified train/validation/test manifests.
- Verify all split-pair overlap counts are zero.
- Use the 8-group/54-value semantic label map transcribed from the dataset
  README.

### Gate B: Working CUDA baseline

- Run a small `--max-records` training smoke test on CUDA.
- Verify `best.pt` can load without downloading another backbone.
- Run standalone mask inference and the complete visual pipeline on one image.
- Confirm JSON, mask, visualization, and latency fields are present.

### Gate C: Model quality

- Train only on the generated train split with stratified validation.
- Report strict top-1, ambiguity-aware top-1, per-head accuracy, and latency.
- Report final quality once on the untouched test split.
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

1. a validated Round1 split summary with zero train/validation/test overlap;
2. a trained `best.pt` checkpoint;
3. standalone image+mask attribute JSON;
4. complete 3.1.1 -> 3.1.2 -> 3.1.3 JSON plus saved region mask;
5. held-out test accuracy and CUDA latency without silent CPU fallback.

The 88% PRD metric is a later quality gate, not a prerequisite for calling the
software path operational.
