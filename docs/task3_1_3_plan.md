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
- Inspect `split_summary.json["stratification_audit"]` to confirm every
  `(attribute group, strict y class)` stratum has explicit train/validation/test
  counts and bounded fraction error.
- Use the 8-group/54-value semantic label map transcribed from the dataset
  README.

### Gate B: Working CUDA baseline

- Run a small `--max-records` training smoke test on CUDA.
- Verify `best.pt` can load without downloading another backbone.
- Run standalone mask inference and the complete visual pipeline on one image.
- Confirm JSON, mask, visualization, and latency fields are present.
- Benchmark one resident predictor after warmup and report wall-time p95, not
  the first-call CUDA startup time.

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

## 2026-07-22 AutoDL Baseline

- Training used 15,930 stratified records and validation used 1,993 records.
- Validation strict accuracy peaked at `0.6086` on epoch 8.
- The untouched 1,993-record test split reached strict accuracy `0.6071` and
  ambiguity-aware accuracy `0.6147`.
- Per-head strict accuracy ranged from `0.4965` for `neck_design_labels` to
  `0.6943` for `pant_length_labels`.
- The standalone image-plus-mask result matched the integrated
  3.1.1 -> 3.1.2 -> 3.1.3 result exactly for the requested heads.
- On RTX 5090, 10 warmup runs plus 30 measured runs produced wall-time p95
  `15.578 ms`, max `19.098 ms`, and model-only mean `2.439 ms` for all 8 heads.
- The baseline satisfies the 20 ms steady-state extraction target but does not
  satisfy the later 88% quality target. Future tuning must use the fixed
  validation split rather than repeatedly consulting the test split.

## Controlled Accuracy Experiments

The training class distributions are not severely skewed: the largest class
within each head represents only `0.1492` to `0.2824` of its records. The more
immediate issue is the `0.2112` train/validation gap and an augmentation policy
that can crop away the exact garment regions needed for neckline, sleeve, and
length labels.

The first validation-only ablation used
`configs/model/fashionai_attributes_full_frame.yaml`. It replaced crop-based
geometry with centered white padding plus square resize while retaining every
other setting. It peaked at epoch 5 with strict accuracy `0.6106`, a `0.0020`
gain. The per-head result was mixed: skirt length gained `0.0517`, while sleeve
length lost `0.0472` and lapel design lost `0.0287`. Keep this checkpoint as
evidence that global length heads benefit from full-frame context, but do not
promote it as the default model.

The next experiment uses
`configs/model/fashionai_attributes_low_backbone_lr.yaml`. It restores the
baseline crop input and changes only the pretrained backbone learning rate from
`3e-4` to `3e-5`; attribute heads remain at `3e-4`. The legacy config continues
to build one optimizer parameter group, while this experiment uses explicit
backbone and head groups. It reached only `0.5610` validation strict accuracy
at epoch 10, `0.0476` below baseline, and is rejected as underfit within the
fixed training budget.

The third experiment uses `configs/model/fashionai_attributes_cosine.yaml`.
It restores the baseline's shared `3e-4` learning rate and changes only its
epoch-level schedule, applying cosine decay to `3e-6` over 10 epochs. This
tests whether fast early transfer plus a gentler late update avoids the fixed
rate model's post-epoch-8 validation decline. Scheduler state is checkpointed
and restored explicitly. Compare its selected checkpoint against `0.6086`
using validation only, and keep the test split closed.

The cosine model peaked at epoch 9 with validation strict accuracy `0.6101`.
Selecting full-frame for coat/neck/pant/skirt, cosine for
collar/lapel/neckline, and crop for sleeve yields a per-head validation oracle
of `0.6287`. This is useful evidence that heads prefer different visual
context, but a three-checkpoint router is deferred because it adds model memory
and repeated encodings while remaining far below the quality target.

The fourth experiment uses `configs/model/fashionai_attributes_resnet18.yaml`.
It changes only the backbone from MobileNetV3-small to ResNet-18 and retains the
baseline crop, fixed `3e-4` learning rate, seed, and 10-epoch budget. Select it
on validation first; only a winning checkpoint proceeds to a separate
steady-state latency benchmark.

ResNet-18 peaked at epoch 6 with validation strict accuracy `0.6884` and
ambiguity-aware accuracy `0.6974`, a strict gain of `0.0798` over the
MobileNetV3-small baseline. Every head improved:

| Attribute head | MobileNetV3 | ResNet-18 | Delta |
| --- | ---: | ---: | ---: |
| coat length | 0.6760 | 0.7003 | +0.0244 |
| collar design | 0.5619 | 0.6381 | +0.0762 |
| lapel design | 0.5862 | 0.6839 | +0.0977 |
| neck design | 0.5286 | 0.6143 | +0.0857 |
| neckline design | 0.5418 | 0.6683 | +0.1265 |
| pant length | 0.6927 | 0.7135 | +0.0208 |
| skirt length | 0.6681 | 0.6983 | +0.0302 |
| sleeve length | 0.6195 | 0.7463 | +0.1268 |

The formal resident image-plus-mask benchmark measured wall-time p95
`15.043 ms`, maximum `16.705 ms`, and model-only mean `1.584 ms` on RTX 5090.
Both p95 and observed maximum pass the 20 ms gate, making ResNet-18 the current
eligible validation champion. The held-out test split remains closed, and the
model does not yet satisfy the later 88% quality target.

The fifth experiment uses
`configs/model/fashionai_attributes_resnet18_cosine.yaml`. It retains the
winning ResNet-18 architecture and every data, preprocessing, optimizer, seed,
and budget setting, adding only the previously verified cosine schedule from
`3e-4` to `3e-6`. Select it against the fixed-rate ResNet-18 result of `0.6884`
using validation only.

ResNet-18 with cosine decay peaked at epoch 9 with validation strict accuracy
`0.7105` and ambiguity-aware accuracy `0.7220`. It improved every head over the
fixed-rate checkpoint, led by neck design (`+0.0571`) and skirt length
(`+0.0474`). Its train/validation strict gap is `0.2711`, so overfitting remains
an explicit limitation. The formal RTX 5090 benchmark measured wall-time p95
`13.960 ms`, maximum `19.627 ms`, and model-only mean `1.512 ms`; both latency
checks pass.

The sixth experiment uses
`configs/model/fashionai_attributes_resnet50_cosine.yaml`. It changes only the
backbone from ResNet-18 to ResNet-50 while retaining the winning cosine
schedule and all data, preprocessing, optimizer, seed, and budget settings.
The extra capacity is eligible only if validation strict accuracy exceeds
`0.7105` and the selected checkpoint independently passes the 20 ms resident
image-plus-mask latency gate. Keep `test.csv` closed during this comparison.

ResNet-50 cosine peaked at the final epoch 10 with validation strict accuracy
`0.7802` and ambiguity-aware accuracy `0.7873`, a strict gain of `0.0697` over
ResNet-18 cosine. All eight heads improved; the largest gains were lapel design
(`+0.1207`) and collar design (`+0.1143`). Its train/validation gap was
`0.2133`. The formal RTX 5090 benchmark measured wall-time p95 `14.344 ms`,
maximum `16.757 ms`, and model-only mean `2.798 ms`, so both latency checks
pass.

The seventh and final model-selection experiment uses
`configs/model/fashionai_attributes_resnet50_cosine_15ep.yaml`. It changes only
`num_epochs` from 10 to 15; because the scheduler derives its cosine horizon
from that value, this must be a fresh run rather than a resume from the
10-epoch scheduler state. Compare against `0.7802` using validation only. After
this ablation, freeze model selection and evaluate only the selected checkpoint
on `test.csv`.
