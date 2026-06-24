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
Review the JSON summary and visualization directory before moving to learned
DINOv2/text-region similarity.

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
the learned text-region matching baseline.

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
a trainable checkpoint and top-1 weak IoU metric before adding heavier
vision-language dependencies.

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

The learned checkpoint is used only for regions where the weak evaluation is
neutral or helpful (`neckline`, `hem`). Shoulder and other open-vocabulary
queries, such as cuff, pocket, zipper, and pattern, fall back to the heuristic
ranker to preserve coverage.

20-image hybrid sanity result: 140/140 ok, average local-region latency
`16.93 ms`, and open-query outputs remain diverse instead of collapsing to the
whole garment.

200-image neckline/hem-only hybrid weak-label result: average weak IoU `0.2822`.
This recovers the tuned heuristic baseline, but the gain is too small to treat
the hash text-geometry scorer as the final model.

Export candidate-level records for the next CLIP/OpenCLIP or DINOv2
text-region ranker:

```bash
python scripts/data/build_local_region_candidate_records.py \
  --records /root/autodl-tmp/outputs/local_region_train_queries.jsonl \
  --output /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --max-records 500000
```

Each input query record is expanded into candidate boxes with IoU labels against
the weak region box. This keeps image paths and candidate boxes together, so the
next training script can crop candidate regions and learn image-text matching
instead of relying only on geometry.

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
  --output /root/autodl-tmp/outputs/local_region_chinese_clip_eval_2k.json
```

This baseline uses the Chinese query directly, crops each candidate box, and
ranks candidates by Chinese-CLIP image-text cosine similarity. It is a stronger
fit than OpenCLIP here because the 3.1.2 queries are Chinese.

If the mirror is unavailable, download `OFA-Sys/chinese-clip-vit-base-patch16`
to an AutoDL-local directory and pass that directory with `--model-name`.

AutoDL dataset and checkpoint paths are configured in `configs/paths.autodl.yaml`.
