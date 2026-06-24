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

AutoDL dataset and checkpoint paths are configured in `configs/paths.autodl.yaml`.
