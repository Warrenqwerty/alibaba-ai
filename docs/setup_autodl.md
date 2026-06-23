## AutoDL 5090 Setup

This project is designed so Mac-side work can focus on coding and tests, while GPU
training runs on AutoDL.

```bash
cd /root/alibaba-ai
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
cd /root/alibaba-ai
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

AutoDL dataset and checkpoint paths are configured in `configs/paths.autodl.yaml`.
