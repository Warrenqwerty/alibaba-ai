# Alibaba AI Fashion Multimodal Project

## Project Goal

Fine-grained fashion visual understanding and multimodal reasoning system.

## Current Stage

Stage 1: Fine-grained visual foundation module.

Current target:

- 3.1.1 Fashion instance segmentation
- 3.1.2 Language-guided local-region localization
- Input: RGB fashion image
- Output: clothing instance masks, local-region masks, bounding boxes, category labels
- Classes: top, pants, skirt, outerwear, dress, shoes, bag, accessory
- Target: single-image latency <= 50 ms, mask IoU >= 0.85

## Repository Structure

- `configs/`: configuration files
- `src/`: reusable project source code
- `scripts/`: executable scripts
- `docs/`: setup notes and project documentation
- `docker/`: Docker environment files
- `tests/`: unit tests

## Environment

```bash
pip install -r requirements.txt
pip install -e .
```

## 3.1.1 Usage

Train on AutoDL:

```bash
python scripts/train/train_instance_segmentation.py \
  --model-config configs/model/instance_segmentation_deepfashion2.yaml \
  --paths-config configs/paths.autodl.yaml
```

Current best DeepFashion2 checkpoint:

```bash
/root/autodl-tmp/checkpoints/deepfashion2_6class_soft_aug_epoch2/instance_segmentation/epoch_001.pt
```

Full validation result on DeepFashion2 validation:

- Images: 32,153
- Ground-truth instances: 52,490
- Mean best mask IoU: 0.8547
- Recall@0.75: 0.8937
- Inference thresholds: score >= 0.3, mask >= 0.4

Run inference with a trained checkpoint:

```bash
python scripts/inference/predict_instance_segmentation.py image.jpg \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_soft_aug_epoch2/instance_segmentation/epoch_001.pt \
  --device cuda
```

## 3.1.2 Usage

Run language-guided local-region localization:

```bash
python scripts/inference/predict_local_region.py image.jpg "右侧的口袋" \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --output outputs/local_region_sample.json \
  --vis-output outputs/local_region_sample.jpg
```

Run a small AutoDL sanity evaluation:

```bash
python scripts/eval/evaluate_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --max-images 20 \
  --output /root/autodl-tmp/outputs/local_region_query_eval.json \
  --vis-dir /root/autodl-tmp/outputs/local_region_vis
```

Run weak-label evaluation with DeepFashion2 annotations:

```bash
python scripts/eval/evaluate_local_region_weak_labels.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/validation/annos \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --device cuda \
  --max-images 50 \
  --output /root/autodl-tmp/outputs/local_region_weak_eval.json
```

Add `--ranker-checkpoint /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry_500k.pt`
to evaluate the hybrid learned ranker on the same weak-label metric.

Build weak query-region records for a learned 3.1.2 ranker:

```bash
python scripts/data/build_deepfashion2_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/train/image \
  --anno-dir /root/autodl-tmp/datasets/DeepFashion2/train/annos \
  --output /root/autodl-tmp/outputs/local_region_train_queries.jsonl
```

Train the lightweight learned ranker:

```bash
python scripts/train/train_local_region_ranker.py \
  --records /root/autodl-tmp/outputs/local_region_train_queries.jsonl \
  --output /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry.pt \
  --device cuda \
  --max-records 50000 \
  --val-records 2000 \
  --num-epochs 1
```

Use `--val-offset` to evaluate on a later JSONL slice during larger runs.

Export candidate-level records for the next vision-language local-region ranker:

```bash
python scripts/data/build_local_region_candidate_records.py \
  --records /root/autodl-tmp/outputs/local_region_train_queries.jsonl \
  --output /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --max-records 500000
```

Install and evaluate Chinese-CLIP candidate reranking:

```bash
pip install "transformers>=4.37.0" sentencepiece
HF_ENDPOINT=https://hf-mirror.com \
python scripts/eval/evaluate_chinese_clip_local_region_ranker.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --model-name OFA-Sys/chinese-clip-vit-base-patch16 \
  --device cuda \
  --max-groups 2000 \
  --region-prior-weights 0,0.01,0.02,0.05,0.1,0.2 \
  --output /root/autodl-tmp/outputs/local_region_chinese_clip_eval_2k.json
```

Run candidate diagnostics:

```bash
python scripts/eval/evaluate_local_region_candidate_baselines.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --max-groups 2000 \
  --output /root/autodl-tmp/outputs/local_region_candidate_baselines_2k.json
```

Train a listwise candidate ranker from weak IoU labels:

```bash
python scripts/train/train_candidate_local_region_ranker.py \
  --candidates /root/autodl-tmp/outputs/local_region_train_candidates.jsonl \
  --output /root/autodl-tmp/checkpoints/local_region_ranker/candidate_listwise_50k.pt \
  --device cuda \
  --max-groups 50000 \
  --val-groups 2000 \
  --num-epochs 1
```

Use a hybrid learned ranker checkpoint during 3.1.2 evaluation:

```bash
python scripts/eval/evaluate_local_region_queries.py \
  --image-dir /root/autodl-tmp/datasets/DeepFashion2/validation/image \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_hard_mining/instance_segmentation/epoch_001.pt \
  --ranker-checkpoint /root/autodl-tmp/checkpoints/local_region_ranker/hash_text_geometry_500k.pt \
  --device cuda \
  --max-images 20 \
  --output /root/autodl-tmp/outputs/local_region_query_eval_learned.json
```
