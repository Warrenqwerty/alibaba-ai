# Alibaba AI Fashion Multimodal Project

## Project Goal

Fine-grained fashion visual understanding and multimodal reasoning system.

## Current Stage

Stage 1: Fine-grained visual foundation module.

Current target:

- 3.1.1 Fashion instance segmentation
- Input: RGB fashion image
- Output: clothing instance masks, bounding boxes, category labels
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
