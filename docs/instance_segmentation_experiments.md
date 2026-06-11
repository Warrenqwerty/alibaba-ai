## 3.1.1 Instance Segmentation Experiments

This note records the main DeepFashion2 instance segmentation experiments for
the 3.1.1 fashion mask parsing task.

### Best Checkpoint

```bash
/root/autodl-tmp/checkpoints/deepfashion2_6class_soft_aug_epoch2/instance_segmentation/epoch_001.pt
```

### Best Config

- Model: Mask R-CNN ResNet-50 FPN
- Dataset: DeepFashion2 train and validation
- Foreground classes: top, pants, skirt, outerwear, dress
- Class-balanced sampling: enabled
- Learning rate: 0.00005
- Augmentation:
  - horizontal flip probability: 0.5
  - scale jitter: [0.95, 1.05]
  - brightness: 0.08
  - contrast: 0.08
  - saturation: 0.05
- Inference score threshold: 0.3
- Inference mask threshold: 0.4

### Full Validation Result

Validated on the full DeepFashion2 validation split.

```json
{
  "max_images": 32153,
  "gt_instances": 52490,
  "mean_best_mask_iou": 0.854691102142995,
  "recall_iou_50": 0.9616307868165365,
  "recall_iou_75": 0.8937321394551343,
  "avg_predictions_per_image": 2.875532609709825,
  "avg_prediction_score": 0.7670541155067138
}
```

Per-class mean best mask IoU:

| Class | GT instances | Mean best mask IoU | Recall@0.75 |
| --- | ---: | ---: | ---: |
| top | 20,957 | 0.8652 | 0.9016 |
| pants | 13,753 | 0.8426 | 0.8754 |
| skirt | 6,522 | 0.8588 | 0.9066 |
| outerwear | 2,153 | 0.8381 | 0.8867 |
| dress | 9,105 | 0.8498 | 0.8958 |

The overall mean best mask IoU reaches 0.8547, satisfying the PRD target of
segmentation IoU >= 0.85.

### Validation Command

```bash
python scripts/eval/validate_instance_segmentation.py \
  --model-config configs/model/instance_segmentation_deepfashion2.yaml \
  --paths-config configs/paths.autodl.yaml \
  --checkpoint /root/autodl-tmp/checkpoints/deepfashion2_6class_soft_aug_epoch2/instance_segmentation/epoch_001.pt \
  --device cuda \
  --max-images 100000 \
  --score-threshold 0.3 \
  --mask-threshold 0.4 \
  --output outputs/validation_deepfashion2_6class_soft_aug_epoch2_full.json \
  --vis-dir outputs/validation_deepfashion2_6class_soft_aug_epoch2_full_vis \
  --vis-count 30
```

### Experiment Notes

- The stronger augmentation setting improved some rare classes but reduced the
  overall score.
- Softer augmentation with a lower learning rate produced the best result.
- Continuing one more epoch after the best checkpoint slightly reduced overall
  mean IoU, so the epoch2 soft-augmentation checkpoint is kept as the current
  best model.
