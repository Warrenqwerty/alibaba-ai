## 3.1.1 Fashion Instance Segmentation

Source requirements were summarized from the three project PDFs provided for the internship task.

### PRD Scope

- Input: RGB product image, arbitrary size.
- Output: one mask, bounding box, and category label per clothing instance.
- Supported classes: top, pants, skirt, outerwear, dress, shoes, bag, accessory.
- Performance target: single-image segmentation latency <= 50 ms, segmentation IoU >= 0.85.

### Current Implementation

- Baseline model: TorchVision Mask R-CNN with ResNet-50 FPN heads resized for 8 clothing classes plus background.
- Dataset adapter: `fashion_mm.data_loaders.DeepFashion2Dataset` reads DeepFashion2 image/annotation folders and maps the original clothing categories into the project taxonomy.
- Training entry point: `scripts/train/train_instance_segmentation.py`.
- Inference entry point: `scripts/inference/predict_instance_segmentation.py`.
- Result object: `SegmentationResult` serializes masks, boxes, labels, scores, and inference time.

### AutoDL Notes

- Training is intended for the AutoDL 5090 server, not the Mac.
- Paths are centralized in `configs/paths.yaml`.
- Install the package in editable mode after dependencies: `pip install -e .`.
- Use `configs/model/instance_segmentation.yaml` to tune batch size, workers, thresholds, epochs, and checkpoint behavior.

### Code Rules Remembered

- Keep modules separated by responsibility.
- Keep parameters in YAML configs, not hard-coded inside scripts.
- Use exceptions around missing files and invalid inputs.
- Use docstrings and type hints for public functions/classes.
- Avoid duplicated code and keep PEP8-style naming.
