from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from fashion_mm.models.instance_segmentation.result import (
    FashionInstance,
    SegmentationResult,
)
from fashion_mm.utils.image_io import ImageInput, load_rgb_image
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


def build_mask_rcnn(num_classes: int, pretrained: bool = True) -> MaskRCNN:
    """Build a Mask R-CNN model with project-specific class heads."""
    weights = "DEFAULT" if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    mask_predictor = model.roi_heads.mask_predictor
    in_channels = mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels,
        hidden_layer,
        num_classes,
    )
    return model


class FashionInstanceSegmentationPredictor:
    """Predict clothing instance masks, boxes, and labels for RGB product images."""

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.categories = {
            int(label_id): name for label_id, name in config["categories"].items()
        }
        self.score_threshold = float(config["inference"]["score_threshold"])
        self.mask_threshold = float(config["inference"]["mask_threshold"])
        self.device = torch.device(device or config["inference"].get("device", "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")

        self.model = self._build_model()
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> MaskRCNN:
        model_config = self.config["model"]
        model_name = model_config.get("name", "mask_rcnn_resnet50_fpn")
        if model_name not in {"mask_rcnn_baseline", "mask_rcnn_resnet50_fpn"}:
            raise ValueError(f"Unsupported instance segmentation model: {model_name}")

        return build_mask_rcnn(
            num_classes=int(model_config["num_classes"]),
            pretrained=bool(model_config.get("pretrained", True)),
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model weights from an AutoDL training checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        LOGGER.info("Loaded instance segmentation checkpoint: %s", path)

    @torch.inference_mode()
    def predict(self, image: ImageInput) -> SegmentationResult:
        """Run instance segmentation for one RGB image."""
        pil_image = load_rgb_image(image)
        image_tensor = F.to_tensor(pil_image).to(self.device)

        start = time.perf_counter()
        prediction = self.model([image_tensor])[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        instances = self._postprocess_prediction(prediction)
        return SegmentationResult(
            image_size=pil_image.size,
            instances=instances,
            inference_time_ms=elapsed_ms,
        )

    def _postprocess_prediction(
        self,
        prediction: dict[str, torch.Tensor],
    ) -> list[FashionInstance]:
        instances: list[FashionInstance] = []
        scores = prediction["scores"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        boxes = prediction["boxes"].detach().cpu()
        masks = prediction["masks"].detach().cpu()

        for score, label_id, box, mask in zip(scores, labels, boxes, masks):
            score_value = float(score.item())
            if score_value < self.score_threshold:
                continue

            label_int = int(label_id.item())
            binary_mask = np.asarray(mask[0].numpy() >= self.mask_threshold)
            instances.append(
                FashionInstance(
                    mask=binary_mask,
                    box=tuple(float(value) for value in box.tolist()),
                    label_id=label_int,
                    label_name=self.categories.get(label_int, "unknown"),
                    score=score_value,
                )
            )
        return instances
