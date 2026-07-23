from __future__ import annotations

import time
from pathlib import Path

import torch

from fashion_mm.data_loaders.fashionai_attributes import build_fashionai_transform
from fashion_mm.data_loaders.fashionai_attributes import FashionAIAttributeSchema
from fashion_mm.models.attributes.model import FashionAttributeClassifier
from fashion_mm.models.attributes.preprocessing import MaskInput
from fashion_mm.models.attributes.preprocessing import prepare_masked_region
from fashion_mm.models.attributes.result import AttributeExtractionResult
from fashion_mm.models.attributes.result import AttributeValuePrediction
from fashion_mm.models.attributes.result import FineGrainedAttributePrediction
from fashion_mm.utils.image_io import ImageInput
from fashion_mm.utils.logger import get_logger


LOGGER = get_logger(__name__)


class FashionAttributePredictor:
    """Checkpoint-backed 3.1.3 attribute extractor for one target mask."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        confidence_threshold: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Attribute checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for 3.1.3 but is unavailable. "
                "Pass device='cpu' only for an explicit local smoke test."
            )

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.schema = FashionAIAttributeSchema.from_dict(checkpoint["schema"])
        model_config = checkpoint.get("model_config", {})
        self.image_size = int(model_config.get("image_size", 224))
        self.input_mode = str(model_config.get("input_mode", "crop"))
        self.padding_fraction = float(model_config.get("mask_padding_fraction", 0.08))
        self.confidence_threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else model_config.get("confidence_threshold", 0.0)
        )
        self.top_k = int(top_k if top_k is not None else model_config.get("top_k", 3))
        self.model = FashionAttributeClassifier(
            self.schema,
            backbone_name=str(model_config.get("backbone", "mobilenet_v3_small")),
            pretrained=False,
            dropout=float(model_config.get("dropout", 0.2)),
            pooling=str(model_config.get("pooling", "global")),
            attention_reduction=int(
                model_config.get("attention_reduction", 16)
            ),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.transform = build_fashionai_transform(
            self.image_size,
            train=False,
            input_mode=self.input_mode,
        )
        self.backend = f"fashionai_multi_head_{self.model.backbone_name}"
        if self.model.pooling != "global":
            self.backend = f"{self.backend}_{self.model.pooling}"
        LOGGER.info("Loaded 3.1.3 attribute checkpoint: %s", self.checkpoint_path)

    @torch.inference_mode()
    def predict(
        self,
        image: ImageInput,
        mask: MaskInput,
        *,
        attributes: list[str] | tuple[str, ...] | None = None,
    ) -> AttributeExtractionResult:
        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        rgb_image, region = prepare_masked_region(
            image,
            mask,
            padding_fraction=self.padding_fraction,
        )
        tensor = self.transform(region.image).unsqueeze(0).to(self.device)
        preprocessing_time_ms = (time.perf_counter() - preprocess_start) * 1000.0

        requested_attributes = tuple(attributes or self.schema.attribute_names)
        unknown = set(requested_attributes) - set(self.schema.attribute_names)
        if unknown:
            raise ValueError(f"Unknown requested attributes: {sorted(unknown)}")

        inference_start = time.perf_counter()
        features = self.model.encode(tensor)
        logits_by_attribute = {
            name: self.model.classify(features, name) for name in requested_attributes
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

        predictions = tuple(
            self._decode_prediction(name, logits_by_attribute[name][0])
            for name in requested_attributes
        )
        return AttributeExtractionResult(
            image_size=rgb_image.size,
            region_box=region.box,
            mask_area=region.mask_area,
            mask_coverage=region.mask_coverage,
            predictions=predictions,
            preprocessing_time_ms=preprocessing_time_ms,
            inference_time_ms=inference_time_ms,
            total_time_ms=(time.perf_counter() - total_start) * 1000.0,
            backend=self.backend,
        )

    def _decode_prediction(
        self,
        attribute_name: str,
        logits: torch.Tensor,
    ) -> FineGrainedAttributePrediction:
        definition = self.schema.definition(attribute_name)
        probabilities = torch.softmax(logits, dim=-1)
        count = min(max(self.top_k, 1), definition.num_classes)
        values, indices = torch.topk(probabilities, k=count)
        decoded = tuple(
            AttributeValuePrediction(
                label_index=int(index),
                label=definition.value_names[int(index)],
                confidence=float(value),
            )
            for value, index in zip(values.detach().cpu(), indices.detach().cpu())
        )
        return FineGrainedAttributePrediction(
            attribute_name=attribute_name,
            value=decoded[0],
            alternatives=decoded[1:],
            is_confident=decoded[0].confidence >= self.confidence_threshold,
        )
