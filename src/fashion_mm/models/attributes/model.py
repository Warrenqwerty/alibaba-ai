from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import mobilenet_v3_small
from torchvision.models import resnet18
from torchvision.models import resnet50

from fashion_mm.data_loaders.fashionai_attributes import FashionAIAttributeSchema


class FashionAttributeClassifier(nn.Module):
    """Shared visual backbone with one classification head per attribute."""

    def __init__(
        self,
        schema: FashionAIAttributeSchema,
        *,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.backbone_name = backbone_name
        self.backbone, feature_dim = _build_backbone(backbone_name, pretrained)
        self.attribute_to_head = {
            definition.name: f"head_{index:03d}"
            for index, definition in enumerate(schema.definitions)
        }
        self.heads = nn.ModuleDict(
            {
                self.attribute_to_head[definition.name]: nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, definition.num_classes),
                )
                for definition in schema.definitions
            }
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def classify(
        self,
        features: torch.Tensor,
        attribute_name: str,
    ) -> torch.Tensor:
        try:
            head_key = self.attribute_to_head[attribute_name]
        except KeyError as error:
            raise KeyError(f"Unknown attribute head: {attribute_name}") from error
        return self.heads[head_key](features)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(images)
        return {
            attribute_name: self.classify(features, attribute_name)
            for attribute_name in self.schema.attribute_names
        }


def _build_backbone(
    backbone_name: str,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    if backbone_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        feature_dim = int(backbone.classifier[-1].in_features)
        backbone.classifier[-1] = nn.Identity()
        return backbone, feature_dim

    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    if backbone_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    if backbone_name == "tiny_cnn":
        feature_dim = 32
        backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        return backbone, feature_dim

    raise ValueError(f"Unsupported attribute backbone: {backbone_name}")
