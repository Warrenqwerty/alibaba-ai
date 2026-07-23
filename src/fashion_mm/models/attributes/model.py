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


FASHION_ATTRIBUTE_POOLING_MODES = frozenset({"global", "attribute_attention"})


class AttributeAttentionHead(nn.Module):
    """Attribute-specific spatial/channel attention followed by classification."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        *,
        dropout: float,
        reduction: int,
    ) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("attention reduction must be positive.")
        hidden_dim = max(feature_dim // reduction, 32)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Conv2d(
            feature_dim,
            1,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.zeros_(self.channel_attention[-2].weight)
        nn.init.zeros_(self.channel_attention[-2].bias)
        nn.init.zeros_(self.spatial_attention.weight)
        nn.init.zeros_(self.spatial_attention.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "Attribute attention requires BCHW spatial feature maps."
            )
        channel_weights = 2.0 * self.channel_attention(features)
        attended = features * channel_weights
        spatial_logits = self.spatial_attention(attended).flatten(2)
        spatial_weights = torch.softmax(spatial_logits, dim=-1)
        pooled = (attended.flatten(2) * spatial_weights).sum(dim=-1)
        return self.classifier(self.dropout(pooled))


class FashionAttributeClassifier(nn.Module):
    """Shared visual backbone with one classification head per attribute."""

    def __init__(
        self,
        schema: FashionAIAttributeSchema,
        *,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2,
        pooling: str = "global",
        attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        if pooling not in FASHION_ATTRIBUTE_POOLING_MODES:
            raise ValueError(
                f"Unsupported attribute pooling {pooling!r}; expected one of "
                f"{sorted(FASHION_ATTRIBUTE_POOLING_MODES)}."
            )
        self.schema = schema
        self.backbone_name = backbone_name
        self.pooling = pooling
        self.attention_reduction = attention_reduction
        self.backbone, feature_dim = _build_backbone(
            backbone_name,
            pretrained,
            spatial_features=pooling == "attribute_attention",
        )
        self.attribute_to_head = {
            definition.name: f"head_{index:03d}"
            for index, definition in enumerate(schema.definitions)
        }
        self.heads = nn.ModuleDict(
            {
                self.attribute_to_head[definition.name]: (
                    AttributeAttentionHead(
                        feature_dim,
                        definition.num_classes,
                        dropout=dropout,
                        reduction=attention_reduction,
                    )
                    if pooling == "attribute_attention"
                    else nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(feature_dim, definition.num_classes),
                    )
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
    *,
    spatial_features: bool,
) -> tuple[nn.Module, int]:
    if backbone_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        if spatial_features:
            feature_dim = int(backbone.classifier[0].in_features)
            return backbone.features, feature_dim
        feature_dim = int(backbone.classifier[-1].in_features)
        backbone.classifier[-1] = nn.Identity()
        return backbone, feature_dim

    if backbone_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = int(backbone.fc.in_features)
        if spatial_features:
            return nn.Sequential(*list(backbone.children())[:-2]), feature_dim
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    if backbone_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        feature_dim = int(backbone.fc.in_features)
        if spatial_features:
            return nn.Sequential(*list(backbone.children())[:-2]), feature_dim
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    if backbone_name == "tiny_cnn":
        feature_dim = 32
        layers: list[nn.Module] = [
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        if not spatial_features:
            layers.extend((nn.AdaptiveAvgPool2d(1), nn.Flatten()))
        return nn.Sequential(*layers), feature_dim

    raise ValueError(f"Unsupported attribute backbone: {backbone_name}")
