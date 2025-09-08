# models/vision/resnet_siamese.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNetSiamese(nn.Module):
    def __init__(self, embed_dim: int = 128, use_pretrained: bool = True, freeze_backbone: bool = False):
        super(ResNetSiamese, self).__init__()
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        self.backbone = resnet18(weights=weights)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)              # (B, F)
        embeddings = self.projection_head(features)   # (B, embed_dim)
        return F.normalize(embeddings, dim=1)

    def forward(self, left_images: torch.Tensor, right_images: torch.Tensor):
        left_embed  = self.forward_once(left_images)
        right_embed = self.forward_once(right_images)
        distance_tensor = F.pairwise_distance(left_embed, right_embed)
        return left_embed, right_embed, distance_tensor
