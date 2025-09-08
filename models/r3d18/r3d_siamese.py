import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mc3_18, MC3_18_Weights

"""
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html
https://arxiv.org/abs/1711.11248
"""

class R3D18Siamese(nn.Module):
    def __init__(self, embed_dim: int = 128, use_pretrained: bool = True, freeze_backbone: bool = False):
        super(R3D18Siamese, self).__init__()
        weights = MC3_18_Weights.DEFAULT if use_pretrained else None
        self.backbone = mc3_18(weights=weights, progress=True)
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

    def forward_once(self, clips_3d: torch.Tensor) -> torch.Tensor:
        features = self.backbone(clips_3d)            # (B, F)
        embeddings = self.projection_head(features)   # (B, embed_dim)
        return F.normalize(embeddings, dim=1)

    def forward(self, left_clips: torch.Tensor, right_clips: torch.Tensor):
        left_embed  = self.forward_once(left_clips)
        right_embed = self.forward_once(right_clips)
        distance_tensor = F.pairwise_distance(left_embed, right_embed)
        return left_embed, right_embed, distance_tensor
