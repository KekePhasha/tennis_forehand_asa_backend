from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# If the backbone is in the same package:
from .pose_bodypart_attn import PoseBodyPartAttentionModel


class SiameseWrapper(nn.Module):
    """
    Wraps a backbone that produces an embedding z for a sequence of poses.
    Computes L2 distance between embeddings for contrastive-style training.

    Forward returns:
      - dist: (B,)   pairwise L2 distance between embeddings
      - (attn1, attn2): optional body-part attentions for each input
    """

    def __init__(self, backbone: PoseBodyPartAttentionModel, margin: float = 0.5, return_attn: bool = True):
        super(SiameseWrapper, self).__init__()
        self.backbone = backbone
        self.margin = margin
        self.return_attn = return_attn

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get just the embedding (inference/validation).
        x: (B,T,J,3)
        returns z: (B,E)
        """
        z, _, _ = self.backbone(x)
        return z

    def forward(
        self,
        x1: torch.Tensor,  # (B,T,J,3)
        x2: torch.Tensor   # (B,T,J,3)
    ):
        z1, attn1, _ = self.backbone(x1)  # z1: (B,E), attn1: (B,T,P)
        z2, attn2, _ = self.backbone(x2)  # z2: (B,E), attn2: (B,T,P)

        dist = torch.norm(z1 - z2, dim=-1)  # (B,)

        if self.return_attn:
            return dist, (attn1, attn2)
        return dist  # toggle this if your training loop expects only the distance

    # ------------------------
    # Loss helpers
    # ------------------------
    def contrastive_loss(self, dist: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Classic Hadsell et al. contrastive loss.
        y ∈ {0,1}: 1 = similar (positive), 0 = dissimilar (negative)
        """
        y = y.float()
        pos = y * (dist ** 2)
        neg = (1.0 - y) * F.relu(self.margin - dist) ** 2
        return (pos + neg).mean()

    def triplet_loss(
        self,
        za: torch.Tensor, zp: torch.Tensor, zn: torch.Tensor, margin: float | None = None
    ) -> torch.Tensor:
        """
        Optional triplet loss if you generate triplets.
        za/zp/zn: (B,E) anchor/positive/negative embeddings
        """
        m = self.margin if margin is None else margin
        d_ap = torch.norm(za - zp, dim=-1)
        d_an = torch.norm(za - zn, dim=-1)
        return F.relu(d_ap - d_an + m).mean()
