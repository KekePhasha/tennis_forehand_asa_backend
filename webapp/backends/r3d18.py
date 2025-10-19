from __future__ import annotations
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from .base import BaseBackend, InferenceResult
from webapp.config import CHECKPOINTS
from webapp.services.video_io import load_video_clip

class TorchR3D18Backend(BaseBackend):
    """
    Two-video similarity with a 3D-ResNet-18 backbone.
    - Preprocess: sample short clip from each video -> [1,3,T,H,W]
    - Model: r3d_18, global pooled embedding via fc=Identity
    - Score: cosine distance + calibrated similarity
    """
    key = "r3d_18"

    def __init__(self, device=None, num_frames=16, size=112, tau=0.4):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.size = size
        self.tau = float(tau)
        self.model: nn.Module | None = None

    def load(self) -> None:
        """
        1) Try custom checkpoint at checkpoints/r3d18.pth (if you trained/fine-tuned it)
        2) Otherwise fall back to torchvision Kinetics-400 weights for embeddings
        """
        ckpt = (CHECKPOINTS / "r3d18.pth")
        if ckpt.exists():
            # init base arch, then replace classification head with Identity for embeddings
            self.model = r3d_18(weights=None)
            self.model.fc = nn.Identity()
            state = torch.load(str(ckpt), map_location="cpu")
            self.model.load_state_dict(state, strict=False)
        else:
            # Use ImageNet/Kinetics-pretrained feature extractor
            weights = R3D_18_Weights.KINETICS400_V1
            self.model = r3d_18(weights=weights)
            self.model.fc = nn.Identity()  # 512-d embedding

        self.model.eval().to(self.device)

    def preprocess(self, sample_path: str, ref_path: str) -> Dict[str, Any]:
        """
        Returns two preprocessed clips for similarity:
          left_clip, right_clip: [1,3,T,H,W] float tensors normalized with Kinetics stats
        """
        left  = load_video_clip(sample_path, num_frames=self.num_frames, size=self.size, device=self.device)
        right = load_video_clip(ref_path,    num_frames=self.num_frames, size=self.size, device=self.device)
        return {"left": left, "right": right}

    @torch.no_grad()
    def infer(self, data: Dict[str, Any]) -> InferenceResult:
        xL, xR = data["left"], data["right"]  # [1,3,T,H,W]

        # Forward -> 512-d embeddings
        zL = self.model(xL)  # [1,512]
        zR = self.model(xR)  # [1,512]

        # L2-normalize and cosine distance
        zL = F.normalize(zL, dim=-1)
        zR = F.normalize(zR, dim=-1)

        cos_sim = (zL * zR).sum(dim=-1).clamp(-1, 1).item()      # [-1,1]
        distance = float(1.0 - cos_sim)                           # 0..2
        similarity = float((cos_sim + 1.0) * 0.5)                 # map to [0,1]

        return InferenceResult(
            distance=distance,
            similarity_score=similarity,
            is_similar=bool(distance < self.tau),
            extras={"cosine_similarity": cos_sim, "tau": self.tau}
        )
