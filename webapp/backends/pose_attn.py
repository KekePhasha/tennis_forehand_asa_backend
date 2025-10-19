from __future__ import annotations
import torch
from .base import BaseBackend, InferenceResult
from services.pose_io import ViTPoseWrapper
from models.pose_attn.siamese_wrapper import SiameseWrapper  # your torch wrapper
from webapp.config import CHECKPOINTS

class PoseAttnBackend(BaseBackend):
    key = "pose_attn"

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pose = None
        self.model = None
        self.tau = 0.75  # example

    def load(self):
        self.pose = ViTPoseWrapper()
        ckpt = CHECKPOINTS / "pose_attn_siamese.pth"
        self.model = SiameseWrapper().to(self.device).eval()
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)

    def preprocess(self, sample_path: str, ref_path: str):
        # Return stacked tensors (B,T,J,3) or whatever your wrapper expects
        xL, _ = self.pose.extract_keypoints(sample_path, save_name='sample', as_tensor=True, device=self.device)
        xR, _ = self.pose.extract_keypoints(ref_path,    save_name='ref',    as_tensor=True, device=self.device)
        return {"left": xL, "right": xR}

    @torch.no_grad()
    def infer(self, data):
        zL, zR, dist = self.model(data["left"], data["right"])  # your wrapper returns (zL, zR, distances)
        d = float(dist.squeeze().item())
        sim = 1.0 / (1.0 + d)
        return InferenceResult(
            distance=d,
            similarity_score=sim,
            is_similar=bool(d < self.tau),
            extras={}
        )
