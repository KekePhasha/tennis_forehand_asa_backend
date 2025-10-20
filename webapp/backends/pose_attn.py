from __future__ import annotations
import torch
from .base import BaseBackend, InferenceResult
from models.pose_attn.siamese_wrapper import SiameseWrapper
from models.pose_attn.pose_bodypart_attn import PoseBodyPartAttentionModel
from webapp.config import CHECKPOINTS
from webapp.services.pose_io import ViTPoseWrapper

# COCO/ViTPose-17 joint groups (edit if you use a different skeleton)
BODY_PARTS = {
    "head":        [0,1,2,3,4],     # nose, eyes, ears
    "shoulders":   [5,6],
    "elbows":      [7,8],
    "wrists":      [9,10],
    "hips":        [11,12],
    "knees":       [13,14],
    "ankles":      [15,16],
}

class PoseAttnBackend(BaseBackend):
    key = "pose_attn"

    def __init__(self, device=None, tau=0.75):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pose = None
        self.model = None
        self.tau = float(tau)

    def _build_backbone(self) -> PoseBodyPartAttentionModel:
        # match your training dims
        return PoseBodyPartAttentionModel(
            body_parts=BODY_PARTS,
            joint_in_dim=3,          # (x,y,conf)
            joint_hidden=128,
            joint_out_dim=128,
            part_heads=4,
            temporal_layers=2,
            temporal_heads=4,
            emb_dim=128,
            num_classes=None,
            dropout=0.1,
        )

    def load(self):
        self.pose = ViTPoseWrapper()
        ckpt = CHECKPOINTS / "pose_attn_siamese.pth"

        backbone = self._build_backbone().to(self.device)
        self.model = SiameseWrapper(backbone=backbone, margin=0.5, return_attn=True).to(self.device).eval()

        state = torch.load(ckpt, map_location=self.device)
        # Handle different checkpoint formats (direct state_dict, wrapped, EMA, etc.)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # If keys are prefixed (e.g., "backbone."), try stripping/prefixing as needed
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError:
            # Fallback: if checkpoint was saved as backbone only
            try:
                self.model.backbone.load_state_dict(state, strict=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load pose_attn checkpoint: {e}")

    def preprocess(self, sample_path: str, ref_path: str):
        # Return tensors (B,T,J,3) on the right device
        xL, _ = self.pose.extract_keypoints(sample_path, save_name='sample', as_tensor=True, device=self.device)
        xR, _ = self.pose.extract_keypoints(ref_path,    save_name='ref',    as_tensor=True, device=self.device)

        # Ensure batch dim
        if xL.ndim == 3: xL = xL.unsqueeze(0)  # (1,T,J,3)
        if xR.ndim == 3: xR = xR.unsqueeze(0)
        return {"left": xL.to(self.device), "right": xR.to(self.device)}

    @torch.no_grad()
    def infer(self, data):
        # SiameseWrapper returns: dist, (attn1, attn2)
        dist, _ = self.model(data["left"], data["right"])
        d = float(dist.squeeze().item())
        sim = 1.0 / (1.0 + d)
        return InferenceResult(
            distance=d,
            similarity_score=sim,
            is_similar=bool(d < self.tau),
            extras={}
        )
