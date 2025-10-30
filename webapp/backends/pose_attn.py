from __future__ import annotations
from typing import Optional, List, Iterable, Set

import math
import torch
from models.pose_attn.siamese_wrapper import SiameseWrapper
from models.pose_attn.pose_bodypart_attn import PoseBodyPartAttentionModel
from webapp.backends.base import BaseBackend, InferenceResult
from webapp.config import CHECKPOINTS
from webapp.services.pose_io import ViTPoseWrapper

# ViTPose-17 joint groups (edit if you use a different skeleton)
BODY_PARTS = {
    "head": [0, 1, 2, 3, 4],  # nose, eyes, ears
    "torso": [5, 6, 11, 12],  # L-shoulder, R-shoulder, L-hip, R-hip
    "left_arm": [5, 7, 9],  # L-shoulder, L-elbow, L-wrist
    "right_arm": [6, 8, 10],  # R-shoulder, R-elbow, R-wrist
    "left_leg": [11, 13, 15],  # L-hip, L-knee, L-ankle
    "right_leg": [12, 14, 16],  # R-hip, R-knee, R-ankle
}

NUM_JOINTS = 17


def _normalize_parts(parts: Optional[Iterable]) -> Optional[List[int]]:
    """
    Convert `parts` (list of ints/strings) into a sorted list of unique valid joint indices.
    Returns None if parts is None or empty AFTER cleaning => 'use all'.
    """
    if parts is None:
        return None
    cleaned: Set[int] = set()
    for p in parts:
        try:
            i = int(p)
        except Exception:
            continue
        if 0 <= i < NUM_JOINTS:
            cleaned.add(i)
    if not cleaned:
        return None
    return sorted(cleaned)


def _apply_joint_mask(x: torch.Tensor, keep_idx: Optional[List[int]]) -> torch.Tensor:
    """
    x: (B, T, J, 3). If keep_idx is provided, zero-out joints not in keep_idx.
    Returns same tensor (in-place safe for speed).
    """
    if keep_idx is None:
        return x
    # Build a mask for joints to zero
    J = x.shape[2]
    device = x.device
    keep = torch.zeros(J, dtype=torch.bool, device=device)
    keep[keep_idx] = True
    drop = ~keep
    if drop.any():
        x[:, :, drop, :] = 0.0
    return x


class PoseAttnBackend(BaseBackend):
    key = "pose_attn"

    def __init__(self, device=None, tau=0.75):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pose = None
        self.model = None
        self.tau = float(tau)
        self.used_parts: Optional[List[int]] = None
        self.calib_scale = 0.05

    def _build_backbone(self) -> PoseBodyPartAttentionModel:
        return PoseBodyPartAttentionModel(
            body_parts=BODY_PARTS,
            joint_in_dim=3,  # (x,y,conf)
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

        # Load checkpoint
        state = torch.load(ckpt, map_location=self.device)
        # Check for wrapped state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError:
            # Fallback: if checkpoint was saved as backbone only
            try:
                self.model.backbone.load_state_dict(state, strict=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load pose_attn checkpoint: {e}")

        # Load calibration if present
        calib = CHECKPOINTS / "pose_attn_calibration.json"
        if calib.exists():
            import json, math
            with open(calib) as f:
                cfg = json.load(f)
            tau = float(cfg.get("tau", self.tau))
            scale = float(cfg.get("scale", self.calib_scale))
            if not math.isfinite(scale) or scale <= 0:
                scale = self.calib_scale  # fall back if bad
            self.tau, self.calib_scale = tau, scale
            print(f"[pose_attn] loaded calibration: tau={self.tau:.4f}, scale={self.calib_scale:.4f}")

    def preprocess(self, sample_path: str, ref_path: str, parts: Optional[Iterable] = None):
        """
        Expect ViTPoseWrapper.extract_keypoints(...) -> (np.ndarray[T,J,3], extras)
        """
        import numpy as np
        import torch

        # 1) call wrapper WITHOUT unsupported kwargs
        xL_np, _ = self.pose.extract_keypoints(sample_path, save_name='sample')
        xR_np, _ = self.pose.extract_keypoints(ref_path, save_name='ref')

        # 2) sanity checks & dtype
        if not isinstance(xL_np, np.ndarray) or not isinstance(xR_np, np.ndarray):
            raise RuntimeError("extract_keypoints must return numpy arrays shaped (T, J, 3).")
        if xL_np.ndim != 3 or xR_np.ndim != 3 or xL_np.shape[2] != 3 or xR_np.shape[2] != 3:
            raise RuntimeError(f"Expected (T,J,3); got {xL_np.shape} and {xR_np.shape}.")
        if xL_np.shape[1] != NUM_JOINTS or xR_np.shape[1] != NUM_JOINTS:
            raise RuntimeError(f"Expected {NUM_JOINTS} joints; got {xL_np.shape[1]} and {xR_np.shape[1]}.")

        # 3) to torch on the right device, add batch dim -> (1,T,J,3)
        xL = torch.from_numpy(xL_np).float().unsqueeze(0).to(self.device)
        xR = torch.from_numpy(xR_np).float().unsqueeze(0).to(self.device)

        # 4) normalise/keep selected joints (None => use all)
        keep_idx = _normalize_parts(parts)
        self.used_parts = keep_idx
        xL = _apply_joint_mask(xL, keep_idx)
        xR = _apply_joint_mask(xR, keep_idx)

        return {"left": xL, "right": xR}

    @torch.no_grad()
    def infer(self, data, parts: Optional[Iterable] = None):
        """
        Run the pose attention Siamese model and return a standardised result.
        :param data: Dict with tensors "left" and "right" of shape (B,T,J,3)
        :param parts: (ignored here, used in preprocessing)
        :return: InferenceResult with distance, similarity_score, is_similar, extras
        """

        dist, _ = self.model(data["left"], data["right"])
        d = float(dist.squeeze().item())
        if d <= 1e-12:
            sim = 1.0
        else:
            display_scale = max(1e-12, self.tau / 4.595)  # ln(99) → ~99% at d=0
            sim = 1.0 / (1.0 + math.exp((d - self.tau) / display_scale))

        return InferenceResult(
            distance=d,
            similarity_score=sim,
            is_similar=bool(d < self.tau),
            extras={
                "used_parts": "all" if self.used_parts is None else self.used_parts,
                "tau": self.tau,
                "scale": self.calib_scale,
            }
        )
