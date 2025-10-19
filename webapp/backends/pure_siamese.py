from __future__ import annotations
import json, math
import numpy as np

from base import BaseBackend, InferenceResult
from webapp.services.pose_io import ViTPoseWrapper
from embedding.pose_embedding import PoseEmbedding
from models import SiameseModelTrainable
from training.checkpoints import load_pure_json
from webapp.config import CHECKPOINTS, TAU_DEFAULT, SCALE_DEFAULT

class PureSiameseBackend(BaseBackend):
    key = "pure"

    def __init__(self):
        self.pose = None
        self.embed = None
        self.model = None
        self.tau = TAU_DEFAULT
        self.scale = SCALE_DEFAULT

    def _to_list51(self, vec):
        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if v.shape[1] != 51:
            raise ValueError(f"Expected 51-dim embedding, got {v.shape[1]}")
        n = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(n, 1e-12)
        return v.tolist()

    def _prob_similar(self, distance: float) -> float:
        z = (self.tau - float(distance)) / max(1e-6, float(self.scale))
        return 1.0 / (1.0 + math.exp(-z))

    def load(self) -> None:
        # pose/keypoints & embedding
        self.pose = ViTPoseWrapper()
        self.embed = PoseEmbedding(confidence_threshold=0.6)

        # checkpoints
        ckpt = CHECKPOINTS / "pure_siamese.json"
        calib = CHECKPOINTS / "pure_siamese_calibration.json"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing {ckpt}")

        self.model = SiameseModelTrainable(input_dim=51, hidden_dim=128, embed_dim=64, seed=7)
        load_pure_json(self.model, str(ckpt))

        if calib.exists():
            with open(calib, "r") as f:
                c = json.load(f)
            self.tau = float(c.get("tau", TAU_DEFAULT))
            self.scale = float(c.get("scale", SCALE_DEFAULT))

    def preprocess(self, sample_path: str, ref_path: str):
        sample_kp, _ = self.pose.extract_keypoints(sample_path, save_name='sample')
        ref_kp, _    = self.pose.extract_keypoints(ref_path, save_name='ref')
        sample_emb = self.embed.generate_embedding(sample_kp)
        ref_emb    = self.embed.generate_embedding(ref_kp)
        return {
            "left":  self._to_list51(sample_emb),
            "right": self._to_list51(ref_emb),
        }

    def infer(self, data):
        left, right = data["left"], data["right"]
        distance = float(self.model.distances(left, right, train=False)[0])
        sim = self._prob_similar(distance)
        return InferenceResult(
            distance=distance,
            similarity_score=sim,
            is_similar=bool(distance < self.tau),
            extras={"tau": self.tau, "scale": self.scale}
        )
