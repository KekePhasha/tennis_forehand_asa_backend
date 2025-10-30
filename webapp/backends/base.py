from __future__ import annotations
from typing import Any, Dict, Optional

class InferenceResult(dict):
    """
    Uniform result:
      - distance: float
      - similarity_score: float in [0, 1]
      - is_similar: bool
      - extras: dict (may include: 'artefacts', 'used_parts', 'tau', 'scale', etc.)
    Optionally accept a top-level 'artifact' and normalise it into extras['artifacts'].
    """

    def __init__(
        self,
        distance: float,
        similarity_score: float,
        is_similar: bool,
        extras: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            distance=float(distance),
            similarity_score=float(similarity_score),
            is_similar=bool(is_similar),
            extras=dict(extras or {}),
        )
        if artifacts:
            # keep backward-compat but prefer extras['artifacts']
            self["artifacts"] = artifacts
            self.ensure_artifacts_in_extras()

    # --- convenience accessors ---
    @property
    def distance(self) -> float:
        return float(self["distance"])

    @property
    def similarity_score(self) -> float:
        return float(self["similarity_score"])

    @property
    def is_similar(self) -> bool:
        return bool(self["is_similar"])

    @property
    def extras(self) -> Dict[str, Any]:
        return self.setdefault("extras", {})

    # --- normalization helpers ---
    def ensure_artifacts_in_extras(self) -> None:
        """Move top-level 'artifacts' into extras['artifacts'] if present."""
        arts = self.pop("artifacts", None)
        if arts is not None:
            self.setdefault("extras", {})["artifacts"] = arts

    @staticmethod
    def coerce(obj: Any) -> "InferenceResult":
        """
        Coerce a dict-like result into InferenceResult.
        Expects keys: distance, similarity_score, is_similar (and optional extras, artifacts).
        """
        if isinstance(obj, InferenceResult):
            return obj
        if isinstance(obj, dict):
            res = InferenceResult(
                distance=obj.get("distance"),
                similarity_score=obj.get("similarity_score"),
                is_similar=obj.get("is_similar"),
                extras=obj.get("extras", {}),
                artifacts=obj.get("artifacts"),
            )
            return res
        raise TypeError(
            "infer() must return a dict-like with keys: distance, similarity_score, is_similar."
        )


class BaseBackend:
    key: str = "base"
    # Optional capability flags for frontends/routers
    supports_parts: bool = False  # pose_attn sets this True

    def load(self) -> None:
        """Load checkpoints, heavy models, etc."""
        raise NotImplementedError

    def preprocess(self, sample_path: str, ref_path: str, **kwargs) -> Dict[str, Any]:
        """
        Return tensors/embeddings needed by infer().
        Accepts **kwargs (e.g., parts=list[int]) for backends that support it.
        """
        raise NotImplementedError

    def infer(self, data: Dict[str, Any], **kwargs) -> InferenceResult:
        """
        Run model and return standardised result.
        Accepts **kwargs (e.g. parts=list[int]) for backends that support it.
        """
        raise NotImplementedError
