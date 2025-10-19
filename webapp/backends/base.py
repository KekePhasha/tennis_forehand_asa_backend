from __future__ import annotations
from typing import Dict, Any, Tuple

class InferenceResult(dict):
    """Uniform result: distance (float), similarity_score (0..1), is_similar (bool), extras (dict)."""
    pass

class BaseBackend:
    key: str = "base"

    def load(self) -> None:
        """Load checkpoints, heavy models, etc."""
        raise NotImplementedError

    def preprocess(self, sample_path: str, ref_path: str) -> Dict[str, Any]:
        """Return any tensors/embeddings needed by infer()."""
        raise NotImplementedError

    def infer(self, data: Dict[str, Any]) -> InferenceResult:
        """Run model and return standardized result."""
        raise NotImplementedError
