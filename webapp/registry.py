from typing import Dict, Type
from backends.base import BaseBackend
from backends.pure_siamese import PureSiameseBackend
from backends.pose_attn import PoseAttnBackend
from backends.r3d18 import TorchR3D18Backend

_REGISTRY: Dict[str, Type[BaseBackend]] = {
    PureSiameseBackend.key: PureSiameseBackend,
    PoseAttnBackend.key: PoseAttnBackend,
    TorchR3D18Backend.key: TorchR3D18Backend,
}


def build_backend(name: str) -> BaseBackend:
    name = (name or "").lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Choices: {list(_REGISTRY.keys())}")
    backend = _REGISTRY[name]()
    backend.load()
    return backend
