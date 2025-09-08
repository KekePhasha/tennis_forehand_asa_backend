# siamese/utils/checkpoint.py
import json, os
from typing import Any, Dict
from siamese.training.layers import Linear, ReLU, Sequential

def save_model_json(model, path: str, meta: Dict[str, Any] = None):
    """
    Save Linear layer params (W, b) (+ momentum buffers if present) to JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump = {
        "arch": {"layers": [type(l).__name__ for l in model.net.layers]},
        "params": [],
        "meta": meta or {}
    }
    for layer in model.net.layers:
        entry = {"type": type(layer).__name__}
        if isinstance(layer, Linear):
            entry["W"] = layer.W
            entry["b"] = layer.b
            # if you later add momentum buffers:
            if hasattr(layer, "vW"): entry["vW"] = layer.vW
            if hasattr(layer, "vb"): entry["vb"] = layer.vb
        dump["params"].append(entry)

    with open(path, "w") as f:
        json.dump(dump, f)

def load_model_json(model, path: str, strict: bool = True):
    """
    Load params into an existing model with same topology.
    """
    with open(path) as f:
        obj = json.load(f)

    saved_layers = obj["params"]
    now_layers = model.net.layers
    if strict and len(saved_layers) != len(now_layers):
        raise ValueError("Layer count mismatch.")
    for saved, layer in zip(saved_layers, now_layers):
        if strict and saved["type"] != type(layer).__name__:
            raise ValueError(f"Layer type mismatch: saved {saved['type']} vs now {type(layer).__name__}")
        if isinstance(layer, Linear):
            layer.W = saved["W"]
            layer.b = saved["b"]
            if "vW" in saved and hasattr(layer, "vW"): layer.vW = saved["vW"]
            if "vb" in saved and hasattr(layer, "vb"): layer.vb = saved["vb"]
