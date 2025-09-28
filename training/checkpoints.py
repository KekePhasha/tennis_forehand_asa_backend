import os, json, torch

import numpy as np

from models.linear.layers.layers import Linear


# Pure-Python: JSON
def save_pure_json(model, json_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    dump = {"layers": []}
    for layer in model.net.layers:
        if isinstance(layer, Linear):
            dump["layers"].append({"type": "Linear", "W": layer.W, "b": layer.b})
        else:
            dump["layers"].append({"type": type(layer).__name__})
    with open(json_path, "w") as f:
        json.dump(dump, f)

def load_pure_json(model, path):
    with open(path, "r") as f:
        state = json.load(f)

    # Go through layers in order
    for layer, layer_state in zip(model.net.layers, state["layers"]):
        if hasattr(layer, "W") and "W" in layer_state:
            layer.W = np.array(layer_state["W"], dtype=float).tolist()
        if hasattr(layer, "b") and "b" in layer_state:
            layer.b = np.array(layer_state["b"], dtype=float).tolist()
        if hasattr(layer, "gamma") and "gamma" in layer_state:
            layer.gamma = np.array(layer_state["gamma"], dtype=float).tolist()
        if hasattr(layer, "beta") and "beta" in layer_state:
            layer.beta = np.array(layer_state["beta"], dtype=float).tolist()

# Torch models: .pth
def save_torch(model, pth_path: str):
    os.makedirs(os.path.dirname(pth_path), exist_ok=True)
    torch.save(model.state_dict(), pth_path)

def load_torch(model, pth_path: str, map_location="cpu"):
    model.load_state_dict(torch.load(pth_path, map_location=map_location))
