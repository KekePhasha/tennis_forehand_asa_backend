import os, json, torch
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

def load_pure_json(model, json_path: str):
    with open(json_path) as f:
        obj = json.load(f)
    for layer, saved in zip(model.net.layers, obj["layers"]):
        if isinstance(layer, Linear) and saved["type"] == "Linear":
            layer.W = saved["W"]; layer.b = saved["b"]

# Torch models: .pth
def save_torch(model, pth_path: str):
    os.makedirs(os.path.dirname(pth_path), exist_ok=True)
    torch.save(model.state_dict(), pth_path)

def load_torch(model, pth_path: str, map_location="cpu"):
    model.load_state_dict(torch.load(pth_path, map_location=map_location))
