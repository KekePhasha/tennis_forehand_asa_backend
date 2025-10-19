from __future__ import annotations
from typing import Optional
import torch
import torchvision.transforms as T
from torchvision.io import read_video

# Kinetics-400 RGB stats used by torchvision video models
_K400_MEAN = [0.43216, 0.394666, 0.37645]
_K400_STD  = [0.22803, 0.22145, 0.216989]

def _uniform_indices(total: int, num: int):
    if total <= 0: return [0] * num
    if total < num:
        # pad by repeating last frame
        idx = list(range(total)) + [total - 1] * (num - total)
        return idx
    step = total / float(num)
    return [int(i * step) for i in range(num)]

def load_video_clip(
    path: str,
    num_frames: int = 16,
    size: int = 112,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Load a short uniformly-sampled clip and normalize for r3d_18.
    Returns: [1, 3, T, H, W] float32 on device.
    """
    # read_video -> (T, H, W, C) uint8 in [0,255]
    video, _, _ = read_video(path, output_format="THWC")  # T,H,W,C
    T_total = video.shape[0]

    sel = _uniform_indices(T_total, num_frames)
    clip = video[sel]  # (T, H, W, C)

    # To float tensor in [0,1] and CHW per frame
    clip = clip.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

    # Resize + center crop per frame (vectorized with torchvision v0.14+ ops)
    # If your torchvision is older, loop frames and apply PIL transforms.
    spatial = T.Compose([
        T.Resize(size, antialias=True),
        T.CenterCrop(size),
        T.Normalize(mean=_K400_MEAN, std=_K400_STD),
    ])
    # Apply per frame
    clip = spatial(clip)  # (T, C, H, W) — Normalize supports batched CHW per docs

    clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
    if device:
        clip = clip.to(device)
    return clip.unsqueeze(0)  # (1, C, T, H, W)
