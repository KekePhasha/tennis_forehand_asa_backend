# data/datasets.py
from __future__ import annotations
from typing import Optional, Callable, Union, List, Tuple
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torchvision.io as io  # video I/O

ArrayLike = Union[str, List[float], np.ndarray, torch.Tensor]
PairList = List[Tuple[ArrayLike, ArrayLike]]
LabelList = List[int]

# =========================================================
# 1) KEYPOINTS (51-d vectors; 0 = similar, 1 = dissimilar)
# =========================================================
class KeypointsPairDataset(Dataset):
    """
    Pairs of 1-D pose embeddings (.npy paths or arrays). Returns:
      (left_keypoints: FloatTensor[51], right_keypoints: FloatTensor[51], label: LongTensor[()]).
    """
    def __init__(
        self,
        pair_paths_or_arrays: PairList,
        pair_labels: LabelList,
        transform_vector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.pair_paths_or_arrays = pair_paths_or_arrays
        self.pair_labels = pair_labels
        self.transform_vector = transform_vector  # e.g., L2-normalize

    def __len__(self) -> int:
        return len(self.pair_paths_or_arrays)

    def _load_keypoint_vector(self, source: ArrayLike) -> torch.Tensor:
        if isinstance(source, str):
            np_array = np.load(source)
        elif isinstance(source, torch.Tensor):
            np_array = source.detach().cpu().numpy()
        else:
            np_array = np.asarray(source)

        vector = torch.from_numpy(np_array.astype(np.float32).reshape(-1))  # (51,)
        if self.transform_vector is not None:
            vector = self.transform_vector(vector)
        return vector

    def __getitem__(self, index: int):
        left_source, right_source = self.pair_paths_or_arrays[index]
        raw_label = int(self.pair_labels[index])  # 0/1

        left_keypoints = self._load_keypoint_vector(left_source)
        right_keypoints = self._load_keypoint_vector(right_source)
        label_tensor = torch.tensor(raw_label, dtype=torch.long)

        return left_keypoints, right_keypoints, label_tensor


# ===============================================
# 2) IMAGES (ResNet-18; 0 = similar, 1 = dissimilar)
# ===============================================
class ImagePairDataset(Dataset):
    """
    Pairs of RGB image paths. Returns:
      (left_image: FloatTensor[3,224,224], right_image: FloatTensor[3,224,224], label: LongTensor[()]).
    """
    def __init__(
        self,
        image_pairs: PairList,
        pair_labels: LabelList,
        image_transform: Optional[T.Compose] = None,
    ):
        self.image_pairs = image_pairs
        self.pair_labels = pair_labels
        self.image_transform = image_transform or T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.image_pairs)

    def _load_image_tensor(self, image_path: str) -> torch.Tensor:
        pil_image = Image.open(image_path).convert("RGB")
        return self.image_transform(pil_image)

    def __getitem__(self, index: int):
        left_image_path, right_image_path = self.image_pairs[index]
        raw_label = int(self.pair_labels[index])

        left_image = self._load_image_tensor(left_image_path)
        right_image = self._load_image_tensor(right_image_path)
        label_tensor = torch.tensor(raw_label, dtype=torch.long)

        return left_image, right_image, label_tensor


# ======================================================
# 3) VIDEO CLIPS (R3D-18; 0 = similar, 1 = dissimilar)
# ======================================================
class VideoPairDataset(Dataset):
    """
    Pairs of video paths. Each item samples a clip and returns:
      (left_clip: FloatTensor[3,T,112,112], right_clip: FloatTensor[3,T,112,112], label: LongTensor[()]).
    """
    def __init__(
        self,
        video_pairs: PairList,
        pair_labels: LabelList,
        clip_length_frames: int = 16,
        spatial_size: int = 112,
    ):
        self.video_pairs = video_pairs
        self.pair_labels = pair_labels
        self.clip_length_frames = clip_length_frames
        self.spatial_size = spatial_size

        # Kinetics-400 normalization (broadcastable shapes)
        self.kinetics_mean_5d = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
        self.kinetics_std_5d  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.video_pairs)

    def _sample_video_clip(self, video_path: str) -> torch.Tensor:
        # frames_uint8: (total_frames, H, W, C)
        frames_uint8, _, _ = io.read_video(video_path, pts_unit="sec")
        total_frames = frames_uint8.shape[0]
        if total_frames == 0:
            raise RuntimeError("Empty or unreadable video: {}", format(video_path))

        # loop-pad if too short
        if total_frames < self.clip_length_frames:
            repeats_needed = (self.clip_length_frames + total_frames - 1) // total_frames
            frames_uint8 = frames_uint8.repeat(repeats_needed, 1, 1, 1)
            total_frames = frames_uint8.shape[0]

        clip_start = random.randint(0, total_frames - self.clip_length_frames)
        sampled_clip = frames_uint8[clip_start:clip_start + self.clip_length_frames]   # (T, H, W, C)
        clip_chw = sampled_clip.permute(3, 0, 1, 2).float() / 255.0                    # (C, T, H, W)

        # Resize to (T, spatial, spatial) via 5D trilinear interpolation
        clip_with_batch = clip_chw.unsqueeze(0)                                        # (1, C, T, H, W)
        clip_resized = F.interpolate(
            clip_with_batch,
            size=(self.clip_length_frames, self.spatial_size, self.spatial_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)                                                                   # (C, T, S, S)

        # Normalize with Kinetics stats
        clip_normalized = (clip_resized.unsqueeze(0) - self.kinetics_mean_5d) / self.kinetics_std_5d
        return clip_normalized.squeeze(0)                                              # (C, T, S, S)

    def __getitem__(self, index: int):
        left_video_path, right_video_path = self.video_pairs[index]
        raw_label = int(self.pair_labels[index])

        left_clip = self._sample_video_clip(left_video_path)
        right_clip = self._sample_video_clip(right_video_path)
        label_tensor = torch.tensor(raw_label, dtype=torch.long)

        return left_clip, right_clip, label_tensor


DatasetLike = Union[KeypointsPairDataset, ImagePairDataset, VideoPairDataset]
# ---------------------------------
# Optional factory for convenience
# ---------------------------------
def build_dataset(
    dataset_kind: str,
    pair_list: PairList,
    pair_labels: LabelList,
    **dataset_kwargs,
) -> DatasetLike:
    """
    dataset_kind: "pure" | "resnet18" | "r3d_18"
    dataset_kwargs forwarded to constructors.
    """
    kind = dataset_kind.lower()
    if kind == "pure":
        return KeypointsPairDataset(
            pair_paths_or_arrays=pair_list,
            pair_labels=pair_labels,
            transform_vector=dataset_kwargs.get("transform_vector"),
        )
    if kind == "resnet18":
        return ImagePairDataset(
            image_pairs=pair_list,
            pair_labels=pair_labels,
            image_transform=dataset_kwargs.get("image_transform"),
        )
    if kind == "r3d_18":
        return VideoPairDataset(
            video_pairs=pair_list,
            pair_labels=pair_labels,
            clip_length_frames=dataset_kwargs.get("clip_length_frames", 16),
            spatial_size=dataset_kwargs.get("spatial_size", 112),
        )
    raise ValueError("Unknown dataset kind: {}".format(dataset_kind))
