# training/pair_builder.py
import os, glob, random
from typing import List, Tuple

def make_video_pairs_from_folders(
    videos_root: str,
    max_pos_pairs_per_class: int = 10_000,
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
) -> tuple[list[tuple[str, str]], list[int]]:
    """
    Expects:
      videos_root/
        classA/*.mp4|avi|mov|mkv
        classB/*.mp4|avi|mov|mkv
    Returns:
      pairs:  list[(left_video_path, right_video_path)]
      labels: list[int]  (0 = similar / same class, 1 = dissimilar)
    """
    # collect per-class videos
    videos_by_class: dict[str, list[str]] = {}
    for class_name in sorted(os.listdir(videos_root)):
        class_dir = os.path.join(videos_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = []
        for root, _, filenames in os.walk(class_dir):
            for fname in filenames:
                if fname.lower().endswith(video_exts):
                    files.append(os.path.join(root, fname))
        if len(files) >= 2:
            videos_by_class[class_name] = files

    # positives (same class)
    positive_pairs: list[tuple[str, str]] = []
    for class_name, file_list in videos_by_class.items():
        file_list = file_list[:]  # copy & shuffle
        random.shuffle(file_list)
        count = 0
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                positive_pairs.append((file_list[i], file_list[j]))
                count += 1
                if count >= max_pos_pairs_per_class:
                    break
            if count >= max_pos_pairs_per_class:
                break
    positive_labels = [0] * len(positive_pairs)  # 0 = similar

    # negatives (different class), balanced to #positives
    class_names = list(videos_by_class.keys())
    negative_pairs: list[tuple[str, str]] = []
    target_negs = len(positive_pairs)
    if len(class_names) >= 2:
        while len(negative_pairs) < target_negs:
            c1, c2 = random.sample(class_names, 2)
            v1 = random.choice(videos_by_class[c1])
            v2 = random.choice(videos_by_class[c2])
            negative_pairs.append((v1, v2))
    negative_labels = [1] * len(negative_pairs)  # 1 = dissimilar

    # merge & shuffle
    pairs  = positive_pairs + negative_pairs
    labels = positive_labels + negative_labels
    idx = list(range(len(pairs)))
    random.shuffle(idx)
    pairs  = [pairs[i] for i in idx]
    labels = [labels[i] for i in idx]
    return pairs, labels
