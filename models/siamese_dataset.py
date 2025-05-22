import numpy as np
from torch.utils.data import Dataset


class SiamesePoseDataset(Dataset):
    """
    Custom dataset for Siamese network with pose embeddings.
    This dataset takes pairs of keypoints and their corresponding labels.
    The labels indicate whether the pairs are similar (1) or dissimilar (0).
    The dataset is used for training a Siamese network to learn a similarity function.
    """
    def __init__(self, pairs, labels):
        """
        Initialize the SiamesePoseDataset with pairs of keypoints and their labels.
        :param pairs: List of tuples containing pairs of keypoints.
        :param labels: List of labels corresponding to the pairs.
        """
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        """
        Return the number of pairs in the dataset.
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2 = self.pairs[idx]
        emb1 = np.load(path1).astype(np.float32)
        emb2 = np.load(path2).astype(np.float32)
        label = np.float32(self.labels[idx])
        return emb1, emb2, label