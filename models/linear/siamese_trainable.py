import math
from typing import List

from models.linear.layers.batchnorm import BatchNorm1d
from models.linear.layers.dropout import Dropout
from models.linear.layers.layers import Linear, ReLU, Sequential
from models.linear.layers.loss import siamese_contrastive_backward


def l2_normalize_rows(matrix_list: List[List[float]]):
    normalized = []
    for row in matrix_list:
        norm = math.sqrt(sum(v*v for v in row))
        normalized.append([v / norm for v in row] if norm > 0 else row[:])
    return normalized

class SiameseModelTrainable:
    """
    Linear model with three hidden layers:
    Input (51) → Linear(128) → BN → ReLU → Dropout
           → Linear(64) → BN → ReLU → Dropout
           → Linear(32) → Embedding
    Each hidden layer has BatchNorm, ReLU, Dropout(0.3).
    The final layer is linear, no activation.
    """

    def __init__(self, input_dim=51, hidden_dim=128, embed_dim=32, seed=7,
                 use_bn=True, use_dropout=True):

        layers = [
            Linear(input_dim, hidden_dim, seed=seed + 1),
        ]
        if use_bn: layers.append(BatchNorm1d(hidden_dim))
        layers.append(ReLU())
        if use_dropout: layers.append(Dropout(p=0.4))

        layers.append(Linear(hidden_dim, 64, seed=seed + 2))
        if use_bn: layers.append(BatchNorm1d(64))
        layers.append(ReLU())
        if use_dropout: layers.append(Dropout(p=0.3))

        layers.append(Linear(64, 64, seed=seed + 3))
        if use_bn: layers.append(BatchNorm1d(64))
        layers.append(ReLU())
        if use_dropout: layers.append(Dropout(p=0.3))

        layers.append(Linear(64, embed_dim, seed=seed + 3))

        self.net = Sequential(layers)



    def zero_grad(self):
        self.net.zero_grad()

    def forward_once(self, x: List[List[float]], train=True):
        out = self.net.forward(x, train=train)
        # normalize each embedding vector to unit length
        normed = []
        for row in out:
            norm = math.sqrt(sum(v * v for v in row))
            normed.append([v / norm for v in row] if norm > 0 else row[:])
        return normed

    def backward_once(self, d_embed: List[List[float]]):
        return self.net.backward(d_embed)

    def step(self, lr: float):
        self.net.step(lr)

    def train_batch(self, x1: List[List[float]], x2: List[List[float]], labels: List[int], lr=1e-3, margin=1.0):
        self.zero_grad()

        # Forward through both towers (shared weights)
        z1 = self.forward_once(x1)  # [B, D]
        z2 = self.forward_once(x2)  # [B, D]

        # Contrastive loss + grads wrt embeddings
        labels_for_loss = [1 - int(l) for l in labels]
        loss, dz1, dz2 = siamese_contrastive_backward(z1, z2, labels_for_loss, margin=margin)

        # Backprop through each tower; since weights are shared, we must:
        #  - run backward on z1 grads
        #  - run backward on z2 grads (accumulate into the same parameter grads)
        self.backward_once(dz2)
        self.backward_once(dz1)

        # One optimizer step for shared parameters
        self.step(lr)
        return loss

    def distances(self, left_vectors, right_vectors, train=False):
        """
        Distances between pairs of embeddings.
        :param train:
        :param left_vectors:
        :param right_vectors:
        :return:
        """
        left_vectors = l2_normalize_rows(left_vectors)
        right_vectors = l2_normalize_rows(right_vectors)
        left_embeddings = self.forward_once(left_vectors, train=train)
        right_embeddings = self.forward_once(right_vectors, train=train)
        dists = []
        for a, b in zip(left_embeddings, right_embeddings):
            dists.append(math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b))))
        return dists
