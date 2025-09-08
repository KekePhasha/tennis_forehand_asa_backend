import math
from typing import List
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
    Linear model with two hidden layers:
      51 -> 64 -> 32
    """

    def __init__(self, input_dim=51, hidden_dim=64, embed_dim=32, seed=7):
        self.net = Sequential([
            Linear(input_dim, hidden_dim, seed=seed + 1),
            ReLU(),
            Linear(hidden_dim, embed_dim, seed=seed + 2),
        ])

    def zero_grad(self):
        self.net.zero_grad()

    def forward_once(self, x: List[List[float]]):
        return self.net.forward(x)  # caches are inside layers

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
        loss, dz1, dz2 = siamese_contrastive_backward(z1, z2, labels, margin=margin)

        # Backprop through each tower; since weights are shared, we must:
        #  - run backward on z1 grads
        #  - run backward on z2 grads (accumulate into the same parameter grads)
        self.backward_once(dz2)
        self.backward_once(dz1)

        # One optimizer step for shared parameters
        self.step(lr)
        return loss

    def distances(self, left_vectors, right_vectors):
        left_vectors = l2_normalize_rows(left_vectors)
        right_vectors = l2_normalize_rows(right_vectors)
        left_embeddings = self.forward_once(left_vectors)
        right_embeddings = self.forward_once(right_vectors)
        dists = []
        for a, b in zip(left_embeddings, right_embeddings):
            dists.append(math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b))))
        return dists
