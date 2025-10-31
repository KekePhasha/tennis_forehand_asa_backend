import math
from typing import List, Tuple
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
    Input (51) →
      Linear(h1) → [BN] → ReLU → [Dropout]
      Linear(h2) → [BN] → ReLU → [Dropout]
      Linear(h3) → [BN] → ReLU → [Dropout]
      Linear(embed_dim) → L2-normalize (unit vectors)
    @param input_dim: input feature dimension (default 51)
    @param hidden_dim: hidden layer dimension (default 128)
    @param embed_dim: output embedding dimension (default 32)
    @param seed: random seed for weight initialisation
    @param use_bn: whether to use BatchNorm after each Linear layer
    @param use_dropout: whether to use Dropout after each ReLU
    """

    def __init__(self, input_dim=51, hidden_dim=128, embed_dim=32, seed=7,
                 use_bn=False, use_dropout=True):

        # You can easily tweak widths here without touching the rest of the code.
        widths: Tuple[int, int, int] = (hidden_dim, 64, 64)

        layers: List[object] = []
        in_features = input_dim
        layer_seed = seed

        for width in widths:
            layers.append(Linear(in_features, width, seed=layer_seed + 1))
            if use_bn:
                layers.append(BatchNorm1d(width))
            layers.append(ReLU())
            if use_dropout:
                layers.append(Dropout(p=0.2))
            in_features = width
            layer_seed += 1


        layers.append(Linear(in_features, embed_dim, seed=layer_seed + 1))

        self.net = Sequential(layers)



    def zero_grad(self):
        self.net.zero_grad()

    def backward_once(self, d_embed: List[List[float]]):
        return self.net.backward(d_embed)

    def step(self, lr: float):
        self.net.step(lr)

    def forward_once(self, features: List[List[float]], train=True):
        out = self.net.forward(features, train=train)

        return l2_normalize_rows(out)

    def train_batch(self, left_batch: List[List[float]], right_batch: List[List[float]], labels: List[int], lr=1e-3, margin=1.0):
        self.zero_grad()

        left_batch = l2_normalize_rows(left_batch)
        right_batch = l2_normalize_rows(right_batch)

        z1 = self.forward_once(left_batch)  # [B, D]
        z2 = self.forward_once(right_batch)  # [B, D]
        loss, d_left, d_right = siamese_contrastive_backward(z1, z2, labels, margin=margin)

        self.backward_once(d_right)
        self.backward_once(d_left)
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
