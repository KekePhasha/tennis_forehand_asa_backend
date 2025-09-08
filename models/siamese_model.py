from typing import List

from siamese.core.activation import ReLU
from siamese.core.container import Sequential
from siamese.core.distance import pairwise_distance
from siamese.core.layers import Linear, BatchNorm1d, Dropout


class SiameseModel:
    def __init__(self, input_dim=51, training=True, seed=7,
                 h1=128, h2=64, embed=32, pdrop=0.2):
        self.training = training
        self.embedding_net = Sequential([
            Linear(input_dim, h1, seed=seed+1),
            BatchNorm1d(h1),
            ReLU(),
            Dropout(pdrop, seed=seed+2),
            Linear(h1, h2, seed=seed+3),
            BatchNorm1d(h2),
            ReLU(),
            Linear(h2, embed, seed=seed+4)
        ])

    def forward_once(self, x: List[List[float]]):
        return self.embedding_net(x, training=self.training)

    def forward(self, x1: List[List[float]], x2: List[List[float]]):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        return pairwise_distance(f1, f2)
