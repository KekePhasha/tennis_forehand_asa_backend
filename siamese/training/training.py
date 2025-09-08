from typing import List
from siamese.training.layers import Linear, ReLU, Sequential
from siamese.training.loss import siamese_contrastive_backward


class SiameseModelTrainable:
    """
    Trainable pure-Python Siamese (no BN/Dropout for simplicity):
      51 -> 64 -> 32
    """

    def __init__(self, input_dim=51, h=64, embed=32, seed=7):
        self.net = Sequential([
            Linear(input_dim, h, seed=seed + 1),
            ReLU(),
            Linear(h, embed, seed=seed + 2),
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
