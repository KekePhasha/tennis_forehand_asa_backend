"""
Implements per-feature batch normalisation with running statistics and affine
parameters (gamma, beta). Designed for small educational MLPs in this project.

"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


class BatchNorm1d:
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.9) -> None:
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        # Learnable affine parameters
        self.gamma: List[float] = [1.0] * dim
        self.beta:  List[float] = [0.0] * dim

        # Running statistics (for eval / inference)
        self.running_mean: List[float] = [0.0] * dim
        self.running_var:  List[float] = [1.0] * dim

        # Gradients wrt parameters
        self.dgamma: List[float] = [0.0] * dim
        self.dbeta:  List[float] = [0.0] * dim

        # Cached intermediates for backward
        self._last_norm: List[List[float]] | None = None   # normalized pre-affine
        self._last_std_inv: List[float] | None = None      # per-feature 1/sqrt(var + eps)

    # --------------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------------
    def forward(self, x: Sequence[Sequence[float]], train: bool = True) -> List[List[float]]:
        if not x:
            return []  # empty batch guard

        batch_size = len(x)
        features = len(x[0])
        assert features == self.dim, f"Expected feature dim={self.dim}, got {features}"

        # -- Compute batch statistics or use running stats
        if train:
            means = [sum(row[j] for row in x) / batch_size for j in range(features)]
            vars_ = [
                sum((row[j] - means[j]) ** 2 for row in x) / batch_size
                for j in range(features)
            ]

            # Update exponential moving averages (in-place, per feature)
            self.running_mean = [
                self.momentum * rm + (1.0 - self.momentum) * m
                for rm, m in zip(self.running_mean, means)
            ]
            self.running_var = [
                self.momentum * rv + (1.0 - self.momentum) * v
                for rv, v in zip(self.running_var, vars_)
            ]
        else:
            means, vars_ = self.running_mean, self.running_var

        # -- Normalize with numerical stability
        std_inv = [1.0 / math.sqrt(v + self.eps) for v in vars_]

        # Cache for backward
        self._last_norm = []
        self._last_std_inv = std_inv

        # Normalize then apply affine transform
        out: List[List[float]] = []
        for row in x:
            normed = [(row[j] - means[j]) * std_inv[j] for j in range(features)]
            self._last_norm.append(normed)
            out.append([self.gamma[j] * normed[j] + self.beta[j] for j in range(features)])

        return out

    # --------------------------------------------------------------------------
    # Backward
    # --------------------------------------------------------------------------
    def backward(self, grad: Sequence[Sequence[float]]) -> List[List[float]]:
        if self._last_norm is None or self._last_std_inv is None:
            raise RuntimeError("BatchNorm1d.backward called before forward")

        batch_size = len(grad)
        if batch_size == 0:
            return []

        features = len(grad[0])
        assert features == self.dim, f"Expected feature dim={self.dim}, got {features}"

        # Accumulate parameter gradients: dβ = sum_i grad_i; dγ = sum_i grad_i * x_hat_i
        for j in range(features):
            sum_db = 0.0
            sum_dg = 0.0
            for i in range(batch_size):
                g_ij = grad[i][j]
                sum_db += g_ij
                sum_dg += g_ij * self._last_norm[i][j]
            self.dbeta[j]  += sum_db
            self.dgamma[j] += sum_dg

        # Gradient wrt inputs (per-feature closed form for batchnorm + affine)
        dX: List[List[float]] = [[0.0] * features for _ in range(batch_size)]

        for j in range(features):
            inv_std = self._last_std_inv[j]
            gamma_j = self.gamma[j]

            # Precompute sums over batch (per feature j)
            sum_g = 0.0       # Σ grad[i][j]
            sum_g_xhat = 0.0  # Σ grad[i][j] * x_hat[i][j]
            for i in range(batch_size):
                g = grad[i][j]
                x_hat = self._last_norm[i][j]
                sum_g += g
                sum_g_xhat += g * x_hat

            # Stable closed-form: dX = (γ / (B)) * inv_std * (B*g - Σg - x_hat*Σ(g*x_hat))
            # See: BN paper or common autodiff derivations.
            scale = (gamma_j * inv_std) / max(1, batch_size)
            for i in range(batch_size):
                g = grad[i][j]
                x_hat = self._last_norm[i][j]
                dX[i][j] = scale * (batch_size * g - sum_g - x_hat * sum_g_xhat)

        return dX

    # --------------------------------------------------------------------------
    # Optimizer hooks
    # --------------------------------------------------------------------------
    def step(self, lr: float) -> None:
        """SGD update for (gamma, beta)."""
        self.gamma = [g - lr * dg for g, dg in zip(self.gamma, self.dgamma)]
        self.beta  = [b - lr * db for b, db in zip(self.beta,  self.dbeta)]

    def zero_grad(self) -> None:
        """Reset accumulated gradients for (gamma, beta)."""
        self.dgamma = [0.0] * self.dim
        self.dbeta  = [0.0] * self.dim
