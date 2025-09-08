import math
import random
from typing import List

from models.linear.core.tensor import zeros2


class Linear:
    """
    y = x @ W^T + b
    x: [B, in_f], W: [out_f, in_f], b: [out_f]
    """
    def __init__(self, in_f, out_f, seed=42):
        # Xavier init
        random.seed(seed)
        lim = 0.5 * math.sqrt(6.0/(in_f+out_f))
        self.W = [[random.uniform(-lim, lim) for _ in range(in_f)] for _ in range(out_f)]
        self.b = [0.0 for _ in range(out_f)]
        # grads
        self.gW = [[0.0]*in_f for _ in range(out_f)]
        self.gb = [0.0]*out_f
        self._cache_stack = []

    def zero_grad(self):
        out_f, in_f = len(self.gW), len(self.gW[0])
        for o in range(out_f):
            self.gb[o] = 0.0
            row_gW = self.gW[o]
            for k in range(in_f):
                row_gW[k] = 0.0

        self._cache_stack = []

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        self._cache_stack.append(x)  # cache for backward
        B, in_f = len(x), len(x[0])
        out_f = len(self.W)
        y = zeros2(B, out_f)
        for i in range(B):
            for o in range(out_f):
                s = self.b[o]
                rowW = self.W[o]
                for k in range(in_f):
                    s += x[i][k] * rowW[k]
                y[i][o] = s
        return y

    def backward(self, dY: List[List[float]]) -> List[List[float]]:
        """
        dY: [B, out_f]
        returns dX: [B, in_f]; accumulates gW, gb
        """
        x = self._cache_stack.pop()
        B, in_f = len(x), len(x[0])
        out_f = len(self.W)
        dX = zeros2(B, in_f)
        for i in range(B):
            xi = x[i]
            for o in range(out_f):
                g = dY[i][o]
                self.gb[o] += g
                rowW = self.W[o]
                row_gW = self.gW[o]
                for k in range(in_f):
                    row_gW[k] += g * xi[k]
                    dX[i][k]  += g * rowW[k]
        return dX

    def step(self, lr: float):
        out_f, in_f = len(self.W), len(self.W[0])
        for o in range(out_f):
            self.b[o] -= lr * self.gb[o]
            rowW = self.W[o]
            row_gW = self.gW[o]
            for k in range(in_f):
                rowW[k] -= lr * row_gW[k]

class ReLU:
    def __init__(self):
        self._mask_stack = []  # stack

    def zero_grad(self):
        self._mask_stack = []
        pass

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        B, F = len(x), len(x[0])
        y = zeros2(B, F)
        mask = zeros2(B, F)
        for i in range(B):
            for j in range(F):
                v = x[i][j]
                if v > 0.0:
                    y[i][j] = v
                    mask[i][j] = 1.0
                else:
                    y[i][j] = 0.0
                    mask[i][j] = 0.0

        self._mask_stack.append(mask)
        return y

    def backward(self, dY: List[List[float]]) -> List[List[float]]:
        mask = self._mask_stack.pop()
        B, F = len(dY), len(dY[0])
        dX = zeros2(B, F)
        for i in range(B):
            for j in range(F):
                dX[i][j] = dY[i][j] * mask[i][j]
        return dX

    def step(self, lr: float):
        pass  # no params

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.cache_inputs = []  # optional, not used here

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    def forward(self, x: List[List[float]]) -> List[List[float]]:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dOut: List[List[float]]) -> List[List[float]]:
        grad = dOut
        # backprop in reverse
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, lr: float):
        for layer in self.layers:
            layer.step(lr)
