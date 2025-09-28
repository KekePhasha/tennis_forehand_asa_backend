import random

class Dropout:
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.mask = None
        if seed is not None:
            random.seed(seed)

    def forward(self, x, train=True):
        if not train or self.p == 0.0:
            return x
        self.mask = [[0 if random.random() < self.p else 1 for _ in row] for row in x]
        return [[val * m for val, m in zip(row, mask_row)] for row, mask_row in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [[g * m for g, m in zip(row, mask_row)] for row, mask_row in zip(grad, self.mask)]

    def step(self, lr):  # no weights to update
        pass

    def zero_grad(self):  # no grads to reset
        pass
