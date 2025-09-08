from typing import List
import math

from siamese.core.tensor import glorot_limit, seeded_rng, zeros2


class Linear:
    def __init__(self, in_f: int, out_f: int, seed: int = 42):
        self.W = []
        self.b = [0.0]*out_f
        lim = glorot_limit(in_f, out_f)
        rng = seeded_rng(seed)
        for _ in range(out_f):
            self.W.append([rng.uniform(-lim, lim) for _ in range(in_f)])

    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        B, F = len(x), len(x[0])
        out_f = len(self.W)
        y = zeros2(B, out_f)
        for i in range(B):
            for o in range(out_f):
                s = self.b[o]
                rowW = self.W[o]
                for k in range(F):
                    s += x[i][k] * rowW[k]
                y[i][o] = s
        return y

class BatchNorm1d:
    def __init__(self, num_f: int, eps=1e-5, momentum=0.1):
        self.gamma = [1.0]*num_f
        self.beta  = [0.0]*num_f
        self.eps = eps
        self.momentum = momentum
        self.running_mean = [0.0]*num_f
        self.running_var  = [1.0]*num_f

    def __call__(self, x: List[List[float]], training=True) -> List[List[float]]:
        B, F = len(x), len(x[0])
        y = zeros2(B, F)
        if training:
            mean = [sum(x[i][j] for i in range(B))/B for j in range(F)]
            var  = []
            for j in range(F):
                s = 0.0
                for i in range(B):
                    d = x[i][j] - mean[j]
                    s += d*d
                var.append(s/B)
            # normalize
            for i in range(B):
                for j in range(F):
                    xhat = (x[i][j] - mean[j]) / math.sqrt(var[j] + self.eps)
                    y[i][j] = self.gamma[j]*xhat + self.beta[j]
            # update running
            for j in range(F):
                self.running_mean[j] = (1-self.momentum)*self.running_mean[j] + self.momentum*mean[j]
                self.running_var[j]  = (1-self.momentum)*self.running_var[j]  + self.momentum*var[j]
        else:
            for i in range(B):
                for j in range(F):
                    xhat = (x[i][j]-self.running_mean[j]) / math.sqrt(self.running_var[j]+self.eps)
                    y[i][j] = self.gamma[j]*xhat + self.beta[j]
        return y

class Dropout:
    def __init__(self, p=0.5, seed=123):
        assert 0.0 <= p < 1.0
        self.p = p
        self.keep = 1.0 - p
        self.rng = seeded_rng(seed)

    def __call__(self, x: List[List[float]], training=True) -> List[List[float]]:
        if not training or self.p == 0.0: return x
        B, F = len(x), len(x[0])
        y = zeros2(B, F)
        scale = 1.0/self.keep
        for i in range(B):
            for j in range(F):
                if self.rng.random() < self.keep:
                    y[i][j] = x[i][j]*scale
                else:
                    y[i][j] = 0.0
        return y
