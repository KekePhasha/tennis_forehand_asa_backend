from typing import List
from models.linear.core.tensor import zeros2
class ReLU:
    def __call__(self, x: List[List[float]]) -> List[List[float]]:
        B, F = len(x), len(x[0])
        y = zeros2(B, F)
        for i in range(B):
            for j in range(F):
                y[i][j] = x[i][j] if x[i][j] > 0.0 else 0.0
        return y
