from typing import List
import math

def pairwise_distance(f1: List[List[float]], f2: List[List[float]]) -> List[float]:
    assert len(f1) == len(f2)
    B, F = len(f1), len(f1[0])
    out = [0.0]*B
    for i in range(B):
        s = 0.0
        for j in range(F):
            d = f1[i][j] - f2[i][j]
            s += d*d
        out[i] = math.sqrt(s)
    return out
