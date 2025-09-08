import random, math
from typing import List

def zeros(n: int) -> List[float]:
    return [0.0]*n

def zeros2(r: int, c: int) -> List[List[float]]:
    return [[0.0]*c for _ in range(r)]

def glorot_limit(f_in: int, f_out: int) -> float:
    return math.sqrt(6.0/(f_in + f_out))

def seeded_rng(seed: int) -> random.Random:
    rng = random.Random()
    rng.seed(seed)
    return rng
