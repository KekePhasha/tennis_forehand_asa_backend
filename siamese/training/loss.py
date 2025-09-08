import math
from typing import List

def siamese_contrastive_backward(
    z1: List[List[float]], z2: List[List[float]],
    labels: List[int], margin: float = 1.0, eps: float = 1e-12
):
    """
    Given embeddings z1, z2 (shape [B, D]), compute:
      - L (scalar average loss)
      - dL/dz1, dL/dz2 (same shape as z1/z2)
    Loss per sample:
      y=0 (similar): 0.5 * ||z1 - z2||^2
      y=1 (dissimilar): 0.5 * max(0, m - ||z1 - z2||)^2
    """
    B, D = len(z1), len(z1[0])
    dz1 = [[0.0]*D for _ in range(B)]
    dz2 = [[0.0]*D for _ in range(B)]
    total = 0.0

    for i in range(B):
        # diff & distance
        diff = [z1[i][j] - z2[i][j] for j in range(D)]
        sqsum = sum(d*d for d in diff)
        d = math.sqrt(sqsum + eps)
        y = labels[i]

        if y == 0:
            # L = 0.5 * sum(diff^2)
            total += 0.5 * sqsum
            # grad wrt z1/z2 are just diff and -diff
            for j in range(D):
                dz1[i][j] = diff[j]
                dz2[i][j] = -diff[j]
        else:
            # L = 0.5 * max(0, m - d)^2
            if d < margin:
                total += 0.5 * (margin - d)**2
                # dL/dd = -(m - d)
                coeff = -(margin - d) / (d + eps)  # chain rule via d
                for j in range(D):
                    g = coeff * diff[j]  # dd/dz1 = diff / d
                    dz1[i][j] = g
                    dz2[i][j] = -g
            else:
                total += 0.0
                # grads remain 0

    return total / B, dz1, dz2
