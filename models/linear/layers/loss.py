import math
from typing import List

def siamese_contrastive_backward(
    z1: List[List[float]], z2: List[List[float]],
    labels: List[int], margin: float = 1.0, eps: float = 1e-12
):
    """
    Given embeddings z1, z2 (shape [batch_size, dim]), compute:
      - L (scalar average loss)
      - dL/dz1, dL/dz2 (same shape as z1/z2)
    Loss per sample:
      y=0 (similar): 0.5 * ||z1 - z2||^2
      y=1 (dissimilar): 0.5 * max(0, m - ||z1 - z2||)^2
    """
    batch_size, dim = len(z1), len(z1[0])
    dz1 = [[0.0]*dim for _ in range(batch_size)]
    dz2 = [[0.0]*dim for _ in range(batch_size)]

    total_loss = 0.0

    for i in range(batch_size):
        # 1) Pairwise difference and Euclidean dance
        diff_i = [z1[i][j] - z2[i][j] for j in range(dim)]
        sqsum = sum(d*d for d in diff_i)
        d = math.sqrt(sqsum + eps)
        y = labels[i]

        if y == 0:
            # L = 0.5 * sum(diff_i^2)
            total_loss += 0.5 * sqsum
            # grad wrt z1/z2 are just diff_i and -diff_i
            for j in range(dim):
                dz1[i][j] = diff_i[j]
                dz2[i][j] = -diff_i[j]
        else:
            # L = 0.5 * max(0, m - d)^2
            gap = margin - d
            if gap > 0.0:
                total_loss += 0.5 * gap * gap
                # dL/dd = -(m - d); chain rule via d(d)/d(z) = diff_i / d
                coeff = -(gap) / (d + eps)
                for j in range(dim):
                    g = coeff * diff_i[j]
                    dz1[i][j] = g
                    dz2[i][j] = -g
                # grads remain 0

    mean_loss = total_loss / max(1, batch_size)
    return mean_loss, dz1, dz2
