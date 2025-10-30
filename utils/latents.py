# utils/latents.py
import numpy as np

def pca_2d(X: np.ndarray):
    """
    X: (N, D) array (rows = samples). Returns (N, 2) coords normalized to [0,1].
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError("pca_2d expects (N,D) with N>=1")

    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # handle N==1 gracefully -> all zeros
    if X.shape[0] == 1:
        Z = np.zeros((1, 2), dtype=np.float32)
    else:
        # SVD on centered data
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # project onto first 2 components (pad if only 1 available)
        W = Vt[:2].T  # (D,2) or (D,1) if rank=1
        Z = Xc @ W
        if Z.shape[1] == 1:
            Z = np.concatenate([Z, np.zeros_like(Z)], axis=1)

    # min-max normalize to [0,1] for easy plotting
    mins = Z.min(axis=0, keepdims=True)
    maxs = Z.max(axis=0, keepdims=True)
    rng = np.maximum(maxs - mins, 1e-6)
    Z01 = (Z - mins) / rng
    return Z01
