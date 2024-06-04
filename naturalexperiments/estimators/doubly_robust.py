import numpy as np

def compute_doubly_robust(X, y, z, p, train_fn, weights0=None, weights1=None):
    if weights0 is None:
        weights0 = np.ones_like(p)
    if weights1 is None:
        weights1 = np.ones_like(p)
    f1= train_fn(
        X, y['y1'], X, weights=weights1, 
    )
    f0= train_fn(
        X, y['y0'], X, weights=weights0, 
    )
    return (
        (y['y1'] - f1) * z / p + f1 \
        - (y['y0'] - f0) * (1-z) / (1-p) - f0
    ).mean()
