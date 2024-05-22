import numpy as np
import torch
import pandas as pd

def compute_double_double(X, y, z, p, train_fn, weights0=None, weights1=None, S1=None):
    if weights0 is None or weights1 is None:
        weights0 = (p/(1-p)**2)
        weights1 = ((1-p)/p**2)
    n = X.shape[0]
    # Random split
    if S1 is None:
        #S1 = np.random.choice(n, n//2, replace=False)
        S1 = [i for i in range(n//2)]
    S2 = np.array([i for i in range(n) if i not in S1])

    f1_S2 = train_fn(
        X[S1], y['y1'][S1], X[S2], weights=weights1[S1], 
    )
    f1_S1 = train_fn(
        X[S2], y['y1'][S2], X[S1], weights=weights1[S2], 
    )
    f0_S2 = train_fn(
        X[S1], y['y0'][S1], X[S2], weights=weights0[S1], 
    )
    f0_S1 = train_fn(
        X[S2], y['y0'][S2], X[S1], weights=weights0[S2], 
    )

    yhat_S2 = (1-p[S2]) * f1_S2 + p[S2] * f0_S2
    yhat_S1 = (1-p[S1]) * f1_S1 + p[S1] * f0_S1

    estimate_S1 = (
        (y['y1'][S1] - yhat_S1) * z[S1] / p[S1] \
        - (y['y0'][S1] - yhat_S1) * (1-z[S1]) / (1-p[S1])
    ).mean()

    estimate_S2 = (
        (y['y1'][S2] - yhat_S2) * z[S2] / p[S2] \
        - (y['y0'][S2] - yhat_S2) * (1-z[S2]) / (1-p[S2])
    ).mean()

    return (estimate_S1 + estimate_S2) / 2

def compute_dd_wo_weighting(X, y, z, p, train_fn):
    weights0 = np.ones_like(p)
    weights1 = np.ones_like(p)
    return compute_double_double(X, y, z, p, train_fn, weights0, weights1)

def compute_dd_wo_2x_weighting(X, y, z, p, train_fn):
    weights0 = p/(1-p)
    weights1 = (1-p)/p
    return compute_double_double(X, y, z, p, train_fn, weights0, weights1)

def compute_dd_wo_stages(X, y, z, p, train_fn):
    n = X.shape[0]
    S1 = list(range(n))
    return compute_double_double(X, y, z, p, train_fn, S1=S1)
    
