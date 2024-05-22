import numpy as np

def compute_propensity_stratification(X, y, z, p, train_fn, k=5):
    n = X.shape[0]
    estimates = []
    for i in range(k):
        idx = p.argsort()[i*n//k:(i+1)*n//k]
        y_idx = y.iloc[idx]
        z_idx = z[idx]
        if z_idx.sum() == 0 or z_idx.sum() == len(z_idx):
            continue
        estimates.append(
            y_idx['y1'][z_idx == 1].mean()  \
            - y_idx['y0'][z_idx == 0].mean()
        )
    return np.mean(estimates)
        
