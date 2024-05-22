def compute_horvitz_thompson(X, y, z, p, train_fn):
    return (
        y['y1'] * z / p \
        - y['y0'] * (1-z) / (1-p)
    ).mean()
