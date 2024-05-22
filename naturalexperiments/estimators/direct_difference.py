def compute_direct_difference(X, y, z, p, train_fn):
    return (
        y['y1'] * z / .5 \
        - y['y0'] * (1-z) / .5
    ).mean()