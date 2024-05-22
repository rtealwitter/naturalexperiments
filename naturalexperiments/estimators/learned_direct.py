def compute_learned_direct(X, y, z, p, train_fn):
    target = y['y0'] * (1-z) + y['y1'] * z
    pred = train_fn(X, target, X)

    return (
        (y['y1'] - pred) * z / .5 \
        - (y['y0'] - pred) * (1-z) / .5
    ).mean()
    