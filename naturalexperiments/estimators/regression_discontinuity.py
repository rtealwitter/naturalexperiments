import numpy as np

def nonzero_mean(x):
    return np.mean(x[x != 0])

def compute_regression_discontinuity(X, y, z, p, train_fn, width=.1):
    in_interval = (p > .5 - width) & (p < .5 + width)
    avg_y0_near_threshold = nonzero_mean(y['y0'][in_interval] * (1-z[in_interval]))

    avg_y1_near_threshold = nonzero_mean(y['y1'][in_interval] * z[in_interval])

    return (
        avg_y1_near_threshold - avg_y0_near_threshold
    )

