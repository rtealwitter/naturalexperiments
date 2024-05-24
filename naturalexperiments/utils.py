import numpy as np
import sklearn
import pandas as pd

def center_distance_matrix(x):
    x = np.expand_dims(x, axis=1)
    X = sklearn.metrics.pairwise_distances(x)
    n = X.shape[0]
    row_means = np.tile(X.mean(axis=0), (n,1))
    col_means = np.tile(X.mean(axis=1), (n,1)).T
    overall_mean = X.mean()
    centered_X = X - row_means - col_means + overall_mean
    return centered_X

def compute_distance_correlation(x, y, limit=10000):
    if len(x) > limit:
        print(f'Warning: n={len(x)}, subsampling to n={limit} for distance correlation computation.')
        x = x[np.random.choice(len(x), limit)]
        y = y[np.random.choice(len(y), limit)]
    # Compute distance correlation between x and y
    # https://en.wikipedia.org/wiki/Distance_correlation
    centered_X = center_distance_matrix(x)
    centered_Y = center_distance_matrix(y)
    n = len(x)
    distance_covariance = np.sum(centered_X * centered_Y) / (n**2)
    distance_variance_X = np.sum(centered_X**2) / (n**2)
    distance_variance_Y = np.sum(centered_Y**2) / (n**2)

    return distance_covariance / np.sqrt(distance_variance_X * distance_variance_Y)

def compute_cross_entropy(p, treatment):
    return - np.mean(treatment * np.log(p) + (1-treatment) * np.log(1-p))

def sig_round(x, precision=3):
    return np.format_float_positional(x, precision=precision, unique=False, fractional=False, trim='k')

def biased_treatment_effect(x, scaling=1):
    # No effect on first half
    # Second half has effect of sqrt(x) * scaling
    adjustment = .5 - np.sqrt(.5) * scaling
    line = x * (x<.5).astype(int) + (x>=.5).astype(int) * (scaling * np.sqrt(x) + adjustment)
    return line

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Build problem where treatment and outcome are correlated
def build_synthetic_outcomes(X, scale=True):
    if scale:
        X = sklearn.preprocessing.StandardScaler().fit(X).transform(X)

    b = np.random.normal(size=X.shape[1])
    important_variable = X @ b
    p = sigmoid(important_variable)
    p = np.clip(p, 0.01, .99) # regularize

    y = pd.DataFrame({
        'y0' : 1-p,
        'y1' : 1-biased_treatment_effect(p)
    }, dtype=float)
    return X, y, p