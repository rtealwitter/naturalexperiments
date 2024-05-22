import numpy as np

# From page 68 in Targeted Maximum Likelihood Estimation for Causal Inference in Observational Studies 
# https://academic.oup.com/aje/article/185/1/65/2662306

def compute_TMLE(X, y, z, p, train_fn):
    boolean_control = z == 0
    predY0 = train_fn(
        X[boolean_control,:],
        y['y0'][boolean_control], X
    )
    boolean_treatment = z == 1
    predY1 = train_fn(
        X[boolean_treatment,:],
        y['y1'][boolean_treatment], X
    )
    Ha = z / p - (1-z) / (1-p)
    # Learn epsilon in E[Y|A,X] = E[predY|A,X] + epsilon * Ha
    Ha = np.expand_dims(Ha, axis=1)
    predY = z * predY1 + (1-z) * predY0
    trueY = z * y['y1'] + (1-z) * y['y0']

    epsilon = train_fn(Ha, trueY - predY, Ha)
    H1 = 1/ p
    H0 = -1 / (1-p)
    starY1 = predY1 + epsilon * H1
    starY0 = predY0 + epsilon * H0
    return (starY1 - starY0).mean()