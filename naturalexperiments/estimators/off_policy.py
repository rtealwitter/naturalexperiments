import numpy as np

# From equations 7a, 7b, and 11 in: Off-policy estimation of linear functionals: Non-asymptotic theory for semi-parametric efficiency
# https://arxiv.org/pdf/2209.13075.pdf

def compute_off_policy(X, y, z, p, train_fn):
    n = X.shape[0]
    # Random split
    S1 = np.random.choice(n, n//2, replace=False)
    S2 = np.array([i for i in range(n) if i not in S1])

    g = 2 * z - 1
    # Train models
    X_with_treatment = np.concatenate([X, z.reshape(-1,1)], axis=1)
    combined_outcomes = y['y1'] * z + y['y0'] * (1-z)
    pi = z * p + (1-z) * (1-p)
    model1 = train_fn(
        X_with_treatment[S1], combined_outcomes[S1], weights=(1/(pi[S1])**2),
        return_model=True
    )
    model2 = train_fn(
        X_with_treatment[S2], combined_outcomes[S2], weights=(1/(pi[S2])**2),
        return_model=True
    )
    pred1 = model1(X_with_treatment[S2])
    pred2 = model2(X_with_treatment[S1])
    X_with_1 = np.concatenate([X, (np.ones(n)).reshape(-1,1)], axis=1)
    X_with_0 = np.concatenate([X, (np.zeros(n)).reshape(-1,1)], axis=1)
    subtract1 = model1(X_with_1[S2]) - model1(X_with_0[S2])
    subtract2 = model2(X_with_1[S1]) - model2(X_with_0[S1])
    
    f1 = g[S2] * pred1 / pi[S2] - subtract1 / 2
    f2 = g[S1] * pred2 / pi[S1] - subtract2 / 2
    return (g[S1] / pi[S1] * combined_outcomes[S1] - f2).mean()/2 \
        + (g[S2] / pi[S2] * combined_outcomes[S2] - f1).mean()/2