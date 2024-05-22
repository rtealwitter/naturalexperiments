# As described in Equation 1 of Bias-Reduced Doubly Robust Estimation
# https://www.tandfonline.com/doi/full/10.1080/01621459.2014.958155

def compute_doubly_robust(X, y, z, p, train_fn):
    boolean_control = z == 0
    pred_y0 = train_fn(
        X[boolean_control,:], y['y0'][boolean_control], X
    )
    boolean_treatment = z == 1
    pred_y1 = train_fn(
        X[boolean_treatment,:], y['y1'][boolean_treatment], X
    )
    treatment_term = z * y['y1'] / p - (z - p) * pred_y1 / p
    control_term = (1 - z) * y['y0'] / (1 - p) - ((1 - z) - (1 - p)) * pred_y0 / (1 - p)
    return (treatment_term - control_term).mean()

