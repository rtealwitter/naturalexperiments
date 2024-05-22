def compute_direct_prediction(X, y, z, p, train_fn):
    boolean_control = z == 0
    pred_y0 = train_fn(
        X[boolean_control,:], y['y0'][boolean_control], X
    )
    boolean_treatment = z == 1
    pred_y1 = train_fn(
        X[boolean_treatment,:], y['y1'][boolean_treatment], X
    )
    return (pred_y1 - pred_y0).mean()