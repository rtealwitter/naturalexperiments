from econml.grf import CausalForest

def compute_causalforest(X, y, z, p, train_fn):
    cf = CausalForest()
    y_observed = (y['y1'] * z + y['y0'] * (1 - z)).values
    cf.fit(X, z, y_observed)
    treatment_effects = cf.predict(X)
    return treatment_effects.mean()