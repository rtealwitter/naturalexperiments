import sys
sys.path.insert(1, '../')
import naturalexperiments as ne
import numpy as np

dataset = 'RORCO'

# DR + Weighting
def dr_weight(X, y, z, p, train_fn):
    weights0 = p/(1-p)
    weights1 = (1-p)/p
    return ne.compute_doubly_robust(X, y, z, p, train_fn, weights0=weights0, weights1=weights1)

# DR + 2x Weighting
def dr_2x_weight(X, y, z, p, train_fn):
    weights0 = p/(1-p)**2
    weights1 = (1-p)/p**2
    return ne.compute_doubly_robust(X, y, z, p, train_fn, weights0=weights0, weights1=weights1)

# DR + Split
def dr_split(X, y, z, p, train_fn):
    weights0 = np.ones_like(p)
    weights1 = np.ones_like(p)
    return ne.compute_double_double(X, y, z, p, train_fn, weights0=weights0, weights1=weights1)

# DR + Split + Weight
def dr_split_weight(X, y, z, p, train_fn):
    weights0 = p/(1-p)
    weights1 = (1-p)/p
    return ne.compute_double_double(X, y, z, p, train_fn, weights0=weights0, weights1=weights1)

methods = {
    'Doubly Robust' : ne.compute_doubly_robust,
    'DR + Weighting' : dr_weight,
    'DR + 2x Weighting' : dr_2x_weight,
    'DR + Split' : dr_split,
    'DR + Split + Weight' : dr_split_weight,
    'Double-Double' : ne.compute_double_double,
}

variance, times = ne.compute_variance(methods, dataset, num_runs=0, folder='saved')

new_variance, new_times = {}, {}
for method in methods:
    new_variance[method] = variance[method]
    new_times[method] = times[method]
    
ne.benchmark_table(new_variance, new_times, print_md=True, print_latex=True, filename=f'tables/ablation_{dataset}.tex')