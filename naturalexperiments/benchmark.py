from tqdm import tqdm
import time
from .model import estimate_propensity, train
from .data import dataloaders
from .utils import compute_cross_entropy, compute_distance_correlation
import numpy as np
import sklearn.preprocessing
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# # # VARIANCE BENCHMARK TABLE # # #

def benchmark_table(variance, times, print_md=True, print_latex=False):
    table = []
    for method in variance:
        # Sometimes NaNs from one of the CATENet methods
        # Remove NaNs from variance[method]
        variance[method] = [x for x in variance[method] if not np.isnan(x)]
        mean = round(np.mean(variance[method]))
        median = round(np.median(variance[method]))
        upper = round(np.percentile(variance[method], 75))
        lower = round(np.percentile(variance[method], 25))
        times_mean = round(np.mean(times[method]))
        table.append([method, mean, lower, median, upper, times_mean])

    if print_md:
        print(tabulate(table, headers=['Method', 'Mean', '1st Quartile', '2nd Quartile', '3rd Quartile', 'Time (s)'], tablefmt="github"))    

    cols = []
    for i in range(len(table[0])-1):
        vals = [row[i+1] for row in table]
        cols += [sorted(vals)]
    for row in table:
        print_row = [row[0]]
        for idx in range(1, len(row)):
            if row[idx] == cols[idx-1][0]:
                print_row.append(r'\textbf{'+row[idx]+'}')
            elif row[idx] == cols[idx-1][1]:
                print_row.append(r'\textit{\textbf{'+row[idx]+'}}')
            elif row[idx] == cols[idx-1][2]:
                print_row.append(r'\underline{\textbf{'+row[idx]+'}}')
            else:
                print_row.append(row[idx])
        if print_latex:
            print(' & '.join(print_row) + '\\\\ \hline')

# # # VARIANCE PLOTS # # #

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('solid',                 (0, ())),
     ('loosely dashdashdotted', (0, (3, 10, 3, 10, 1, 10))),
]
colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', 'red', 'black', 'blue', 'orange', 'purple', 'teal', 'green']

def plot_estimates(run_estimates, xlabel, figure_name, folder, show=False, save=True, unselect_keys=[], xlogscale=False, ignore_nan=True):
    i = 0
    # Sort estimators by median at start
    method_names = list(run_estimates.keys())
    first_key = list(run_estimates[method_names[0]].keys())[0]
    values = [np.median(run_estimates[method_name][first_key]) for method_name in method_names]
    # Sort method_names by values
    method_names = [x for _, x in sorted(zip(values, method_names))]
    for method_name in reversed(method_names):
        if method_name not in unselect_keys:
            medians = []
            upper_quantile = []
            lower_quantile = []
            keys = []
            for key in sorted(run_estimates[method_name].keys()):
                keys += [key]
                # ignore nan
                current_estimates = run_estimates[method_name][key]#[:min_num]
                if ignore_nan:
                    current_estimates = [x for x in current_estimates if not np.isnan(x)]
                medians.append(np.median(current_estimates))
                upper_quantile.append(np.quantile(current_estimates, .75))
                lower_quantile.append(np.quantile(current_estimates, .25))
            plt.plot(keys, medians, label=method_name, linestyle=linestyle_tuple[i][1], color=colors[i])
            plt.fill_between(keys, lower_quantile, upper_quantile, color=colors[i], alpha=.2)
            i+= 1
        
    plt.ylabel('Squared Error')
    plt.title(figure_name.replace('RORCO ', ''))
    plt.yscale('log')
    if xlogscale:
        plt.xscale('log')
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f'{folder}{figure_name}.pdf', dpi=500)
    if show:
        plt.show()

# # # SYNTHETIC OUTCOMES # # #

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

def wrap_dataloader(dataset, num):
    if dataset == 'IHDP':
        X, y, z = dataloaders['IHDP'](num % 10 + 1)
    if dataset == 'NEWS':
        X, y, z = dataloaders['NEWS'](num % 50 + 1)
    else: # For datasets where outcomes only known for treatment or control but not both
        X, _, _ = dataloaders[dataset]()
        X, y, p = build_synthetic_outcomes(X)
        uniform = np.random.random_sample(X.shape[0])
        z = (uniform < p).astype(int)
    return X, y, z

# # # VARIANCE BENCHMARK # # #

def compute_variance(methods, dataset, num_runs=10, train_fn=train, variance=None, times=None):
    if variance is None: 
        variance = {method: [] for method in methods}
        times = {method: [] for method in methods}

    for method in methods:
        if method not in variance:
            variance[method] = []
            times[method] = []    

    for num in tqdm(range(num_runs)): 
        X, y, z = wrap_dataloader(dataset, num)
        p_estimated = estimate_propensity(X, z)
        true_effect = (y['y1'] - y['y0']).mean()
 
        for method, get_estimate in methods.items():
            if len(variance[method]) < num_runs:
                time_start = time.time()
                estimate = get_estimate(X, y, z, p_estimated, train_fn)
                square_difference = (estimate - true_effect)**2
                run_time = time.time() - time_start            
                variance[method].append(square_difference) 
                times[method].append(run_time)
                print(f'{method} variance: {square_difference} time: {run_time}')

    return variance, times

def compute_variance_by_n(methods, dataset, ns, num_runs=10, train_fn=train, variance=None):
    if variance is None: 
        variance = {}

    for method in methods:
        if method not in variance:
            variance[method] = {n : [] for n in ns} 
        for n in ns:
            if n not in variance[method]:
                variance[method][n] = []

    for num in tqdm(range(num_runs)): 
        for i, n in enumerate(ns):
            X, y, z = wrap_dataloader(dataset, num + i*len(ns))
            sample_indices = np.random.choice(X.shape[0], n, replace=False)
            X, y, z = X[sample_indices], y.iloc[sample_indices], z[sample_indices]
            # Reset index for y
            y = y.reset_index(drop=True)

            p_estimated = estimate_propensity(X, z)
            true_effect = (y['y1'] - y['y0']).mean()
    
            for method, get_estimate in methods.items():
                if len(variance[method][n]) < num_runs:
                    estimate = get_estimate(X, y, z, p_estimated, train_fn)
                    square_difference = (estimate - true_effect)**2
                    variance[method][n].append(square_difference) 

    return variance

def compute_variance_by_entropy(methods, dataset, noise_levels=[0, .2, .3, .4, .5], increment=.1, num_runs=10, train_fn = train, variance=None):
    if variance is None: 
        variance = {}

    for method in methods:
        if method not in variance:
            variance[method] = {}

    for _ in tqdm(range(num_runs)): 
        for noise_level in noise_levels:
            X, _, _ = dataloaders[dataset]()

            # Need to know propensity to add noise
            X, y, p = build_synthetic_outcomes(X)
            uniform = np.random.random_sample(X.shape[0])
            z = (uniform < p).astype(int)

            noised_p = p + np.random.normal(0, noise_level, size=p.shape)
            noised_p = np.clip(noised_p, 0.01, .99) # regularize

            cross_entropy = compute_cross_entropy(noised_p, z)

            cross_entropy = np.round(cross_entropy / increment) * increment

            true_effect = (y['y1'] - y['y0']).mean()
    
            for method, get_estimate in methods.items():
                if cross_entropy not in variance[method]:
                    variance[method][cross_entropy] = []
                if len(variance[method][cross_entropy]) < num_runs:
                    estimate = get_estimate(X, y, z, noised_p, train_fn)
                    square_difference = (estimate - true_effect)**2
                    variance[method][cross_entropy].append(square_difference) 

    return variance

def compute_variance_by_correlation(methods, dataset, alphas=[0, .15, .2, .25, .35, .5], increment=.1, num_runs=10, train_fn = train, variance=None):
    if variance is None: 
        variance = {}

    for method in methods:
        if method not in variance:
            variance[method] = {}

    for _ in tqdm(range(num_runs)): 
        for alpha in alphas:
            X, _, _ = dataloaders[dataset]()

            b = np.random.normal(size=X.shape[1])
            important_variable = X @ b
            p = sigmoid(important_variable)
            p = np.clip(p, 0.01, .99) # regularize

            random_vector = np.random.normal(size=X.shape[0])
            
            y = pd.DataFrame({
                'y0' : (1-p) * (1-alpha) + random_vector * alpha,
                'y1' : 1-biased_treatment_effect(p) * (1-alpha) + random_vector * alpha
            }, dtype=float)

            uniform = np.random.random_sample(X.shape[0])
            z = (uniform < p).astype(int)

            correlation = (
                compute_distance_correlation(y['y0'][z==0], p[z==0]) + \
                compute_distance_correlation(y['y1'][z==1], p[z==1])
            )

            correlation = np.round(correlation / increment) * increment

            p_estimated = estimate_propensity(X, z)

            true_effect = (y['y1'] - y['y0']).mean()
    
            for method, get_estimate in methods.items():
                if correlation not in variance[method]:
                    variance[method][correlation] = []
                if len(variance[method][correlation]) < num_runs:
                    estimate = get_estimate(X, y, z, p_estimated, train_fn)
                    square_difference = (estimate - true_effect)**2
                    variance[method][correlation].append(square_difference) 

    return variance