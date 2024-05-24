import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from .utils import biased_treatment_effect, compute_cross_entropy, compute_distance_correlation, sig_round
from .model import estimate_propensity

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

def plot_propensity(propensity, dataset, filename=None): 
    plt.hist(propensity)
    plt.ylabel('Frequency')
    plt.xlabel('Propensity Score')
    plt.title('Propensity Score Histogram: ' + dataset)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_propensity_calibration(propensity, z, dataset, filename=None):
    num_buckets = 10
    #percentiles = np.percentile(propensity, np.linspace(0, 100, num_buckets+1))
    min_val = np.min(propensity)
    max_val = np.max(propensity)
    percentiles = np.linspace(min_val, max_val, num_buckets+1)
    mean_treatments = np.zeros(num_buckets)
    mean_propensities = np.zeros(num_buckets)
    for i in range(num_buckets):
        bucket = (propensity >= percentiles[i]) & (propensity < percentiles[i+1])
        mean_treatments[i] = np.mean(z[bucket])
        mean_propensities[i] = np.mean(propensity[bucket])
    plt.plot(mean_propensities, mean_treatments, 'o-', color='teal', linewidth=3)
    plt.xlabel('Mean Propensity')
    plt.ylabel('Mean Treatment')
    plt.title('Propensity Calibration Plot: ' + dataset)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_outcome_by_propensity(propensity, y, z, dataset, filename=None):
    num_buckets = 5
    #percentiles = np.percentile(propensity, np.linspace(0, 100, num_buckets+1))
    min_val = np.min(propensity)
    max_val = np.max(propensity)
    percentiles = np.linspace(min_val, max_val, num_buckets+1)
    plot_percentiles = (percentiles[1:] + percentiles[:-1]) / 2
    treatment_outcomes, treatment_sizes = [], []
    control_outcomes, control_sizes = [], []
    total_size = len(propensity)
    for i in range(num_buckets):
        bucket = (propensity >= percentiles[i]) & (propensity < percentiles[i+1])
        treatment_in_bucket = np.sum(z * bucket)
        treatment_outcomes.append(np.sum((y['y1']*z)[bucket]) / treatment_in_bucket)
        treatment_sizes.append(treatment_in_bucket/total_size * 500)
        control_in_bucket = np.sum((1-z) * bucket)
        control_outcomes.append(np.sum((y['y0']*(1-z))[bucket]) / control_in_bucket)
        control_sizes.append(control_in_bucket / total_size * 500)
    plt.scatter(plot_percentiles, treatment_outcomes, s=treatment_sizes, color='teal')
    plt.plot(plot_percentiles, treatment_outcomes, label='Treatment', color='teal', linewidth=3)

    plt.scatter(plot_percentiles, control_outcomes, s=control_sizes, color='purple')
    plt.plot(plot_percentiles, control_outcomes, label='Control', color='purple', linestyle='dashed', linewidth=3)

    plt.xlabel('Propensity (Likelihood of Receiving Treatment)')
    plt.ylabel('Outcome')
    plt.title('Outcome by Propensity: ' + dataset)

    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_synthetic_outcome_by_propensity(filename=None):
    p = np.linspace(0,1)
    plt.plot(p,1-biased_treatment_effect(p), label='Treatment', color='teal', linewidth=3)
    plt.plot(p,1-p, label='Control', color='purple', linewidth=3, linestyle='dashed')
    plt.xlabel('Propensity (Likelihood of Receiving Treatment)')
    plt.ylabel('Outcome')
    plt.legend()
    plt.title('Outcome vs Propensity')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def wrap_estimate_propensity(X, z):
    if z is not None:
        propensity = estimate_propensity(X, z)

    else:
        beta = np.random.normal(size=X.shape[1])
        var = X @ beta
        propensity = (var - var.min() ) / (var.max() - var.min())
        propensity = np.clip(propensity, 0.01,.99)
        # Sample z from a Bernoulli distribution with propensity
        z = np.random.binomial(1, propensity)
    
    return propensity, z

def plot_all_data(dataloaders, folder=''):
    for dataname, dataloader in dataloaders.items(): 
        X, y, z = dataloader()

        propensity, z = wrap_estimate_propensity(X, z)

        plot_propensity(propensity, dataname, filename=folder+'propensity_hist_'+dataname+'.pdf')
        plot_propensity_calibration(propensity, z, dataname, filename=folder+'propensity_calibration_'+dataname+'.pdf')
        plot_outcome_by_propensity(propensity, y, z, dataname, filename=folder+'outcome_by_propensity_'+dataname+'.pdf')

def dataset_table(dataloaders, print_md=True, print_latex=False):
    table = []
    for dataname, dataloader in dataloaders.items():
        X, y, z = dataloader()

        propensity, z = wrap_estimate_propensity(X, z)
        
        # Number of observations
        n = X.shape[0]
        # Number of variables
        m = X.shape[1]
        # Percent treated 
        rate_treated = sig_round(np.mean(z) * 100)
        # Cross entropy of propensity score and z
        cross_entropy = sig_round(compute_cross_entropy(propensity, z))
        # Correlation between propensity and y1
        corr_y1 = sig_round(np.corrcoef(propensity[z==1], y['y1'][z==1])[0,1])
        #corr_y1 = sig_round(compute_distance_correlation(propensity[z==1], y['y1'][z==1].values))
        # Correlation between propensity and y0
        corr_y0 = sig_round(np.corrcoef(propensity[z==0], y['y0'][z==0])[0,1])
        #corr_y0 = sig_round(compute_distance_correlation(propensity[z==0], y['y0'][z==0].values))

        table += [[dataname, n, m, rate_treated, cross_entropy, corr_y1, corr_y0]]

    headers = ['Dataset', 'Size', 'Variables', '% Treated', 'Cross Entropy', 'Corr(y1, p)', 'Corr(y0, p)']

    if print_md:
        print(tabulate(table, headers=headers))

    if print_latex:
        bold_headers = ['\\textbf{' + h + '}' for h in headers]
        print(' & '.join(bold_headers) + r'\\ \hline')
        for row in table:
            print(' & '.join([str(x) for x in row]) + r'\\ \hline')
