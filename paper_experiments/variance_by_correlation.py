import sys
sys.path.append('../')
import naturalexperiment5 as ne
import pickle
import os

dataset = 'RORCO'

use_methods = ['FlexTENet', 'TNet', 'TARNet', 'RANet', 'Double-Double']

methods = {method: ne.methods[method] for method in use_methods}

variance = ne.compute_variance_by_correlation(methods, dataset, num_runs=0, folder='saved')


ne.plot_estimates(
    variance, 
    xlabel = r'Distance Correlation',
    figure_name=rf'{dataset}: Squared Error by Correlation',
    folder = '../images/',
)