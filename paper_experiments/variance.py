import sys
sys.path.append('../')
import naturalexperiment5 as ne
import numpy as np


dataset = 'ACIC 2016'
dataset = 'ACIC 2017'
dataset = 'IHDP'
dataset = 'JOBS'
dataset = 'NEWS'
dataset = 'TWINS'
dataset = 'RORCO'

methods = ne.methods

variance, times = ne.compute_variance(ne.methods, dataset, num_runs=0, folder='saved')

new_variance, new_times = {}, {}
for method in methods:
    new_variance[method] = variance[method]
    new_times[method] = times[method]

ne.benchmark_table(new_variance, new_times, print_md=True, print_latex=True, filename=f'../tables/variance_{dataset}.tex')