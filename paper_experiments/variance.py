import sys
sys.path.append('../')
import naturalexperiments as ne
import numpy as np


dataset = 'ACIC 2016'
dataset = 'ACIC 2017'
dataset = 'IHDP'
dataset = 'JOBS'
dataset = 'NEWS'
dataset = 'TWINS'
dataset = 'RORCO'

variance, times = ne.compute_variance(ne.methods, dataset, num_runs=0, folder='saved')

ne.benchmark_table(variance, times, print_md=True, print_latex=True, filename=f'tables/variance_{dataset}.tex')