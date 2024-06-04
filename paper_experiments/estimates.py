# Add repo parent directory to path
import sys
sys.path.append('../')
import naturalexperiment5 as ne

from importlib import reload
reload(ne)

dataset = 'RORCO Real'

methods = ne.methods

variance, times = ne.compute_estimates(methods, dataset, num_runs=0, folder='saved')

ne.benchmark_table(variance, times, print_md=True, print_latex=True, filename=f'../tables/estimates_{dataset}.tex', include_color=False)


