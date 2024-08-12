import sys
sys.path.insert(1, '../')
import naturalexperiments as ne


dataset = 'RORCO'

methods = {name : ne.methods[name] for name in ne.methods.keys() if 'Net' not in name}

variance, times = ne.compute_variance(methods, dataset, train_fn=ne.train_bart, num_runs=10, folder='saved_bart')

new_variance, new_times = {}, {}
for method in methods:
    new_variance[method] = variance[method]
    new_times[method] = times[method]

ne.benchmark_table(new_variance, new_times, print_md=True, print_latex=True)