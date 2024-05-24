from naturalexperiments import *
import numpy as np

def test(dataset, method_name):
    X, y, z = dataloaders[dataset]()
    if z is None:
        z = np.random.randint(0, 2, y.shape[0])
    p = estimate_propensity(X, z) 
    estimate = methods[method_name](X, y, z, p, train)

if __name__ == '__main__':
    #dataset_table(dataloaders, print_md=True, print_latex=True)
    #plot_all_data(dataloaders, folder='output')
    
    method_name = 'Double-Double'
    dataset = 'RORCO'
    #for dataset in dataloaders:
    test(dataset, method_name)
    #compute_estimates(methods, dataset, num_runs=1, folder='output')
    #variance, times = compute_variance(methods, dataset, num_runs=0, folder='output')
    #benchmark_table(variance, times, print_md=True, print_latex=True)
    #compute_variance_by_n(methods, dataset, ns=[1000,3000,4000], num_runs=3, folder='output')
    #compute_variance_by_correlation(methods, dataset, num_runs=1, folder='output')
    #compute_variance_by_entropy(methods, dataset, num_runs=1, folder='output')