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
    #plot_all_data(dataloaders)
    
    method_name = 'Regression Discontinuity'
    dataset = 'TWINS'
    #for dataset in dataloaders:
    test(dataset, method_name)