import sys
sys.path.append('../')
import naturalexperiment5 as ne

dataset = 'RORCO'

use_methods = ['FlexTENet', 'TNet', 'TARNet', 'RANet', 'Double-Double']

methods = {method: ne.methods[method] for method in use_methods}

dataset = 'RORCO'
ns = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]

variance = ne.compute_variance_by_n(methods, dataset, ns=ns, num_runs=0, folder='saved')

ne.plot_estimates(
    variance, 
    xlabel = r'Number of Observations',
    figure_name=rf'{dataset}: Squared Error by Number of Observations',
    folder = '../images/',
)