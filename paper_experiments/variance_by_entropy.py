import naturalexperiments as ne


dataset = 'RORCO'

use_methods = ['Regression Discontinuity', 'Propensity Stratification', 'Adjusted Direct', 'Off-policy', 'Double-Double']

methods = {method: ne.methods[method] for method in use_methods}

variance = ne.compute_variance_by_entropy(methods, dataset, num_runs=0, folder='saved')

ne.plot_estimates(
    variance, 
    xlabel = r'Cross Entropy',
    figure_name=rf'{dataset}: Squared Error by Cross Entropy',
    folder = 'images/',
)