import sys
sys.path.append('../')

import naturalexperiments as ne

dataloaders = {name: ne.dataloaders[name] for name in ['ACIC 2017']}

ne.dataset_table(ne.dataloaders, print_md=True, print_latex=True)

ne.plot_all_data(ne.dataloaders, folder='saved/')