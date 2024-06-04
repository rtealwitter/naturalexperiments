import sys
sys.path.append('../')

import naturalexperiment5 as ne

ne.dataset_table(ne.dataloaders, print_md=True, print_latex=True)

ne.plot_all_data(ne.dataloaders, folder='saved/')