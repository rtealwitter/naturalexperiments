import sys
sys.path.insert(1, '../')
import naturalexperiments as ne

import numpy as np

dataset = 'RORCO'

from tqdm import tqdm

# Load the data

reps = 100

real_ces = []
pred_ces = []
for i in tqdm(range(reps)):
    X, y, z, p = ne.load_rorco(return_p=True)
    # Compute the cross entropy between p and z
    real_ce = -np.mean(z * np.log(p) + (1-z) * np.log(1-p))
    real_ces.append(real_ce)

    # Compute the cross entropy between the estimated propensity and z
    p_estimated = ne.estimate_propensity(X, z)
    pred_ce = -np.mean(z * np.log(p_estimated) + (1-z) * np.log(1-p_estimated))
    pred_ces.append(pred_ce)

print(f'Real CE: {np.mean(real_ces)} +/- {np.std(real_ces)}')
print(f'Pred CE: {np.mean(pred_ces)} +/- {np.std(pred_ces)}')

## Output
#Real CE: 0.20161135561298912 +/- 0.028944375744682272
#Pred CE: 0.19626732339239336 +/- 0.028314443357118992
