import sys
sys.path.insert(1, '../')
import naturalexperiments as ne
from tqdm import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib

def plot_heatmap(algo, clip_values, noise_multipliers, np_array): 
    plt.imshow(np_array, cmap='hot', interpolation='nearest', norm=matplotlib.colors.LogNorm())
    y_ticks = [f'{x:.0e}' for x in clip_values]
    x_ticks = [f'{nm:.0e}' for nm in noise_multipliers]
    plt.title(f'Squared Error of {algo} Estimator')
    plt.xticks(range(len(noise_multipliers)), x_ticks, rotation=45)
    plt.yticks(range(len(clip_values)), y_ticks)
    plt.ylabel('Gradient Clip Value')
    plt.xlabel('Noise Multiplier')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'../images/{algo}_heatmap.pdf')
    plt.clf()

lr = .001

noise_multipliers = []
for i in range(0, 2):
    noise_multipliers += [1*10**-i]
    noise_multipliers += [.75*10**-i]
    noise_multipliers += [.5*10**-i]
    noise_multipliers += [.25*10**-i]

clip_values = [.01, .05, .1, .5, 1, 5, 10]

clip_values = list(sorted(clip_values))
noise_multipliers = list(sorted(noise_multipliers))


if os.path.exists('saved/heatmap.pkl'):
    saved = pickle.load(open('saved/heatmap.pkl', 'rb'))
else:
    saved = {'Double-Double' : {}, 'Off-policy' : {}}

X, _, _ = ne.dataloaders['RORCO']()

num_runs = 100

clip_grid, noise_multiplier_grid = np.meshgrid(clip_values, noise_multipliers)

for clip_value, noise_multiplier in tqdm(zip(clip_grid.flatten(), noise_multiplier_grid.flatten())):
    for name in saved:
        if clip_value not in saved[name]:
            saved[name][clip_value] = {}
        if noise_multiplier not in saved[name][clip_value]:
            saved[name][clip_value][noise_multiplier] = []
    num_to_go = num_runs - len(saved['Double-Double'][clip_value][noise_multiplier])
    def train_fn(X_train, y_train, X_test=None, weights=None, epochs=200, return_model=False):
        return ne.train(X_train=X_train, y_train=y_train, X_test=X_test, weights=weights, lr=lr, noise_multiplier=noise_multiplier, epochs=epochs, return_model=return_model, clip_value=clip_value)
    for _ in range(num_to_go):
        X = X[np.random.permutation(X.shape[0])]
        _, y, p = ne.build_synthetic_outcomes(X=X)
        uniform = np.random.uniform(size=X.shape[0])
        z = (uniform < p).astype(int)
        p_estimated = ne.estimate_propensity(X, z)

        error_3d = ne.compute_double_double(X, y, z, p_estimated, train_fn)
        saved['Double-Double'][clip_value][noise_multiplier].append(error_3d)

        error_off_policy = ne.compute_off_policy(X, y, z, p_estimated, train_fn)
        saved['Off-policy'][clip_value][noise_multiplier].append(error_off_policy)
    
    pickle.dump(saved, open('saved/heatmap.pkl', 'wb'))


for algo in saved:
    np_array = np.array([[np.mean(saved[algo][clip_value][nm]) for nm in noise_multipliers] for clip_value in clip_values])
    plot_heatmap(algo, clip_values, noise_multipliers, np_array)