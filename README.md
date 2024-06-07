# naturalexperiments

The `naturalexperiments` package is a comprehensive toolbox for treatment effect estimation. The package includes a variety of datasets, estimators, and evaluation metrics for treatment effect estimation. The package is designed to be accessible to researchers and practitioners who are new to treatment effect estimation and to provide a comprehensive set of tools for experienced researchers.

To quickly run the code described in the README, you can open the [demo.ipynb](demo.ipynb) notebook on Google Colab.

## Installation

The `naturalexperiments` package is available on PyPI. You can install the package using pip.

```bash
pip install naturalexperiments
```

## Datasets

We introduce a novel treatment effect dataset from an early childhood literacy natural experiment. The treatment in the experiment is participation in Reach Out and Read Colorado (RORCO). The dataset has an observational version called RORCO Real with real literacy outcomes and a semi-synthetic version called RORCO for estimator evaluation purposes.

In addition to RORCO and RORCO Real, we provide easy access to standard treatment effect datasets including ACIC 2016, ACIC 2017, IHDP, Jobs, News, and Twins.

All of the datasets can be loaded using the `dataloaders` object.

```python
import naturalexperiments as ne

dataset = 'RORCO'

X, y, z = ne.dataloaders[dataset]()
```

## Estimators

We propose a novel, theoretically motivated doubly robust estimator called Double-Double. In addition to Double-Double, we provide implementations of more than 20 established estimators from the literature.

All of the estimators can be easily loaded using the `methods` object.

```python
method = 'Double-Double'
estimator = ne.methods[method]
```

The available methods include Regression Discontinuity, Propensity Stratification, Direct Difference, Adjusted Direct, Horvitz-Thompson, TMLE, Off-policy, Double-Double, Doubly Robust, Direct Prediction, SNet, FlexTENet, OffsetNet, TNet, TARNet, DragonNet, SNet3, DRNet, RANet, PWNet, RNet, and XNet.

Each method takes the following arguments: the covariates `X`, the outcomes `y`, the treatment assignment `z`, propensity score estimates `p`, and a function for training `train` predictions in the estimator.

We can use the `estimate_propensity` function to estimate the propensity scores.

```python
p = ne.estimate_propensity(X, z)
```

Then, with the propensity scores, we can estimate the treatment effects.

```python
estimated_effect = estimator(X, y, z, p, ne.train)
```

By default, `train` trains a three-layer neural network with 100 units in each layer and ReLU activations. 
Some estimators, such as regression discontinuity, do not use the training functions and some estimators, such as the CATENet estimators, use custom training functions defined in the estimator.

## Exploring the Datasets

We can explore the datasets with a tabular comparison and several figures.

```python
# Produces a markdown table comparing the size, number of variables, treatment rate, etc.
ne.dataset_table(ne.dataloaders, print_md=True)
```

The table will look like this:

| Dataset    |   Size |   Variables |   % Treated |   Cross Entropy |   Corr(y1, p) |   Corr(y0, p) |
|------------|--------|-------------|-------------|-----------------|---------------|---------------|
| ACIC 2016  |   4802 |          54 |        18.4 |           0.379 |        0.129  |        0.0393 |
| ACIC 2017  |   4302 |          50 |        51   |           0.606 |       -0.116  |       -0.0247 |
| IHDP       |    747 |          26 |        18.6 |           0.459 |        0.0785 |        0.0199 |
| JOBS       |    722 |           8 |        41.1 |           0.0827 |        0.0356 |        0.0766 |
| NEWS       |   5000 |           3 |        45.8 |           0.545 |        0.858  |       -0.563  |
| TWINS      |  50820 |          40 |        49   |           0.506 |        0.00133|       -0.000665|
| RORCO Real |   4178 |          78 |        25.3 |           0.162 |       -0.027  |       -0.101  |
| RORCO      |  21663 |          78 |        49.4 |           0.151 |       -0.979  |       -0.98   |

We can also plot the distribution of the covariates, the propensity scores, and the outcomes.

```python
# Produces plots of the propensity score distribution, outcomes by propensity scores, and propensity calibration
# To limit number of plots, we provide RORCO and RORCO Real datasets
dataloaders = {name : ne.dataloaders[name] for name in ['RORCO', 'RORCO Real']}
ne.plot_all_data(ne.dataloaders)
```

## Benchmarking the Estimators

We can benchmark the estimators on the datasets using the `compute_variance` function.

```python
methods = {name: ne.methods[name] for name in ['Double-Double', 'Regression Discontinuity', 'DragonNet']}

variance, times = ne.compute_variance(methods, dataset, num_runs=5)
```

Due to the computational complexity of some estimators (e..g, the CATENets), the benchmark subsamples the data by default. We can adjust the subsample size with the `limit` argument. Even then, many estimators may take a long time to run.

Once we benchmark the estimators, we can print the results in a table.

```python
ne.benchmark_table(variance, times, print_md=True)
```

The table will look like this:

| Method                   |     Mean |   1st Quartile |   2nd Quartile |   3rd Quartile |   Time (s) |
|--------------------------|----------|----------------|----------------|----------------|------------|
| Double-Double            | 1.04e-05 |       8.18e-06 |       1.01e-05 |       1.38e-05 |   27.6     |
| Regression Discontinuity | 0.00467  |       0.00411  |       0.00493  |       0.00588  |    0.00115 |
| DragonNet                | 0.0223   |       0.0217   |       0.0227   |       0.0252   |    4.62    |

## Additional Features

The `naturalexperiments` package includes additional features for comprehensively evaluating treatment effect estimation.

There are functions for computing the empirical variance as a function of the sample size, the correlation in the outcomes, and the propensity score accuracy.

These functions and more appear in the `paper_experiments` folder (as the name suggests, the folder includes code to reproduce the results in the paper). Because some experiments are computationally intensive, the functions are designed to run in parallel by writing the results to a shared cache. We include the cached results from >100 repetitions of each experiment in `paper_experiments/saved` so that the tables and plots can be reproduced easily.

## Package Structure

The `naturalexperiments` package is designed to be modular and extensible. The package is organized in the following way:

- `naturalexperiments/`
  - `data/`: Contains modules for loading datasets.
  - `estimators/`: Contains functions for loading estimators.
  - `__init__.py`: Initializes the package.
  - `benchmark.py`: Contains functions for benchmarking estimators.
  - `data_summary.py`: Contains functions for exploring datasets.
  - `model.py`: Contains functions for training the predictions used in the estimators.
  - `utils.py`: Contains shared utility functions.

## Adding a Dataset

The `naturalexperiments` package is designed to grow with the treatment effect estimation community. To this end, we welcome contributions of new datasets.

To add a dataset, create a new module in the `naturalexperiments/data` folder. The module should implement a function that loads covariates `X`, outcomes `y`, and treatment assignments `z`. In order to keep the package accessible, we ask that you cache the data in the module and provide a function that loads the data without any manual intervention. For examples, please refer to the existing modules in `naturalexperiments/data`.

Once the module is created, add the module to the `__init__.py` file in the `naturalexperiments/data` folder.

## Adding an Estimator

The `naturalexperiments` package is designed to grow with the treatment effect estimation community. To this end, we welcome contributions of new estimators.

To add a new estimator, create a new file in the `naturalexperiments/methods` folder that defines a function that takes the covariates `X`, the outcomes `y`, the treatment assignment `z`, the propensity score estimates `p`, and a function for training `train` predictions in the estimator. The function should return the estimated treatment effect. 

Once the estimator is created, add the estimator to the `__init__.py` file in the `naturalexperiments/methods` folder.

## Questions and Feedback

If you have any questions or feedback, please feel free to open an issue on the GitHub repository.

## Citation

If you use the `naturalexperiments` package in your research, please cite the paper: Coming soon.
