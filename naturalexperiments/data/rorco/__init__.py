import numpy as np
import pandas as pd
import sklearn.preprocessing
from ...utils import build_synthetic_outcomes
import os
import requests

def load_rorco_real():
    filename = __file__.replace('__init__.py', 'rorco_data.csv')

    if not os.path.exists(filename):
        print('Downloading RORCO data...')
        # Download the data 
        url = 'https://raw.githubusercontent.com/rtealwitter/naturalexperiments/main/naturalexperiments/data/rorco/rorco_data.csv'

        r = requests.get(url)
        open(filename, 'wb').write(r.content)

    data = pd.read_csv(filename)

    # Restrict to rural schools
    data = data[data['is_rural'] == 1]

    outcome_variable = 'Mean Scale Score'
    treatment_variable = 'is_RORCO'

    z = data[treatment_variable].values
    y_all = data[outcome_variable].values
    y_all = (y_all - y_all.mean()) / y_all.std()
    y = pd.DataFrame({
        'y1' : y_all * z,
        'y0' : y_all * (1-z)
    }, dtype=float)

    X = data.drop(columns = [treatment_variable, outcome_variable, 'num_from_RORCO'])

    X = X.select_dtypes(include = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    X.replace(np.nan, 0, inplace=True)

    X = sklearn.preprocessing.StandardScaler().fit(X.values).transform(X.values)

    return X, y, z

def load_rorco():
    filename = __file__.replace('__init__.py', 'rorco_data.csv')

    if not os.path.exists(filename):
        print('Downloading RORCO data...')
        # Download the data 
        url = 'https://raw.githubusercontent.com/rtealwitter/naturalexperiments/main/naturalexperiments/data/rorco/rorco_data.csv'

        r = requests.get(url)
        open(filename, 'wb').write(r.content)

    data = pd.read_csv(filename)

    outcome_variable = 'Mean Scale Score'
    treatment_variable = 'is_RORCO'

    X = data.drop(columns = [treatment_variable, outcome_variable, 'num_from_RORCO'])

    X = X.select_dtypes(include = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    X.replace(np.nan, 0, inplace=True)

    X = sklearn.preprocessing.StandardScaler().fit(X.values).transform(X.values)

    X, y, p = build_synthetic_outcomes(X)
    uniform = np.random.random_sample(X.shape[0])
    z = (uniform < p).astype(int)
    return X, y, z
