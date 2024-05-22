import numpy as np
import pandas as pd
import pickle as pkl
import sklearn.preprocessing
import os
import requests

def load_rorco():
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
    y = pd.DataFrame({
        'y1' : data[outcome_variable].values * z,
        'y0' : data[outcome_variable].values * (1-z)
    }, dtype=float)

    X = data.drop(columns = [treatment_variable, outcome_variable, 'num_from_RORCO'])

    X = X.select_dtypes(include = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    X.replace(np.nan, 0, inplace=True)

    X = sklearn.preprocessing.StandardScaler().fit(X.values).transform(X.values)

    return X, y, z

