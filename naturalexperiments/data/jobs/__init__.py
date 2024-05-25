import pickle as pkl
import pandas as pd
import os
import requests

def load_jobs():
    filename = __file__.replace('__init__.py', 'jobs_data.csv')

    if not os.path.exists(filename):
        print('Downloading Jobs data...')
        # Download the data 
        url = 'https://raw.githubusercontent.com/rtealwitter/naturalexperiments/main/naturalexperiments/data/jobs/jobs_data.csv'

        r = requests.get(url)
        open(filename, 'wb').write(r.content)

    data = pd.read_csv(filename) 

    z = data['treat'].values
    y = pd.DataFrame({
        'y1' : data['re78'].values * z,
        'y0' : data['re78'].values * (1-z)
    }, dtype=float)

    X = data.drop(columns=['treat', 're78']).values

    return X, y, z

if __name__ == '__main__':

    url_treatment = "https://users.nber.org/~rdehejia/data/nsw_treated.txt"
    url_control = "https://users.nber.org/~rdehejia/data/nsw_control.txt"

    col_names = ["treat", "age", "educ", "black", "hisp", "married", "nodegr", "re75", "re78"]

    treatment = pd.read_csv(url_treatment, names=col_names, sep="  ", engine='python')
    control = pd.read_csv(url_control, names=col_names, sep="  ", engine='python')

    data = pd.concat([treatment, control], ignore_index=True)

    data.to_csv('jobs_data.csv', index=False)