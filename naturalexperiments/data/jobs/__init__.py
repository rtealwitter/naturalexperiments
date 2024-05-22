import pickle as pkl
import pandas as pd
import os

def load_jobs():
    filename = __file__.replace('__init__.py', 'jobs_data.pkl')

    data = pkl.load(open(filename, 'rb'))

    z = data['treat'].values
    y = pd.DataFrame({
        'y1' : data['re78'].values * z,
        'y0' : data['re78'].values * (1-z)
    }, dtype=float)

    X = data.drop(columns=['treat', 're78']).values

    return X, y, z

if __name__ == '__main__':
    if not os.path.exists('jobs_data.pkl'):
        url_treatment = "https://users.nber.org/~rdehejia/data/nsw_treated.txt"
        url_control = "https://users.nber.org/~rdehejia/data/nsw_control.txt"

        col_names = ["treat", "age", "educ", "black", "hisp", "married", "nodegr", "re75", "re78"]

        treatment = pd.read_csv(url_treatment, names=col_names, sep="  ", engine='python')
        control = pd.read_csv(url_control, names=col_names, sep="  ", engine='python')

        data = pd.concat([treatment, control], ignore_index=True)

        pkl.dump(data, open('jobs_data.pkl', 'wb'))