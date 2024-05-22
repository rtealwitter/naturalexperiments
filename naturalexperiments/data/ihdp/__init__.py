import os
import pickle as pkl
import pandas as pd

def load_ihdp(num=1):
    assert num in range(1,11), "Invalid dataset number"
    filename = __file__.replace('__init__.py', 'ihdp_data.pkl')

    datasets = pkl.load(open(filename, 'rb'))

    data = datasets[num-1]

    z = data['treatment'].values

    y = pd.DataFrame({
        'y1' : data['y_cfactual'].values * z + data['y_factual'].values * (1-z),
        'y0' : data['y_cfactual'].values * (1-z) + data['y_factual'].values * z,
    }, dtype=float)

    X = data.drop(['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1'], axis=1).values
    
    return X, y, z

if __name__ == '__main__':
    if not os.path.exists('ihdp_data.pkl'):
        datasets = []
        url_prefix = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/IHDP/csv/ihdp_npci_"
        url_suffix = ".csv" 
        colnames = ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1']
        for i in range(1,26):
            colnames += ['x' + str(i)]

        for i in range(1,11):
            url = url_prefix + str(i) + url_suffix
            data = pd.read_csv(url, header=None, names=colnames)
            datasets.append(data)
        pkl.dump(datasets, open('ihdp_data.pkl', 'wb'))