import os
import pickle as pkl
import pandas as pd

def load_twins():
    filename = __file__.replace('__init__.py', 'twins_data.pkl')

    data = pkl.load(open(filename, 'rb'))

    # Drop cols with too many missing values
    nan_cols = data.isnull().sum() > 10000
    nan_cols = nan_cols[nan_cols].index
    data = data.drop(columns=nan_cols).dropna()

    z = None

    y = pd.DataFrame({
        'y1' : data['mort_1'].values,
        'y0' : data['mort_0'].values
    })

    cols_to_drop = [x for x in data.columns if '_0' in x or '_1' in x]

    X = data.drop(columns=cols_to_drop)

    return X, y, z 

if __name__ == '__main__':
    if not os.path.exists('twins_data.pkl'):
        url_weight = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/TWINS/twin_pairs_T_3years_samesex.csv"

        url_X = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/TWINS/twin_pairs_X_3years_samesex.csv"

        url_y = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/TWINS/twin_pairs_Y_3years_samesex.csv"

        weight = pd.read_csv(url_weight) # Birth weight 
        X = pd.read_csv(url_X) # Covariates
        y = pd.read_csv(url_y) # Mortality (binary)

        data = pd.concat([weight, X, y], axis=1).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

        pkl.dump(data, open('twins_data.pkl', 'wb'))
