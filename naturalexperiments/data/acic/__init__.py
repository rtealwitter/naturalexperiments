import pandas as pd
import os
import pickle as pkl
import numpy as np
import requests

def load_acic(year):
    # Print absolute path
    # Get absolute path to this file
    assert year in ['16', '17'], "Year must be '16' or '17'"
    filename = __file__.replace('__init__.py', f'acic{year}_data.csv')

    if not os.path.exists(filename):
        print('Downloading ACIC data...')
        # Download the data 
        url = f'https://raw.githubusercontent.com/rtealwitter/naturalexperiments/main/naturalexperiments/data/acic/acic{year}_data.csv'

        r = requests.get(url)
        open(filename, 'wb').write(r.content)
    
    data = pd.read_csv(filename)

    # Remove string columns
    data = data.select_dtypes(include = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    data = data.astype(np.float32)

    # Set data type of columns to float32

    # No data description available
    # Assuming the first binary column is the treatment indicator
    binary_cols = []
    for col in data.columns:
        if data[col].nunique() == 2:
            binary_cols += [col]
    binary_col = binary_cols[0]

    z = data[binary_col].values

    # Assuming the last column is the target column
    target_col = data.columns[-1]
    y = pd.DataFrame({
        'y1' : data[target_col].values * z,
        'y0' : data[target_col].values * (1-z),
    }, dtype=float)

    

    X = np.array(data.drop(columns=[binary_col, target_col]).values, dtype=np.float32)

    return X, y, z

def load_acic16():
    return load_acic('16')

def load_acic17():
    return load_acic('17')

if __name__ == '__main__':
    for year in ['16', '17']:
        if os.path.exists(f'input_20{year}.RData'):
            import rdata
            parsed = rdata.parser.parse_file(f'input_20{year}.RData')

            converted = rdata.conversion.convert(parsed)

            data = pd.DataFrame(converted[f'input_20{year}'])

            data.to_csv(f'acic{year}_data.csv', index=False)
        else: 
            print(f'Please download the ACIC{year} data from https://github.com/vdorie/aciccomp/blob/master/20{year}/data/input_20{year}.RData')
            print(f'Place the input_20{year}.RData file in the data folder')
            print('Then run this script again')
   
