import pandas as pd
import os
import pickle as pkl
import numpy as np
import requests

def load_acic():
    # Print absolute path
    # Get absolute path to this file
    filename = __file__.replace('__init__.py', 'acic_data.csv')

    if not os.path.exists(filename):
        print('Downloading ACIC data...')
        # Download the data 
        url = 'https://raw.githubusercontent.com/rtealwitter/naturalexperiments/main/naturalexperiments/data/acic/acic_data.csv'

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

if __name__ == '__main__':
    if os.path.exists('input_2016.RData'):
        import rdata
        parsed = rdata.parser.parse_file('input_2016.RData')

        converted = rdata.conversion.convert(parsed)

        data = pd.DataFrame(converted['input_2016'])

        pkl.dump(data, open('acic_data.pkl', 'wb'))
    else:
        print('Please download the ACIC data from https://github.com/vdorie/aciccomp/blob/master/2016/data/input_2016.RData')
        print('Place the input_2016.RData file in the data folder')
        print('Then run this script again')
