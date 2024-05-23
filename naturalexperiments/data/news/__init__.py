import pickle as pkl
import pandas as pd
import os
import requests

def load_news(num=1):
    assert num in range(1, 51), 'Invalid dataset number'
    filename = __file__.replace('__init__.py', f'news_data_{num}.csv')

    if not os.path.exists(filename):
        print('Downloading News data...')
        # Download the data 
        url = f'https://raw.githubusercontent.com/rtealwitter/naturalexperiments/main/naturalexperiments/data/news/news_data_{num}.csv'

        r = requests.get(url)
        open(filename, 'wb').write(r.content)

    data = pd.read_csv(filename, header=None)

    z = data[0].values

    y = pd.DataFrame({
        'y1' : data[4].values * z,
        'y0' : data[4].values * (1-z)
    }, dtype=float)

    X = data.drop(columns=[0,4]).values

    return X, y, z


if __name__ == '__main__':
    # Read filenames in csv folder
    try:
        for seed_num in range(1, 51):
            filename = __file__.replace('__init__.py', f'csv/topic_doc_mean_n5000_k3477_seed_{seed_num}.csv.y')
            data = pd.read_csv(filename, header=None)
            data.to_csv(__file__.replace('__init__.py', f'news_data_{seed_num}.csv'), index=False, header=False)
    except FileNotFoundError:
        print('Please download the news data from https://shubhanshu.com/awesome-causality/#data')
        print('Extract the zip file and place the csv folder in the news directory')
        print('Then run this script again')