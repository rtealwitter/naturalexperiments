import pickle as pkl
import pandas as pd
import os

def load_news(num=1):
    assert num in range(1, 51), 'Invalid dataset number'
    filename = __file__.replace('__init__.py', 'news_data.pkl')

    datasets = pkl.load(open(filename, 'rb'))

    data = datasets[num-1]

    z = data[0].values

    y = pd.DataFrame({
        'y1' : data[4].values * z,
        'y0' : data[4].values * (1-z)
    }, dtype=float)

    X = data.drop(columns=[0,4]).values

    return X, y, z


if __name__ == '__main__':
    # Read filenames in csv folder
    if os.path.exists('csv'):    
        datasets = []
        filenames = os.listdir('csv')
        for filename in filenames:
            if filename.endswith('.csv.y'):
                data = pd.read_csv('csv/' + filename, header=None)
                datasets += [data]
        pkl.dump(datasets, open('news_data.pkl', 'wb'))
    else:
        print('Please download the news data from https://shubhanshu.com/awesome-causality/#data')
        print('Extract the zip file and place the csv folder in the data folder')
        print('Then run this script again')