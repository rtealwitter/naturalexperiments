from data import dataloaders
from model import train, estimate_propensity
from estimators import methods

def test(dataset, method_name):
    X, y, z = dataloaders[dataset]()
    p = estimate_propensity(X, z)
    estimate = methods[method_name](X, y, z, p, train)

if __name__ == '__main__':
    dataset = 'ACIC'
    method_name = 'Regression Discontinuity'
    test(dataset, method_name)