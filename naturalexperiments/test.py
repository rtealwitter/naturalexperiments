from data import dataloaders
from model import train, estimate_propensity
from estimators import methods

def test():
    X, y, z = dataloaders['RORCO']()
    p = estimate_propensity(X, z)
    estimate = methods['Regression Discontinuity'](X, y, z, p, train)

if __name__ == '__main__':
    test()