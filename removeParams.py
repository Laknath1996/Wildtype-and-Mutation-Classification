# author : Ashwin de Silva
# eliminate low variance parameters

import numpy as np

def removeParams(X, thresh):
    stdev = np.std(X,0)
    X_n = X[:, stdev > thresh]
    return X_n

