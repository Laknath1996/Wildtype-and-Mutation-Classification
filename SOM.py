
import numpy as np

class SOM(object):

    def __init__(self, n_feats = 33):
        self.W = np.random.random((100,n_feats ));

        self.Y = np.array([[i, j] for i in range(10) for j in range(10)])



    def fit_transform(self, X):
        for i in range(100):
            lr = .1
            radius = 2
            for x in X:
                radius*=0.99
                lr *= 1-i*1./100
                bmu = np.linalg.norm(x-self.W, axis=1).argmin()
                ldist = np.linalg.norm(self.Y[bmu]-self.Y, axis=1)
                neighbors = np.where(ldist <= radius)[0]

                theta = np.array([np.exp(-.5*ldist[neighbors]**2/radius**2)]).T

                self.W[neighbors]+=(x-self.W[bmu])*theta*lr


        ret = []

        for x in X:
            ret.append(self.Y[np.linalg.norm(x-self.W, axis=1).argmin()])

        return np.array(ret)
