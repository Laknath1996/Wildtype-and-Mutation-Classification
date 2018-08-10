# author : Ashwin de Silva
# perform the chi-squared test on the dataset

# import the relevant libraries

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# load the dataset

var = sio.loadmat('dataset_28.mat')
data = var['dataset_28']
params = var['params']

params = params[0]
names = []
for i in params :
    names.append(str(i)[2:-2])



