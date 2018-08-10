# author : Ashwin de Silva
# plot the correlation plot for the variables

# import the relevant libraries

import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load the dataset

var = sio.loadmat('dataset_28.mat')
data = var['dataset_28']
params = var['params']

params = params[0]
names = []
for i in params :
    names.append(str(i)[2:-2])

data = pd.DataFrame(data)
corr = data.corr()

# plot correlation matrix

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,np.shape(names)[0],1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names,rotation = 'vertical')
ax.set_yticklabels(names)
plt.show()





