# author : Ashwin de Silva
# plot the correlation plot for the variables

# import the relevant libraries

import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utilities import *

# load the dataset

X_train, y_train, time_points, params = load_full_dataset()

# X_mut = X_train[y_train == 1]
X_wild = X_train[y_train == 0]

# data = pd.DataFrame(X_mut)
data = pd.DataFrame(X_train)
corr = data.corr()

# plot correlation matrix

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, np.shape(params)[0],1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(params, rotation = 'vertical')
ax.set_yticklabels(params)
plt.show()





