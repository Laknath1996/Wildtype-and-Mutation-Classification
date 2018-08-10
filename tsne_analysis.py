# author : Ashwin de Silva
# perform t-sne on the dataset

# import the libraries

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from matplotlib import cm
from numpy import linspace
from sklearn.preprocessing import StandardScaler
from removeParams import *
from utilities import *
from sklearn.svm import SVC
from utilities import plot_decision_regions

# load the dataset
#X_train, y_train, time_points, params = load_full_dataset() # for full datset
X_train, y_train, params = load_single_dataset() # for singular dataset

# remove the low variance params
X_train = removeParams(X_train, 50)

# standardize the data
sc = StandardScaler()
#sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)
#X_test_std = sc.transform(X_test)

# implement t-sne algorithm

tsne = TSNE(n_components=2, perplexity=100, early_exaggeration=12, learning_rate=300, init='random', n_iter=1000,
            min_grad_norm=1e-7, verbose=True, random_state=25) # this model worked well for DIV28


# reduce the dimension
X_train_tsne = tsne.fit_transform(X_train_std)

# plot the results
#multi_timepoints_plot(X_train_vae, y_train, time_points) # for the full dataset
single_timepoint_plot(X_train_tsne, y_train, 'DIV28')

# perform classification

svm = SVC(kernel='rbf', random_state=0, gamma=0.01, C=10.0, verbose=True)
svm.fit(X_train_tsne, y_train)
plot_decision_regions(X_train_tsne, y_train, classifier=svm)

# lr = LogisticRegression(C=1000.0, random_state=0)
# lr.fit(X_train_tsne, y_train)
# plot_decision_regions(X_train_tsne, y_train, classifier=lr)

plt.xlabel('Dim 1')
plt.ylabel('DIm 2')
#plt.legend(('Mutation', 'Wild Type'))
plt.title('Reduced Dimension Visualization')
plt.show()


