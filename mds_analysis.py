# author : Ashwin de Silva
# perform t-sne on the dataset

# import the libraries

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utilities import plot_decision_regions
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from removeParams import removeParams
from mpl_toolkits.mplot3d import Axes3D

# load the dataset

var = sio.loadmat('dataset_28.mat')
data = var['dataset_28']
params = var['params']

params = params[0]
names = []
for i in params :
    names.append(str(i)[2:-2])

# select the validation split and training split

X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

#X_train = removeParams(X_train, 50)

# standardize the data

sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)
#X_test_std = sc.transform(X_test)

# implement isomap algorithm

mds = MDS(n_components=2, metric=True, n_init=10, max_iter=400, verbose=0, eps=0.00001, n_jobs=1, random_state=None, dissimilarity='euclidean')
X_train_tsne = mds.fit_transform(X_train_std)

# training data visualization

colors = ['r', 'b']
markers = ['s', 'x']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_tsne[y_train == l, 0], X_train_tsne[y_train == l, 1], c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()


# lr = LogisticRegression()
# lr.fit(X_train_tsne, y_train)
# plot_decision_regions(X_train_tsne, y_train, classifier=lr)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()

# # reducing in to 3 dimensions
#
# pca = TSNE(n_components=3)
# X_train_pca = pca.fit_transform(X_train_std)
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# colors = ['r', 'b']
# markers = ['s', 'x']
#
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     ax.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], X_train_pca[y_train==l, 2], c=c, label=l, marker=m)
#
# plt.show()

