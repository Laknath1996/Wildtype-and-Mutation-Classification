# author : Ashwin de Silva
# perform isomap on the dataset

# import the libraries

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utilities import plot_decision_regions
from sklearn.manifold import Isomap
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

# standardize the data

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# implement isomap algorithm

isomap = Isomap(n_neighbors=2)
X_train_isomap = isomap.fit_transform(X_train_std)

# training data visualization

colors = ['r', 'b']
markers = ['s', 'x']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_isomap[y_train == l, 0], X_train_isomap[y_train == l, 1], c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()


lr = LogisticRegression()
lr.fit(X_train_isomap, y_train)
plot_decision_regions(X_train_isomap, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

# test data visualization

X_test_isomap = isomap.transform(X_test_std)
plot_decision_regions(X_test_isomap, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

# reducing in to 3 dimensions

pca = isomap(n_components=5)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'b']
markers = ['s', 'x']

for l, c, m in zip(np.unique(y_train), colors, markers):
    ax.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], X_train_pca[y_train==l, 2], c=c, label=l, marker=m)

plt.show()

