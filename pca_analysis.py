# author : Ashwin de Silva
# perform pca analysis on the dataset

# import the libraries

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utilities import plot_decision_regions
from sklearn.decomposition import KernelPCA
from removeParams import removeParams
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import FastICA
from matplotlib import cm
from numpy import linspace
from sklearn.manifold import MDS
from removeParams import *

# load the dataset

var = sio.loadmat('dataset_flat.mat')
data = var['data']
params = var['params']

params = params[0]
names = []
for i in params :
    names.append(str(i)[2:-2])


X, y = data[:, :-2], data[:, -2]
time_points = data[:, -1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0)
X_train, y_train = X, y

X_train = removeParams(X, 50)

#X_train = removeParams(X_train, 70)

# standardize the data

sc = StandardScaler()
#sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)
#X_test_std = sc.transform(X_test)

#mds = MDS(n_components=2, metric=True, n_init=10, max_iter=400, verbose=0, eps=0.00001, n_jobs=1, random_state=None, dissimilarity='euclidean')
#X_train_pca = mds.fit_transform(X_train_std)

#pca = PCA(n_components=2)
#X_train_pca = pca.fit_transform(X_train_std)

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0,
                 fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False,
                random_state=None, copy_X=True, n_jobs=1)

X_train_pca = kpca.fit_transform(X_train_std)

start = 0.5
stop = 1.0
number_of_lines = 8
cm_subsection = linspace(start, stop, number_of_lines)

wildtype_colors = [cm.Reds(x) for x in cm_subsection]
mutation_colors = [cm.Blues(x) for x in cm_subsection]
markers = ['s', 'x']

X_wild = X_train_pca[y_train == 0]
t_wild = time_points[y_train == 0]
X_mut = X_train_pca[y_train == 1]
t_mut = time_points[y_train == 1
]
for t, w in zip(np.unique(time_points), wildtype_colors):
     plt.scatter(X_wild[t_wild == t, 0], X_wild[t_wild == t, 1], c=w, label=t)

plt.hold(True)

for t, m in zip(np.unique(time_points), mutation_colors):
     plt.scatter(X_mut[t_mut == t, 0], X_mut[t_mut == t, 1], c=m, label=t)


# colors = ['r', 'b']
# markers = ['s', 'x']
#
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()


# # compute the covariance matrix and its eigenpairs
#
# cov_mat = np.cov(X_train_std.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) # can use singular value decomposition or hermetian matrix decomposition
#
# # plot the variance explained ratios
#
# tot = sum(eigen_vals)
# var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)
#
# plt.bar(range(1,len(eigen_vals)+1), var_exp, alpha=0.5, align='center', label='individual explained variance')
# plt.step(range(1,len(eigen_vals)+1), cum_var_exp, where='mid', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()
#
# # select the two eigenvectors with the highest eigenvalues
#
# eigen_pairs =[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
# eigen_pairs.sort(reverse=True)
#
# w= np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis])) #projection matrix
#
# # project the data to the 2D plane
#
# X_train_pca = X_train_std.dot(w)

# # training data visualization
#
# colors = ['r', 'b']
# markers = ['s', 'x']
#
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m)
#
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower right')
# plt.show()
#
#
# lr = LogisticRegression()
# lr.fit(X_train_pca, y_train)
# plot_decision_regions(X_train_pca, y_train, classifier=lr)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()

# # test data visualization
#
# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)
# plot_decision_regions(X_test_pca, y_test, classifier=lr)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend(loc='lower left')
# plt.show()

# # reducing in to 3 dimensions
#
# pca = PCA(n_components=3)
# X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)
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
