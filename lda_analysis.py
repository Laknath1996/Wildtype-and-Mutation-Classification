# author : Ashwin de Silva
# perform lda analysis on the dataset

# import the libraries

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utilities import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from utilities import *

# load the dataset
X_train, y_train, time_points, params = load_full_dataset() # for full datset
#X_train, y_train, params = load_single_dataset() # for singular dataset

# standardize the data
sc = StandardScaler()
#sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)

# compute the mean vectors

mean_vecs = []
for label in range(0,2):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))

# compute the within class scatter matrix

d = np.shape(X_train_std)[1]
S_W = np.zeros((d, d))

for label, mv in zip(range(0, 2), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter

# compute the between class scatter matrix

mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

# compute the eigen values

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))

X_train_lda = X_train_std.dot(w)

# plot the results
multi_timepoints_plot(X_train_lda, y_train, time_points) # for the full dataset
#single_timepoint_plot(X_train_lda, y_train, 'DIV28')



