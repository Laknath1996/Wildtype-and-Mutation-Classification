# author : Ashwin de Silva
# perform unsupervised clustering

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
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation

# define useful parameters

# selection paramters
TIME_POINTS = [7, 8]
LOW_VARIANCE_THRESHOLD = 30
ELIMINATE_REDUNDANT_PARAMS = False

# outlier removal parameters
REMOVE_OUTLIERS = True
M = 7

# T_SNE paramters
PERPLEXITY = 15
EARLY_EXAGGERATION = 12
LEARNING_RATE = 30
N_ITER = 1000
MIN_GRAD_NORM = 1e-7
RANDOM_STATE = 25

# SVM parameters
GAMMA = 0.01
C = 15.0

# define redundant features
redundant_params  = ['cvSCBDuration','cvSCBSize','cvDuration','cvInbis',
    'cvJitter','mfrRatio','cvNBSI']

# load the dataset
DIV = TIME_POINTS
X_train, y_train, time_points, params = load_custom_dataset(DIV)  # for custom dataset

# remove redunduant params
if (ELIMINATE_REDUNDANT_PARAMS):
    X_train, params = removeRedundantParams(X_train, redundant_params)

# remove the low variance params
X_train = removeParams(X_train, LOW_VARIANCE_THRESHOLD)

# remove the outliers
if (REMOVE_OUTLIERS):
    X_train, y_train = reject_outliers(X_train, y_train, m=M)

# standardize the data
sc = StandardScaler()
#sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)

# implement t-sne algorithm
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, early_exaggeration=EARLY_EXAGGERATION , learning_rate=LEARNING_RATE,
            init='random', n_iter=N_ITER,
            min_grad_norm=MIN_GRAD_NORM, verbose=True, random_state=RANDOM_STATE) # this model worked well for DIV28

# reduce the dimension
X_train_redu = tsne.fit_transform(X_train_std)

# perform clustering
#y_train_pred = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit_predict(X_train_redu)
#y_train_pred = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto',
#                leaf_size=30, p=None, n_jobs=1).fit_predict(X_train_redu)
#y_train_pred = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None,
#                                   affinity='euclidean', verbose=False).fit_predict(X_train_redu)



#plot the results
plt.scatter(X_train_redu[:, 0], X_train_redu[:, 1], c=y_train_pred)
plt.show()

# performance metrics
NMI = metrics.adjusted_mutual_info_score(y_train, y_train_pred)
ACCURACY_SCORE = metrics.accuracy_score(y_train, y_train_pred)


# print the summary
print('\n')
print('Time Points Used : ', DIV)
print('Low Variance Threshold : ', LOW_VARIANCE_THRESHOLD)
print('Absolute distance to the median : ', M)
print('Perplexity : ', PERPLEXITY)
print('Accuracy Score : ', ACCURACY_SCORE)
print('Training Set Normalized Mutual Information : ', NMI)


# plot the results

#multi_timepoints_plot(X_train_tsne, y_train, time_points) # for the full dataset
# single_timepoint_plot(X_train_redu, y_train, DIV) # plot the redcued dimensional embedding
#
# plot_decision_regions(X_train_redu, y_train, classifier=svm) # plot the regional boundaries
#
# plt.xlabel('Dim 1')
# plt.ylabel('DIm 2')
# plt.title('Reduced Dimension Visualization')
# plt.show()




