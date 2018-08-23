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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# define useful parameters

# selection paramters
TIME_POINTS = [7, 8]
LOW_VARIANCE_THRESHOLD = 30
ELIMINATE_REDUNDANT_PARAMS = False

# T_SNE paramters
PERPLEXITY = 10
EARLY_EXAGGERATION = 12
LEARNING_RATE = 30
N_ITER = 1000
MIN_GRAD_NORM = 1e-7
RANDOM_STATE = 25

# SVM parameters
GAMMA = 0.01
C = 10.0

# other params
RESOLUTION = 1

# define redundant features
redundant_params  = ['cvSCBDuration','cvSCBSize','cvDuration','cvInbis',
    'cvJitter','mfrRatio','cvNBSI']

# load the dataset
DIV = TIME_POINTS
X, y, time_points, params = load_custom_dataset(DIV)  # for custom dataset

# select the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# remove redunduant params
if (ELIMINATE_REDUNDANT_PARAMS):
    X_train, params = removeRedundantParams(X_train, redundant_params)
    X_test, params = removeRedundantParams(X_test, redundant_params)

# remove the low variance params
X_train = removeParams(X_train, LOW_VARIANCE_THRESHOLD)
X_test = removeParams(X_test, LOW_VARIANCE_THRESHOLD)

# standardize the data
sc = StandardScaler()
#sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)

# implement t-sne algorithm

Train_Acc = []
Test_Acc = []

for PERPLEXITY in range(10, 201, RESOLUTION):
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, early_exaggeration=EARLY_EXAGGERATION , learning_rate=LEARNING_RATE,
                init='random', n_iter=N_ITER,
                min_grad_norm=MIN_GRAD_NORM, verbose=True, random_state=RANDOM_STATE) # this model worked well for DIV28

    # reduce the dimension
    X_train_redu = tsne.fit_transform(X_train_std)
    #X_train_redu = pca.fit_transform(X_train_std)
    X_test_redu  = tsne.fit_transform(X_test)

    # perform classification

    svm = SVC(kernel='rbf', random_state=0, gamma=GAMMA, C=C, verbose=True)
    svm.fit(X_train_redu, y_train)


    Train_Acc.append(svm.score(X_train_redu, y_train, sample_weight=None))
    Test_Acc.append(svm.score(X_test_redu, y_test, sample_weight=None))

plt.plot(list(range(10, 201, RESOLUTION)), Train_Acc, 'b', list(range(10, 201, RESOLUTION)), Test_Acc, 'r')
plt.title('Perplexity vs TA')
plt.xlabel('Perplexity')
plt.ylabel('Training Accuracy')
plt.show()





