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
from sklearn import metrics
from core import *
from utils import *
import scipy.io as sio
import umap

# define useful parameters

# selection paramters
TIME_POINTS = [7,8]
REMOVE_LOW_VARIANCE_PARAMS = True
LOW_VARIANCE_THRESHOLD = 30
ELIMINATE_REDUNDANT_PARAMS = False

# outlier removal parameters
REMOVE_OUTLIERS = False
M = 3

# T_SNE paramters
PERPLEXITY = 15
EARLY_EXAGGERATION = 12
LEARNING_RATE = 30
N_ITER = 1000
MIN_GRAD_NORM = 1e-7
RANDOM_STATE = 25

# SVM parameters
GAMMA = 0.001
C = 300.0

# define redundant features
redundant_params  = ['cvSCBDuration','cvSCBSize','cvDuration','cvInbis',
    'cvJitter','mfrRatio','cvNBSI']

# load the dataset
DIV = TIME_POINTS
X_train, y_train, time_points, params = load_custom_dataset(DIV, 1)  # for custom dataset

# remove redunduant params
if (ELIMINATE_REDUNDANT_PARAMS):
    X_train, params = removeRedundantParams(X_train, redundant_params)

# remove the low variance params
if (REMOVE_LOW_VARIANCE_PARAMS):
    X_train = removeParams(X_train, LOW_VARIANCE_THRESHOLD)

# remove the outliers
if (REMOVE_OUTLIERS):
    X_train, y_train = reject_outliers(X_train, y_train, m=M)

# standardize the data
sc = StandardScaler()
#sc = MinMaxScaler()
X_train_std = sc.fit_transform(X_train)

# save the X_train_std to a .mat file
#dict = {'X' : X_train_std, 'Y' : y_train}
#sio.savemat('X_train_std.mat', dict)



# implement dimension reduction algorithm
# tsne = TSNE(n_components=2, perplexity=PERPLEXITY, early_exaggeration=EARLY_EXAGGERATION , learning_rate=LEARNING_RATE,
#             init='random', n_iter=N_ITER,
#             min_grad_norm=MIN_GRAD_NORM, verbose=True, random_state=RANDOM_STATE) # this model worked well for DIV28

X_train_redu = umap.UMAP(n_neighbors=15,
                      min_dist=0.1,
                      metric='correlation').fit_transform(X_train_std)
# pca = PCA(n_components=2)

# implement the parametric t-sne

# high_dims = 15 #X_train_std.shape[1]
# num_outputs = 2
# perplexity =  0
# ptSNE = Parametric_tSNE(high_dims, num_outputs, perplexity)
# X_train_redu = ptSNE.transform(X_train_std)


# reduce the dimension
# X_train_redu = tsne.fit_transform(X_train_std)


# perform classification
svm = SVC(kernel='rbf', random_state=0, gamma=GAMMA, C=C, verbose=True, probability=True)
svm.fit(X_train_redu, y_train)


# performance metrics
y_train_pred = svm.predict(X_train_redu) # the training set predictions
scores = svm.predict_proba(X_train_redu)
print(np.shape(scores))
ACCURACY_SCORE = svm.score(X_train_redu, y_train, sample_weight=None)
NMI = metrics.adjusted_mutual_info_score(y_train, y_train_pred)


# print the summary
print('\n')
print('Time Points Used : ', DIV)
print('Total Datapoints used : ', np.size(X_train_redu, 0))
print('Remove Low Variance Variables : ', REMOVE_LOW_VARIANCE_PARAMS )
if (REMOVE_LOW_VARIANCE_PARAMS):
    print('Low Variance Threshold : ', LOW_VARIANCE_THRESHOLD)
print('Remove Outliers : ', REMOVE_OUTLIERS)
if (REMOVE_OUTLIERS):
    print('Absolute distance to the median : ', M)
print('Perplexity of TSNE : ', PERPLEXITY)
print('\n Model Performance Metrics \n')
print('Training Set Accuracy : ', ACCURACY_SCORE)
print('Training Set Normalized Mutual Information : ', NMI)
print('\n')
print(metric_report(y_train, y_train_pred))
print('Jaccard Similarity Score : ', jaccard_score(y_train, y_train_pred))
#print('Model Params : ', params)



# plot the lower dimensional embedding
#multi_timepoints_plot(X_train_tsne, y_train, time_points) # for the full dataset
single_timepoint_plot(X_train_redu, y_train, DIV) # plot the redcued dimensional embedding

# plot the svm decision boundaries
plot_decision_regions(X_train_redu, y_train, classifier=svm) # plot the regional boundaries

# plot the precision recall call for different threshold probabilities
plot_precision_recall_curve(y_train, scores)

# plot the ROC curve
plot_ROC_curve(y_train, scores)






