# Author : Ashwin de Silva
# svm classifier

# import libraries
import scipy.io as sio
from sklearn.svm import SVC
from utilities import plot_decision_regions
from sklearn import metrics
from utilities import *
import numpy as np

# some useeful parameter
MODEL_NAME = 'model_test2'
PATH = '/Users/ashwin/Florey Work MAc/Wild type and Mutation Classification/Parametric t-SNE/Results/{}/{}.mat'.format(MODEL_NAME, MODEL_NAME)
SAVE_PATH =  '/Users/ashwin/Florey Work MAc/Wild type and Mutation Classification/Parametric t-SNE/Results/{}'.format(MODEL_NAME)
TXT_PATH = '/Users/ashwin/Florey Work MAc/Wild type and Mutation Classification/Parametric t-SNE/Results/{}/accuracy_report.txt'.format(MODEL_NAME)

# SVM parameters
GAMMA = 0.001
C = 300.0

# load the data
dict = sio.loadmat(PATH)
X_train = dict['mapped_train_X']
y_train = dict['train_labels']
X_test = dict['mapped_test_X']
y_test = dict['test_labels']

y_train = np.reshape(y_train, (np.size(y_train, 0),))
y_test = np.reshape(y_test, (np.size(y_test, 0),))

# perform classification
svm = SVC(kernel='rbf', random_state=0, gamma=GAMMA, C=C, verbose=True, probability=True)
svm.fit(X_train, y_train)

# performance metrics
y_train_pred = svm.predict(X_train) # the training set predictions
scores_train = svm.predict_proba(X_train)

y_test_pred = svm.predict(X_test)
scores_test = svm.predict_proba(X_test)

ACCURACY_SCORE_TRAIN = svm.score(X_train, y_train, sample_weight=None)
ACCURACY_SCORE_TEST = svm.score(X_test, y_test, sample_weight=None)

NMI_TRAIN = metrics.adjusted_mutual_info_score(y_train, y_train_pred)
NMI_TEST  =  metrics.adjusted_mutual_info_score(y_test, y_test_pred)

# print the summary
print('\n Model Performance Metrics \n')
print('Training Set Accuracy : ', ACCURACY_SCORE_TRAIN)
print('Test Set Accuracy : ', ACCURACY_SCORE_TEST)
print('Training Set Normalized Mutual Information : ', NMI_TRAIN)
print('Test Set Normalized Mutual Information : ', NMI_TEST)
print('\n')
print(metric_report(y_train, y_train_pred))
print(metric_report(y_test, y_test_pred))
print('Jaccard Similarity Score for train set : ', jaccard_score(y_train, y_train_pred))
print('Jaccard Similarity Score for test set : ', jaccard_score(y_test, y_test_pred))

file = open(TXT_PATH, 'w')
file.write('Model Performance Metrics \n')
file.write('Training Set Accuracy : {} \n'.format(ACCURACY_SCORE_TRAIN))
file.write('Test Set Accuracy : {} \n'.format(ACCURACY_SCORE_TEST))
file.write('Training Set Normalized Mutual Information  : {} \n'.format(NMI_TRAIN))
file.write('Test Set Normalized Mutual Information  : {} \n'.format(NMI_TEST))
file.write('Jaccard Similarity Score for train set : {} \n'.format(jaccard_score(y_train, y_train_pred)))
file.write('Jaccard Similarity Score for test set : {} \n'.format(jaccard_score(y_test, y_test_pred)))
file.close()

# plot the lower dimensional embedding
#multi_timepoints_plot(X_train_tsne, y_train, time_points) # for the full dataset
# single_timepoint_plot(X_train, y_train, [7, 8]) # plot the redcued dimensional embedding

# plot the svm decision boundaries for train set
plot_decision_regions(X_train, y_train, classifier=svm) # plot the regional boundaries
plt.savefig(SAVE_PATH + '/svm_model_viz.jpg')

# plot the svm decision boundaries for test set
plot_decision_regions(X_test, y_test, classifier=svm) # plot the regional boundaries
plt.savefig(SAVE_PATH + '/svm_model_viz_test.jpg')

# plot the precision recall call for different threshold probabilities for train set
plot_precision_recall_curve(y_train, scores_train)
plt.savefig(SAVE_PATH + '/PR_train.jpg')

# plot the precision recall call for different threshold probabilities for test set
plot_precision_recall_curve(y_test, scores_test)
plt.savefig(SAVE_PATH + '/PR_test.jpg')

# plot the ROC curve
plot_ROC_curve(y_train, scores_train)
plt.savefig(SAVE_PATH + '/ROC_train.jpg')
plot_ROC_curve(y_test, scores_test)
plt.savefig(SAVE_PATH + '/ROC_test.jpg')
