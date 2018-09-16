from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace
from matplotlib import cm
import scipy.io as sio
from sklearn.metrics import classification_report, jaccard_similarity_score, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
import scikitplot as skplt


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    plt.xlabel('Dim 1')
    plt.ylabel('DIm 2')
    plt.title('Reduced Dimension Visualization')
    # plt.show()


def multi_timepoints_plot(X_reduced, y_train, time_points):
    start = 0.5
    stop = 1.0
    number_of_lines = len(np.unique(time_points))
    cm_subsection = linspace(start, stop, number_of_lines)

    wildtype_colors = [cm.Reds(x) for x in cm_subsection]
    mutation_colors = [cm.Blues(x) for x in cm_subsection]

    X_wild = X_reduced[y_train == 0]
    t_wild = time_points[y_train == 0]
    X_mut = X_reduced[y_train == 1]
    t_mut = time_points[y_train == 1]

    dict = {1: 'DIV14', 2: 'DIV16', 3: 'DIV18', 4: 'DIV20', 5: 'DIV22', 6: 'DIV24', 7: 'DIV26', 8: 'DIV28'}

    for t, w in zip(np.unique(time_points), wildtype_colors):
        plt.scatter(X_wild[t_wild == t, 0], X_wild[t_wild == t, 1], c=w, label=t)

    plt.hold(True)

    for t, m in zip(np.unique(time_points), mutation_colors):
        plt.scatter(X_mut[t_mut == t, 0], X_mut[t_mut == t, 1], c=m, label=t)

    labels = ('DIV14-W', 'DIV16-W', 'DIV18-W', 'DIV20-W', 'DIV22-W', 'DIV24-W', 'DIV26-W',
              'DIV28-W', 'DIV14-M', 'DIV16-M', 'DIV18-M', 'DIV20-M', 'DIV22-M', 'DIV24-M', 'DIV28-M', 'DIV28-M')

    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(labels, loc='lower right')
    plt.title('Multi-Timepoint Visualization')
    plt.show()


def single_timepoint_plot(X_reduced, y_train, DIV):
    dict = {1: 'DIV14', 2: 'DIV16', 3: 'DIV18', 4: 'DIV20', 5: 'DIV22', 6: 'DIV24', 7: 'DIV26', 8: 'DIV28'}
    colors = ['r', 'b']
    markers = ['s', 'x']

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_reduced[y_train == l, 0], X_reduced[y_train == l, 1], c=c, label=l, marker=m)

    title = ', '.join([dict[x] for x in DIV])

    plt.xlabel('Dim 1')
    plt.ylabel('DIm 2')
    plt.legend(('Mutation', 'Wild Type'), loc='lower right')
    plt.title(title)
    plt.show()


def load_full_dataset(sel):
    # load the full dataset
    if (sel == 1):
        var = sio.loadmat('/Users/ashwin/Florey Work MAc/Wild type and Mutation Classification/dataset/dataset_R853Q.mat')
    if (sel == 2):
        var = sio.loadmat('/Users/ashwin/Florey Work MAc/Wild type and Mutation Classification/dataset/dataset_R1882Q.mat')
    data = var['data']
    params = var['params']

    params = params[0]
    names = []
    for i in params:
        names.append(str(i)[2:-2])

    X_train, y_train, time_points = data[:, :-2], data[:, -2], data[:, -1]

    return X_train, y_train, time_points, params


def load_single_dataset():
    # load a single dataset
    var = sio.loadmat('dataset_28.mat')
    data = var['dataset_28']
    params = var['params']

    params = params[0]
    names = []
    for i in params:
        names.append(str(i)[2:-2])

    X_train, y_train = data[:, :-1], data[:, -1]

    return X_train, y_train, params


def load_custom_dataset(DIV, sel ):
    # load the volume dataset
    if (sel == 1):
        var = sio.loadmat('/Users/ashwin/Florey Work/Wild type and Mutation Classification/dataset/dataset_R853Q.mat')
    if (sel == 2):
        var = sio.loadmat('/Users/ashwin/Florey Work/Wild type and Mutation Classification/dataset/dataset_R1882Q.mat')
    data = var['data']
    params = var['params']

    params = params[0]
    names = []
    for i in params:
        names.append(str(i)[2:-2])

    # select the given DIVs
    X_train, y_train, time_points = data[:, :-2], data[:, -2], data[:, -1]
    X_train = X_train[np.isin(time_points, DIV)]
    y_train = y_train[np.isin(time_points, DIV)]
    time_points = time_points[np.isin(time_points, DIV)]

    return X_train, y_train, time_points, params


def removeRedundantParams(X, redundant_params):
    # define the param dict
    params_dict = {'meanSCBDuration': 1, 'stdSCBDuration': 2, 'cvSCBDuration': 3,
              'rangeSCBDuration': 4, 'meanSCBSize': 5, 'stdSCBSize': 6, 'cvSCBSize': 7, 'rangeSCBSize': 8,
              'avgChansInNB': 9, 'meanDuration': 10, 'stdDuration': 11, 'cvDuration': 12,
              'rangeDuration': 13, 'meanInbis': 14, 'stdInbis': 15, 'cvInbis': 16, 'rangeInbis': 17,
              'meanJitter': 18, 'stdJitter': 19, 'cvJitter': 20, 'rangeJitter': 21, 'NBRate': 22, 'totNoOfSpikes': 23,
              'totNBAmp': 24, 'avgNBPeakAmp': 25, 'avgNBTimeAmp': 26, 'mfrAll': 27, 'mfrIn': 28, 'mfrOut': 29,
              'mfrRatio': 30, 'noOfSpikingChans': 31, 'chansInNBs': 32, 'avgSpikesInNB': 33, 'avgAmp': 34,
              'spikesInNBs': 35, 'meanNBSI': 36, 'stdNBSI': 37, 'cvNBSI': 38, 'rangeNBSI': 39, 'Kappa': 40};

    # get the redundant indices
    redundant_indices = [params_dict[x] for x in redundant_params]
    redundant_indices = np.subtract(redundant_indices, 1)

    # remove the redundant features
    X_new = np.delete(X, redundant_indices, 1)

    # get the param ids
    params = params_dict.keys()
    params = [x for x in params if x not in redundant_params]

    return X_new, params


def reject_outliers(X, y, m):
    n = np.size(X, 1)
    indices = []
    for i in range(n):
        data = X[:, i]
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        ix = s < m
        ix = np.logical_not(ix)
        ix = np.array(ix, int)
        indices.extend([i for i, x in enumerate(ix) if x])

    indices = np.unique(indices)
    X = np.delete(X, indices, axis=0)
    y = np.delete(y, indices, axis=0)

    return X, y


def metric_report(y_true, y_pred):
    target_names = ['wildtype', 'mutation']
    return classification_report(y_true, y_pred, target_names=target_names)

def jaccard_score(y_true, y_pred):
    return jaccard_similarity_score(y_true, y_pred)


def plot_precision_recall_curve(y_true, scores ):
    skplt.metrics.plot_precision_recall_curve(y_true, scores)
    plt.show()


def plot_ROC_curve(y_true, scores):
    skplt.metrics.plot_roc_curve(y_true, scores)
    plt.show()




