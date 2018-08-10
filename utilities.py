from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace
from matplotlib import cm
import scipy.io as sio


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



def multi_timepoints_plot(X_reduced, y_train, time_points):
    start = 0.5
    stop = 1.0
    number_of_lines = 8
    cm_subsection = linspace(start, stop, number_of_lines)

    wildtype_colors = [cm.Reds(x) for x in cm_subsection]
    mutation_colors = [cm.Blues(x) for x in cm_subsection]
    markers = ['s', 'x']

    X_wild = X_reduced[y_train == 0]
    t_wild = time_points[y_train == 0]
    X_mut = X_reduced[y_train == 1]
    t_mut = time_points[y_train == 1]

    for t, w in zip(np.unique(time_points), wildtype_colors):
        plt.scatter(X_wild[t_wild == t, 0], X_wild[t_wild == t, 1], c=w, label=t)

    plt.hold(True)

    for t, m in zip(np.unique(time_points), mutation_colors):
        plt.scatter(X_mut[t_mut == t, 0], X_mut[t_mut == t, 1], c=m, label=t)

    labels = ('DIV14-W', 'DIV16-W', 'DIV18-W', 'DIV20-W', 'DIV22-W', 'DIV24-W', 'DIV26-W',
              'DIV28-W', 'DIV14-M','DIV16-M','DIV18-M','DIV20-M','DIV22-M','DIV24-M','DIV28-M', 'DIV28-M')

    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(labels, loc='lower right')
    plt.title('Multi-Timepoint Visualization')
    plt.show()


def single_timepoint_plot(X_reduced, y_train, title):
    colors = ['r', 'b']
    markers = ['s', 'x']

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_reduced[y_train == l, 0], X_reduced[y_train == l, 1], c=c, label=l, marker=m)


    plt.xlabel('Dim 1')
    plt.ylabel('DIm 2')
    plt.legend(('Mutation', 'Wild Type'), loc='lower right')
    plt.title(title)
    plt.show()


def load_full_dataset():
    # load the full dataset
    var = sio.loadmat('dataset_flat.mat')
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


def load_custom_dataset(DIV):
    # load the volume dataset
    var = sio.loadmat('dataset.mat')
    data = var['data']
    params = var['params']

    # select the given DIVs

