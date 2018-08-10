# author : Ashwin de Silva
# feature importance with random forests

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# load the dataset

var = sio.loadmat('dataset_flat.mat')
data = var['data']
params = var['params']

params = params[0]
names = []
for i in params :
    names.append(str(i)[2:-2])

# select the train data and the labels

X, y = data[:, :-2], data[:, -2]
time_points = data[:, -1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0)
X_train, y_train = X, y

# define the random forest classifier

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

# get the features importances from the forest

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

#plot the importances

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), params[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# select and scale the dataset with the important features

X_train = X_train[:, indices[:3]]
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

# perform dimension reduction

# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train_std)

#X_train_pca = LDA(X_train_std, y_train)

tsne = TSNE(n_components=2, perplexity=200, early_exaggeration=12, learning_rate=200, init='random', n_iter=1000,
            min_grad_norm=1e-7, verbose=True)
X_train_pca = tsne.fit_transform(X_train_std)

# plot the results

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
t_mut = time_points[y_train == 1]
for t, w in zip(np.unique(time_points), wildtype_colors):
     plt.scatter(X_wild[t_wild == t, 0], X_wild[t_wild == t, 1], c=w, label=t)

plt.hold(True)

for t, m in zip(np.unique(time_points), mutation_colors):
     plt.scatter(X_mut[t_mut == t, 0], X_mut[t_mut == t, 1], c=m, label=t)

plt.show()
