# author : Ashwin de Silva
# autoencoder for dimension reduction

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot
import scipy.io as sio
from numpy import linspace
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from removeParams import *

var = sio.loadmat('dataset_flat.mat')
data = var['data']
params = var['params']

params = params[0]
names = []
for i in params:
    names.append(str(i)[2:-2])

X_train, y_train = data[:, :-2], data[:, -2]
time_points = data[:, -1]

X_train = removeParams(X_train, 30)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)

input_size = np.shape(X_train)[1]
hidden_size_1 = 16
hidden_size_2 = 5
code_size = 2

input_vec = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size_1, activation='sigmoid')(input_vec)
hidden_2 = Dense(hidden_size_2, activation='sigmoid')(hidden_1)
code = Dense(code_size, activation='relu')(hidden_2)
hidden_3 = Dense(hidden_size_2, activation='sigmoid')(code)
hidden_4 = Dense(hidden_size_1, activation='sigmoid')(hidden_3)
output_vec = Dense(input_size, activation='sigmoid')(hidden_4)

autoencoder = Model(input_vec,output_vec)

autoencoder.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')

history = autoencoder.fit(X_train, X_train, epochs=1000, batch_size=32)

encoder = Model(input_vec, code)
X_train_pca = encoder.predict(X_train)

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

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()
