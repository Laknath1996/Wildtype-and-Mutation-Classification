# author : Ashwin de Silva
# variational autoencoder for dimension reduction


from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Input, Lambda
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from removeParams import *
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from utilities import *


def sampling(args):
    mean, log_stddev = args
    batch = K.shape(mean)[0]
    dim = K.int_shape(mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return mean + K.exp(log_stddev) * epsilon


def vae_loss(inputs, outputs):
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + log_stddev - K.square(mean) - K.square(K.exp(log_stddev)), axis=-1)
    # return the average loss over all images in batch
    total_loss = K.mean(reconstruction_loss + 0*kl_loss)
    return total_loss

# load the dataset
#X_train, y_train, time_points, params = load_full_dataset() # for full datset
X_train, y_train, params = load_single_dataset() # for singular dataset

# remove Low Variance Params
X_train = removeParams(X_train, 30)

# scale the trainig examples
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)

# define the layer sizes
input_size = np.shape(X_train)[1]
hidden_size_1 = 16
hidden_size_2 = 5
code_size = 2

# the model

# encoder
input_vec = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size_1, activation='relu')(input_vec)
hidden_2 = Dense(hidden_size_2, activation='relu')(hidden_1)

# middle layers
mean = Dense(code_size)(hidden_2)
log_stddev = Dense(code_size)(hidden_2)
latent_vector = Lambda(sampling, output_shape=(2,), name='z')([mean, log_stddev])

# decoder
hidden_3 = Dense(hidden_size_2, activation='relu')(latent_vector)
hidden_4 = Dense(hidden_size_1, activation='relu')(hidden_3)
output_vec = Dense(input_size, activation='sigmoid')(hidden_4)

# define the model
autoencoder = Model(input_vec, output_vec)
autoencoder.compile(optimizer=Adam(lr=0.00001), loss=[vae_loss])
history = autoencoder.fit(X_train, X_train, epochs=1000, batch_size=16)
encoder = Model(input_vec, latent_vector)

# reduce dimensions using the encoder
X_train_vae = encoder.predict(X_train)

# plot the results
#multi_timepoints_plot(X_train_vae, y_train, time_points) # for the full dataset
single_timepoint_plot(X_train_vae, y_train, 'DIV28')
