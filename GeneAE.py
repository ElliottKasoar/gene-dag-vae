#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from load import save_h5ad, load_h5ad
#from loss import NB_loglikelihood

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

import tensorflow as tf
import keras.backend as K

from keras.utils import plot_model
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU

if int(tf.__version__[0]) < 2:
    tf2_flag = False
else:
    tf2_flag = True

# NEED TO PUT THIS IN DIFFERENT FILE with code from temp.py
# create directory 'models' if it doesn't exist
# base_dir = '.'
# plots_dir = base_dir + '/plots'
# models_dir = plots_dir + '/models'

# from pathlib import Path
# for i in [plots_dir, models_dir]:
#     Path(i).mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model parameters
# =============================================================================

# Size of encoded representation
encoding_dim = 256

# Fraction of data used in training
train_size = 0.7

epochs = 250
batch_size = 256

# =============================================================================
# Load data
# =============================================================================

adata = load_h5ad('preprocessed')
X = adata.X
print (X.shape)

# Input shape
input_dim = X.shape[1]
input_shape = (input_dim,)

# scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1))

X = scaler.fit_transform(X)

# scale = X.max(axis=0)
# X = np.divide(X, scale)

X_train, X_test = train_test_split(X, train_size=train_size)

# =============================================================================
# Build models
# =============================================================================

model = 'nb'
#model = 'gaussian'

# Encoder Model
input   = Input(shape=input_shape)
encoded = Dense(1024, activation='relu')(input)
encoded = Dense(512, activation='relu')(encoded)
latent  = Dense(encoding_dim, activation='relu')(encoded)

encoder = Model(input, latent, name='encoder')

# plot_model(encoder, to_file=models_dir + '/' + model + '_encoder.png',
#            show_shapes=True, show_layer_names=True)

# Encoded representation of the input (with sparsity contraint via regularizer)
# encoded = Dense(encoding_dim, activation='relu', 
#                 activity_regularizer=regularizers.l1(10e-5))(VAE_input)

# Decoder Model 
# Lossy reconstruction of the input
lat_input = Input(shape=(encoding_dim,))
decoded   = Dense(512, activation='relu')(lat_input)
decoded   = Dense(128, activation='relu')(decoded)

if model == 'gaussian': 
    outputs = Dense(input_dim, activation='sigmoid')(decoded)

elif model == 'nb':
    MeanAct = lambda a: tf.clip_by_value(K.exp(a), 1e-5, 1e6)
    DispAct = lambda a: tf.clip_by_value(tf.nn.softplus(a), 1e-4, 1e4)

    mu = Dense(input_dim, activation = MeanAct, name='mu')(decoded)
    disp = Dense(input_dim, activation = DispAct, name='disp')(decoded)
    # pi = Dense(input_dim, activation = 'sigmoid', name='pi')(decoded)

    outputs = [mu, disp]

decoder = Model(lat_input, outputs, name='decoder')

# plot_model(decoder, to_file=models_dir + '/' + model + '_decoder.png',
#            show_shapes=True, show_layer_names=True)         

# Autoencoder Model
outputs = decoder(encoder(input))
autoencoder = Model(input, outputs, name='autoencoder')

print (autoencoder.summary())
# plot_model(autoencoder, to_file=models_dir + '/' + model + '_autoencoder.png',
#            show_shapes=True, show_layer_names=True)         

# =============================================================================
# Define custom loss
# =============================================================================

def NB_loglikelihood(r):

    def loss (y_true, y_pred):
        y = y_true[0]
        mu = y_pred[0]
        
        if tf2_flag:
            l1 = tf.math.lgamma(y+r) - tf.math.lgamma(r) - tf.math.lgamma(y+1.0)
            l2 = y * tf.math.log(mu/(r+mu)) + r * tf.math.log(r/(r+mu))
        else:
            l1 = tf.lgamma(y+r) - tf.lgamma(r) - tf.lgamma(y+1.0)
            l2 = y * tf.log(mu/(r+mu)) + r * tf.log(r/(r+mu))
            
        log_likelihood = l1 + l2

        return  -K.sum(log_likelihood, axis=-1)

    return loss

if model == 'nb':
    autoencoder.compile(optimizer='adam', loss=NB_loglikelihood(outputs[1]))


# alternative method: add_loss does not require you to restrict the parameters
# of the loss to y_pred and y_actual 
# may change to this

'''
def NB_loglikelihood(y, mu, r):

    if tf2_flag:
        l1 = tf.math.lgamma(y+r) - tf.math.lgamma(r) - tf.math.lgamma(y+1.0)
        l2 = y * tf.math.log(mu/(r+mu)) + r * tf.math.log(r/(r+mu))
    else:
        l1 = tf.lgamma(y+r) - tf.lgamma(r) - tf.lgamma(y+1.0)
        l2 = y * tf.log(mu/(r+mu)) + r * tf.log(r/(r+mu))
    
    log_likelihood = l1 + l2
    
    return log_likelihood


if model == 'nb':
    reconstruction_loss = - K.sum(NB_loglikelihood(input, outputs[0],
                                                    outputs[1]), axis=-1)

    print (K.print_tensor(NB_loglikelihood(input, outputs[0], outputs[1])))
    print (K.print_tensor(reconstruction_loss))

  autoencoder.add_loss(K.mean(reconstruction_loss))
    autoencoder.add_loss(reconstruction_loss)

    autoencoder.compile(optimizer='adam', loss=None)
'''

if model == 'gaussian':
    autoencoder.compile(optimizer='adam', loss='mse')

# =============================================================================
# Train model
# =============================================================================

print (outputs[1].shape)

loss = autoencoder.fit(X_train,
                       [X_train, X_train],
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True)

autoencoder.save('AE.h5')

# =============================================================================
# Test model
# =============================================================================

encoded_data = encoder.predict(X_test)
decoded_data = decoder.predict(encoded_data)


