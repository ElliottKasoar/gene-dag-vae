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
from keras.layers import multiply

# (Almost reproducible)
# np.random.seed(1337)
# tf.random.set_seed(1235)

# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

# tf.config.experimental_run_functions_eagerly(True)

from time import time
from keras.callbacks import TensorBoard

if int(tf.__version__[0]) < 2:
    tf2_flag = False
else:
    tf2_flag = True

# NEED TO PUT THIS IN DIFFERENT FILE with code from temp.py
# create directory 'models' if it doesn't exist
base_dir = '.'
plots_dir = base_dir + '/plots'
models_dir = plots_dir + '/models'

from pathlib import Path
for i in [plots_dir, models_dir]:
    Path(i).mkdir(parents=True, exist_ok=True)

# =============================================================================
# Model parameters
# =============================================================================

# Size of encoded representation
encoding_dim = 256

# Fraction of data used in training
train_size = 0.7

epochs = 10
batch_size = 16

# =============================================================================
# Load data
# =============================================================================

adata = load_h5ad('preprocessed')    # need to add code to ensure this exists	

# Input shape
input_dim = adata.X.shape[1]
input_shape = (input_dim,)

# scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
# scaler = StandardScaler()
x = 1e-10
gene_scaler = MinMaxScaler(feature_range=(x, 1-x))
sf_scaler = MinMaxScaler(feature_range=(x, 1-x))

adata.X = gene_scaler.fit_transform(adata.X)
adata.obs['sf'].values[:] = sf_scaler.fit_transform(adata.obs['sf'].values.reshape(-1, 1)).reshape(1, -1)

# adata.obs['sf'].values[:] = 1

# scale = X.max(axis=0)
# X = np.divide(X, scale)

X_train, X_test = train_test_split(adata.X, 
                                   train_size=train_size, shuffle=False)
sf_train, sf_test = train_test_split(adata.obs['sf'].values, 
                                     train_size=train_size, shuffle=False)

# =============================================================================
# Build models
# =============================================================================

use_sf = True
calc_sf_count = True
calc_sf_obs = False
model = 'zinb'
# model = 'nb'
#model = 'gaussian'

# Encoder Model
count_input = Input(shape=input_shape, name='count_input')
x = Dense(1024, activation='relu')(count_input)
x = Dense(512, activation='relu')(x)
latent = Dense(encoding_dim, activation='relu', name='latent')(x)

encoder = Model(count_input, latent, name='encoder')

plot_model(encoder, to_file=models_dir + '/' + model + '_encoder.png',
           show_shapes=True, show_layer_names=True)

# Size factors
# try learning sf from the input data, hopefully will prevent factors from being so large
#sf_input = Input(shape-input_shape, name='size_factor_input')
if use_sf:
    if calc_sf_count:
        sf = Dense(1, activation='relu')(count_input)
        sf_model = Model(count_input, sf, name='sf_model')
    elif calc_sf_obs:
        sf_input = Input(shape=(1,), name='size_factor_input')
        sf = Dense(1, activation='relu')(sf_input)
        sf_model = Model(sf_input, sf, name='sf_model')
    else:
        sf = Input(shape=(1,), name='size_factor_input')

# Decoder Model 
# Lossy reconstruction of the input
lat_input = Input(shape=(encoding_dim,))
x = Dense(512, activation='relu')(lat_input)
x = Dense(1024, activation='relu')(x)

if model == 'gaussian': 
    decoder_outputs = Dense(input_dim, activation='sigmoid')(x)

elif model == 'nb' or model == 'zinb':

    MeanAct = lambda a: tf.clip_by_value(K.exp(a), 1e-5, 1e6)
    DispAct = lambda a: tf.clip_by_value(tf.nn.softplus(a), 1e-4, 1e4)

    mu = Dense(input_dim, activation = MeanAct, name='mu')(x)
    disp = Dense(input_dim, activation = DispAct, name='disp')(x)

    # Multiply mu by sf here, broadcasting not done >> need to use K.repeat_elements
    # sf_reshape = tf.reshape(sf, [-1,1]) # Column vector reshaped to row vector    
    # sf = K.repeat_elements(sf, input_dim, 1)
    # mu_sf = mu * sf_reshape # Multiply by sf here
    
    if use_sf:
        mu_sf = multiply([mu, sf])
        decoder_outputs = [mu_sf, disp]
        
        if calc_sf_count:
            decoder_inputs = [lat_input, count_input]
        elif calc_sf_obs:
            decoder_inputs = [lat_input, sf_input]
        else:
            decoder_inputs = [lat_input, sf]
            
    else:
        decoder_inputs = lat_input
        decoder_outputs = [mu, disp]
    
    if model == 'zinb':
        # Activation is sigmoid because values restricted to [0,1]
        pi = Dense(input_dim, activation = 'sigmoid', name='pi')(x)
        decoder_outputs.append(pi)    


decoder = Model(decoder_inputs, decoder_outputs, name='decoder')

plot_model(decoder, to_file=models_dir + '/' + model + '_decoder.png',
           show_shapes=True, show_layer_names=True)         

# Autoencoder Model

if use_sf:
    if calc_sf_count:    
        AE_inputs = count_input
        AE_outputs = decoder([encoder(count_input), count_input])
    elif calc_sf_obs:
        AE_inputs = [count_input, sf_input]
        AE_outputs = decoder([encoder(count_input), sf_input])
    else:
        AE_inputs = [count_input, sf]
        AE_outputs = decoder([encoder(count_input), sf])
else:
    AE_inputs = count_input
    AE_outputs = decoder(encoder(count_input))

autoencoder = Model(AE_inputs, AE_outputs, name='autoencoder')

print (autoencoder.summary())
plot_model(autoencoder, to_file=models_dir + '/' + model + '_autoencoder.png',
           show_shapes=True, show_layer_names=True)         

# =============================================================================
# Define custom loss
# =============================================================================

def NB_loglikelihood(outputs):
    
    def loss (y_true, y_pred):

        eps = 1e-10 # Prevent NaN loss value?        

        y = y_true
        
        mu = outputs[0]
        r = outputs[1]
        
        if tf2_flag:
            l1 = tf.math.lgamma(y+r+eps) - tf.math.lgamma(r+eps) - tf.math.lgamma(y+1.0)
            l2 = y * tf.math.log((mu+eps)/(r+mu+eps)) + r * tf.math.log((r+eps)/(r+mu+eps))
        else:
            l1 = tf.lgamma(y+r) - tf.lgamma(r) - tf.lgamma(y+1.0)
            l2 = y * tf.log(mu/(r+mu)) + r * tf.log(r/(r+mu))
            
        log_likelihood = l1 + l2

        return  -K.sum(log_likelihood, axis=-1)

    return loss


def ZINB_loglikelihood(outputs):
    
    # return scalar loss for each data point 
    def loss (y_true, y_pred):
        
        eps = 1e-10 # Prevent NaN loss value?
        
        y = y_true
        
        mu = outputs[0]
        r = outputs[1]
        pi = outputs[2]
        
        if tf2_flag:
            l1 = tf.math.lgamma(y+r+eps) - tf.math.lgamma(r+eps) - tf.math.lgamma(y+1.0)
            l2 = y * tf.math.log((mu+eps)/(r+mu+eps)) + r * tf.math.log((r+eps)/(r+mu+eps))
        else:
            l1 = tf.lgamma(y+r) - tf.lgamma(r) - tf.lgamma(y+1.0)
            l2 = y * tf.log(mu/(r+mu)) + r * tf.log(r/(r+mu))
            
        nb_log_likelihood = l1 + l2

        if tf2_flag:
            case_zero = tf.math.log(eps + pi + (1.0 - pi) * tf.math.pow((r/(r+mu+eps)), r))
            case_nonzero = tf.math.log(1.0 - pi + eps) + nb_log_likelihood
        else:
            case_zero = tf.log(pi + (1.0-pi) * tf.pow((r/(r+mu)), r))
            case_nonzero = tf.log(1.0-pi) + nb_log_likelihood

        # whenever a count value < 1e-8, use case_zero for the log-likelihood
        zinb_log_likelihood = tf.where(tf.less(y, 1e-8), case_zero, case_nonzero)
        # return scalar for each cell; 
        # cell likelihood = product of likelihoods over genes
        return  -K.sum(zinb_log_likelihood, axis=1)

    return loss


# Loss function run thrice (once for each output) but only one used
if model == 'zinb':
    autoencoder.compile(optimizer='adam',
                        loss=ZINB_loglikelihood(AE_outputs),
                        loss_weights=[1., 0.0, 0.0])

if model == 'nb':
    autoencoder.compile(optimizer='adam',
                        loss=NB_loglikelihood(AE_outputs),
                        loss_weights=[1., 0.0])

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

# from tb_callback import MyTensorBoard
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

# =============================================================================
# Train model
# =============================================================================

if use_sf and not calc_sf_count:
    fit_x = [X_train, sf_train]
else:
    fit_x = X_train
    
if model == 'gaussian':
    fit_y = X_train
elif model == 'nb':
    fit_y = [X_train, X_train]
elif model == 'zinb':
    fit_y = [X_train, X_train, X_train]

# Pass adata.obs['sf'] as an input. 2nd, 3rd elements of y not used
loss = autoencoder.fit(fit_x, fit_y, epochs=epochs, batch_size=batch_size,
                       shuffle=False, callbacks=[tensorboard])

autoencoder.save('AE.h5')

# =============================================================================
# Test model
# =============================================================================

if use_sf:
    if calc_sf_count:
        encoded_data = encoder.predict(adata.X)
        decoded_data = decoder.predict([encoded_data, adata.X])
    else:
        encoded_data = encoder.predict(adata.X)
        decoded_data = decoder.predict([encoded_data, adata.obs['sf'].values])
else:
    encoded_data = encoder.predict(adata.X)
    decoded_data = decoder.predict(encoded_data)

# adata.X = gene_scaler.inverse_transform(decoded_data[0])
save_h5ad(adata, 'denoised')


def test_sf():
    
    if calc_sf_count:
        sf = sf_model.predict(adata.X)
    else:
        sf = sf_model.predict(adata.obs['sf'].values)
        
    return sf


def test_AE():
    
    if use_sf:
        if calc_sf_count:
            encoded_data = encoder.predict(X_train[0:batch_size])
            decoded_data = decoder.predict([encoded_data, X_train[0:batch_size]])
        else:
            encoded_data = encoder.predict(X_train[0:batch_size])
            decoded_data = decoder.predict([encoded_data, adata.obs['sf'].values[0:batch_size]])
    else:
        encoded_data = encoder.predict(X_train[0:batch_size])
        decoded_data = decoder.predict(encoded_data)
        
    return decoded_data
