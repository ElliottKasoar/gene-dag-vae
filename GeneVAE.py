#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from load import save_h5ad, load_h5ad
#from loss import NB_loglikelihood
from temp import save_figure, plotTSNE

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

import tensorflow as tf
import keras.backend as K

from keras.utils import plot_model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import multiply, Lambda
from keras.optimizers import Adam

import scanpy as sc

# (Almost reproducible)
# np.random.seed(1337)
# tf.random.set_seed(1234)

debug = False

if debug:
    from tensorflow.python import debug as tf_debug
    sess = K.get_session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline") #Spyder
    K.set_session(sess)

# tf.config.experimental_run_functions_eagerly(True)

from time import time
from keras.callbacks import TensorBoard

if int(tf.__version__[0]) == 2:
    tf2_flag = True
else:
    tf2_flag = False

# To do: move this
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

latent_dim = 32 # Size of encoded representation

train_size = 0.9 # Fraction of data used in training

epochs = 200
batch_size = 512

beta_vae = 2

gene_layers = 3 # Hidden layers between input and latent layers
gene_nodes = 512 # Size of initial hidden layer
gene_flat = False # Keep all hidden layers flat (else halve each layer)
gene_alpha = 0.2 # LeakyReLU alpha
gene_momentum=0.8 # BatchNorm momentum
gene_dropout = 0.2 # Dropout rate

sf_layers = 4 # Hidden layers between input and latent layers
sf_nodes = 1024 # Size of initial hidden layer (half each layer)
sf_alpha = 0.2 # LeakyReLU alpha
momentum=sf_momentum=0.8 # BatchNorm momentum
sf_dropout = 0.2 # Dropout rate

# Adam optimiser parameters
lr = 0.0005 #Default = 0.001 (Adam default = 0.001)
beta_1=0.75 # Default = 0.9 (Adam default = 0.9)
beta_2=0.99 # Default = 0.999 (Adam default = 0.999)

# =============================================================================
# Load data
# =============================================================================

adata = load_h5ad('preprocessed')    # need to add code to ensure this exists	

# Input shape
input_dim = adata.X.shape[1]
input_shape = (input_dim,)

# scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
# scaler = StandardScaler()
x = 0
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
# Sampling
# =============================================================================

# reparametrisation trick
def sampling(args):
    mean, log_var = args
    epsilon_std = 1.0
    epsilon_mean = 0.0
    batch = K.shape(mean)[0]
    dim = K.int_shape(mean)[1]
    epsilon = K.random_normal(shape=(batch, dim),
                              mean=epsilon_mean, stddev=epsilon_std)
    return mean + K.exp(0.5 * log_var) * epsilon

# =============================================================================
# Build models
# =============================================================================

use_sf = False
learn_sf = False
#model = 'zinb'
#model = 'nb'
model = 'gaussian'     # likelihood of (input) data conditioned on Gaussian model => mse loss
vae = True

# =============================================================================
# Encoder Model: count data
# =============================================================================

count_input = Input(shape=input_shape, name='count_input')
x = Dense(gene_nodes)(count_input)
x = BatchNormalization(momentum=gene_momentum)(x)
x = LeakyReLU(gene_alpha)(x)
x = Dropout(gene_dropout)(x)

for i in range(1, gene_layers):
    
    if gene_flat:
        nodes = gene_nodes
    else:
        nodes = gene_nodes // (2**i)
        if nodes < latent_dim:
            print("Warning: layer has fewer nodes than latent layer")
            print(f'Layer nodes: {nodes}. Latent nodes: {latent_dim}')
    
    x = Dense(nodes)(x)
    x = BatchNormalization(momentum=gene_momentum)(x)
    x = LeakyReLU(gene_alpha)(x)
    x = Dropout(gene_dropout)(x)

if vae:
    z_mean = Dense(latent_dim, name='latent_mean')(x)
    z_log_var = Dense(latent_dim, name='latent_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(count_input, [z_mean, z_log_var, z], name='encoder')

else:
    latent = Dense(latent_dim, activation='relu', name='latent')(x)
    encoder = Model(count_input, latent, name='encoder')

plot_model(encoder, to_file=models_dir + '/' + model + '_encoder.png',
           show_shapes=True, show_layer_names=True)

# =============================================================================
# Size factors
# =============================================================================

if use_sf:
    if learn_sf:
        
        x = Dense(sf_nodes)(count_input)
        x = BatchNormalization(momentum=sf_momentum)(x)
        x = LeakyReLU(sf_alpha)(x)
        x = Dropout(sf_dropout)(x)
        
        for i in range(1, sf_layers):
            nodes = sf_nodes // (2**i)
            x = Dense(nodes)(x)
            x = BatchNormalization(momentum=sf_momentum)(x)
            x = LeakyReLU(sf_alpha)(x)
            x = Dropout(sf_dropout)(x)
        
        if vae:
            sf_mean = Dense(1, name='sf_mean')(x)
            sf_log_var = Dense(1, name='sf_log_var')(x)
            
            sf = Lambda(sampling, output_shape=(1,))([sf_mean, sf_log_var])
            sf_encoder = Model(count_input, [sf_mean, sf_log_var, sf], name='sf_encoder')
        else:
            sf = Dense(1, name='sf_latent')(x)
            sf_encoder = Model(count_input, sf, name='sf_encoder')

    else:
        sf_input = Input(shape=(1,), name='size_factor_input')
        #sf = Input(shape=(1,), name='size_factor_input')

if use_sf and learn_sf:
    plot_model(sf_encoder, to_file=models_dir + '/' + model + '_sf_encoder.png',
           show_shapes=True, show_layer_names=True)         

# =============================================================================
# Decoder Model 
# =============================================================================

# Lossy reconstruction of the input
lat_input = Input(shape=(latent_dim,))

if gene_flat:
    x = Dense(gene_nodes)(lat_input)
else:
    nodes = gene_nodes // (2 ** (gene_layers - 1))
    x = Dense(nodes)(lat_input)

x = BatchNormalization(momentum=gene_momentum)(x)
x = LeakyReLU(gene_alpha)(x)
x = Dropout(gene_dropout)(x)

for i in range(1, gene_layers):
    if gene_flat:
        nodes = gene_nodes
    else:
        nodes = gene_nodes // (2 ** (gene_layers - (i+1)))

    x = Dense(nodes)(x)
    x = BatchNormalization(momentum=gene_momentum)(x)
    x = LeakyReLU(gene_alpha)(x)
    x = Dropout(gene_dropout)(x)

# Decoder inputs
if use_sf:
    if learn_sf:
        decoder_inputs = [lat_input, count_input]
    else:
        decoder_inputs = [lat_input, sf_input]
        #decoder_inputs = [lat_input, sf]
else:
    decoder_inputs = lat_input

# Decoder outputs

if model == 'gaussian': 
    decoder_outputs = Dense(input_dim, activation='sigmoid')(x)

elif model == 'nb' or model == 'zinb':

    # Must ensure all values positive since loss takes logs etc.
    MeanAct = lambda a: tf.clip_by_value(K.exp(a), 1e-5, 1e6)
    DispAct = lambda a: tf.clip_by_value(tf.nn.softplus(a), 1e-4, 1e4)

    mu = Dense(input_dim, activation = MeanAct, name='mu')(x)
    disp = Dense(input_dim, activation = DispAct, name='disp')(x)

    # Decoder outputs
    if use_sf:
        sfAct = Lambda(lambda a: K.exp(a), name = 'expzsf') 
        sf = sfAct(sf) if learn_sf else sfAct(sf_input)
        #sf = sfAct(sf)
        mu_sf = multiply([mu, sf]) # Uses broadcasting
        
        decoder_outputs = [mu_sf, disp]
    else:
        decoder_outputs = [mu, disp]
    
    if model == 'zinb':
        # Activation is sigmoid because values restricted to [0,1]
        pi = Dense(input_dim, activation = 'sigmoid', name='pi')(x)
        decoder_outputs.append(pi)    


decoder = Model(decoder_inputs, decoder_outputs, name='decoder')

plot_model(decoder, to_file=models_dir + '/' + model + '_decoder.png',
           show_shapes=True, show_layer_names=True)         


# =============================================================================
# Autoencoder Model
# =============================================================================
# Connect encoder and decoder models 
if use_sf:
    
    if learn_sf:    
        
        AE_inputs = count_input
        
        if vae:
            AE_outputs = decoder([encoder(count_input)[2], count_input])
        else:
            AE_outputs = decoder([encoder(count_input), count_input])
   
    else:
        
        AE_inputs = [count_input, sf_input]
        #AE_inputs = [count_input, sf]
        
        if vae:
            AE_outputs = decoder([encoder(count_input)[2], sf_input])
            #AE_outputs = decoder([encoder(count_input)[2], sf])
        else:
            AE_outputs = decoder([encoder(count_input), sf_input])
            #AE_outputs = decoder([encoder(count_input), sf])
        
else:
    
    AE_inputs = count_input
    
    if vae:
        AE_outputs = decoder(encoder(count_input)[2])
    else:
        AE_outputs = decoder(encoder(count_input))

autoencoder = Model(AE_inputs, AE_outputs, name='autoencoder')

print (autoencoder.summary())
plot_model(autoencoder, to_file=models_dir + '/' + model + '_autoencoder.png',
           show_shapes=True, show_layer_names=True)         

# =============================================================================
# Define custom loss
# =============================================================================

def NB_loglikelihood(mu, r, y, eps=1e-10):
    
    if tf2_flag:
        l1 = tf.math.lgamma(y+r+eps) - tf.math.lgamma(r+eps) - tf.math.lgamma(y+1.0)
        l2 = y * tf.math.log((mu+eps)/(r+mu+eps)) + r * tf.math.log((r+eps)/(r+mu+eps))
    else:
        l1 = tf.lgamma(y+r+eps) - tf.lgamma(r+eps) - tf.lgamma(y+1.0)
        l2 = y * tf.log((mu+eps)/(r+mu+eps)) + r * tf.log((r+eps)/(r+mu+eps))
    
    log_likelihood = l1 + l2
    
    return log_likelihood


def ZINB_loglikelihood(mu, r, pi, y, eps=1e-10):
    
    nb_log_likelihood = NB_loglikelihood(mu, r, y, eps)
    
    if tf2_flag:
        case_zero = tf.math.log(eps + pi + (1.0 - pi) * tf.math.pow((r/(r+mu+eps)), r))
        case_nonzero = tf.math.log(1.0 - pi + eps) + nb_log_likelihood
    else:
        case_zero = tf.log(pi + (1.0-pi) * tf.pow((r/(r+mu)), r))
        case_nonzero = tf.log(1.0-pi) + nb_log_likelihood
    
    # If count value < 1e-8, use case_zero for the log-likelihood
    zinb_log_likelihood = tf.where(tf.less(y, 1e-8), case_zero, case_nonzero)
    
    return zinb_log_likelihood

# KL divergence between 2 Gaussians, one of which is N(0,1)
# def gaussian_kl_z(mean, log_var):
#     kl = - 0.5 * (1 + log_var - K.square(mean) - K.exp(log_var))
#     #return K.sum(kl, axis=-1)
#     return kl

# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/kullback_leibler.py
# KL divergence between 2 Gaussians, g1 and g2
# note: g1[1] (and g2[1]) are the stdev, not log variance
# def gaussian_kl(g1, g2):
    
#     if tf2_flag:
#         import tensorflow_probability as tfp
#         ds = tfp.distributions
#     else:
#         ds = tf.contrib.distributions
#     g1 = ds.Normal(loc=g1[0], scale=g1[1])
#     g2 = ds.Normal(loc=g2[0], scale=g2[1])
#     kl = ds.kl_divergence(g1, g2)
    
#     #return K.sum(kl, axis=-1)
#     return kl

# KL divergence between 2 Gaussians
# g1[0] = mean, g1[1] = log_var
def gaussian_kl(g1, g2):
    kl = - 0.5 * (1 - g2[1] + g1[1]) + 0.5 * K.exp(- g2[1]) * ( K.exp(g1[1]) + K.square(g1[0] - g2[0]) )
    #return K.sum(kl, axis=-1)
    return kl

# ones = np.ones((10,))
# zeros = np.zeros((10,))
# a = np.array([0,1,2,3,4,5,6,7,8,9], dtype='float64')
# print ('Here')
# K.print_tensor(gaussian_kl_z(a,zeros))

# K.print_tensor(gaussian_kl_1([a,ones],[zeros, np.sqrt(np.exp(a))]))
# K.print_tensor(gaussian_kl([a,zeros],[zeros,a]))

'''
def VAE_loss(outputs):
    
    def loss (y_true, y_pred):
        
        eps = 1e-10 # Prevent NaN loss value
        mu = outputs[0]
        r = outputs[1]
        y = y_true
        
        # Reconstruction loss for NB/ZINB distribution
        if model=='nb':
            to_sum = - NB_loglikelihood(mu, r, y, eps)
        
        elif model=='zinb':
            pi = outputs[2]
            to_sum = - ZINB_loglikelihood(mu, r, pi, y, eps)
        
        total_loss = K.sum(to_sum, axis=-1)
        
        # KL loss for gene expressions
        if vae:
            kl_loss = beta_vae * gaussian_kl_z(z_mean, z_log_var)
            total_loss += kl_loss
        
        # KL loss for size factors
        if use_sf and learn_sf:
            log_counts = np.log(adata.obs['n_counts'])
            
            # ones_shape = batch_size
            ones_shape = K.shape(y_pred)[0]
            
            ones = tf.ones((ones_shape, 1))
            
            m = np.mean(log_counts) * ones
            v = np.var(log_counts) * ones
            
            sf_kl_loss = beta_vae * gaussian_kl([sf_mean, sf_log_var], [m, v])
            total_loss += sf_kl_loss
        
        return total_loss
    
    return loss
'''

# using add_loss method

'''
def VAE_loss(outputs):
    def loss(y_true, y_pred):
        return K.sum(y_true, axis=-1)
    return loss

VAE_loss = tf.keras.losses.mean_squared_error(AE_inputs, AE_outputs[0])
autoencoder.add_loss(VAE_loss)
'''

def VAE_loss(y_true, outputs):
    
    eps = 1e-10 # Prevent NaN loss value
    mu = outputs[0]
    r = outputs[1]
    #y = y_true[0] if use_sf and not learn_sf else y_true
    
    if use_sf and not learn_sf:
        y = y_true[0]
    else:
        y = y_true
        
    # Reconstruction loss for NB/ZINB distribution
    if model=='nb':
        to_sum = - NB_loglikelihood(mu, r, y, eps)
    
    elif model=='zinb':
        pi = outputs[2]
        to_sum = - ZINB_loglikelihood(mu, r, pi, y, eps)
        
    total_loss = K.sum(to_sum, axis=-1)
        
    # KL loss for latent gene expressions
    if vae:
        #kl_loss = beta_vae * gaussian_kl_z(z_mean, z_log_var)
        zeros = tf.zeros(K.shape(z_mean))
        kl_loss = beta_vae * gaussian_kl([z_mean, z_log_var], [zeros, zeros])
        total_loss += kl_loss
        
        # KL loss for size factors
        if use_sf and learn_sf:
            # simply use all the data, rather than the data corresponding to the batch
            log_counts = np.log(adata.obs['n_counts'])
            
            # ones_shape = batch_size
            ones_shape = K.shape(outputs[0])[0]
            #ones_shape = K.shape(sf_mean)[0]
            
            ones = tf.ones((ones_shape, 1))
            
            m = np.mean(log_counts) * ones
            v = np.var(log_counts) * ones
            
            sf_kl_loss = beta_vae * gaussian_kl([sf_mean, sf_log_var], [m, v])
            total_loss += sf_kl_loss
        
    return total_loss


# Loss function run thrice (once for each output) but only one used
if model == 'zinb':
    loss_weights=[1., 0.0, 0.0]
elif model == 'nb':
    loss_weights=[1., 0.0]

opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)

if model == 'gaussian':
    autoencoder.compile(optimizer=opt, loss='mse')
else:
    # autoencoder.compile(optimizer=opt,
    #                     loss=VAE_loss(AE_outputs),
    #                     loss_weights=loss_weights)

    autoencoder.add_loss(VAE_loss(AE_inputs, AE_outputs))

    autoencoder.compile(optimizer=opt,
                        loss=None,
                        loss_weights=loss_weights)


# from tb_callback import MyTensorBoard
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

# =============================================================================
# Train model
# =============================================================================

if use_sf and not learn_sf:
    fit_x = [X_train, sf_train]
    val_x = [X_test, X_test]
else:
    fit_x = X_train
    val_x = X_test
    
if model == 'gaussian':
    fit_y = X_train
    val_y = X_test
elif model == 'nb':
    fit_y = [X_train, X_train]
    val_y = [X_test, X_test]
elif model == 'zinb':
    fit_y = [X_train, X_train, X_train]
    val_y = [X_test, X_test, X_test]

# Pass adata.obs['sf'] as an input. 2nd, 3rd elements of y not used
loss = autoencoder.fit(fit_x, fit_y, epochs=epochs, batch_size=batch_size,
                       shuffle=False, callbacks=[tensorboard],
                       validation_data=(val_x, val_y))

autoencoder.save('AE.h5')


# =============================================================================
# Plot loss
# =============================================================================

plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])

# =============================================================================
# Test model
# =============================================================================

if vae:
    encoded_data = encoder.predict(adata.X)[0]
else:
    encoded_data = encoder.predict(adata.X)

if use_sf:
    if learn_sf:
        decoded_data = decoder.predict([encoded_data, adata.X])
    else:
        decoded_data = decoder.predict([encoded_data, adata.obs['sf'].values])
else:
    decoded_data = decoder.predict(encoded_data)

adata.X = decoded_data[0]
adata.X = gene_scaler.inverse_transform(decoded_data[0])
save_h5ad(adata, 'denoised')


def test_sf():
    
    if learn_sf:
        sf = sf_encoder.predict(adata.X)
    else:
        sf = sf_encoder.predict(adata.obs['sf'].values)
    
    return sf


def test_AE():
    
    if vae:
        encoded_data = encoder.predict(X_train[0:batch_size])[0]
    else:
        encoded_data = encoder.predict(X_train[0:batch_size])
    
    if use_sf:
        if learn_sf:
            decoded_data = decoder.predict([encoded_data, X_train[0:batch_size]])
        else:            
            decoded_data = decoder.predict([encoded_data, adata.obs['sf'].values[0:batch_size]])
    else:
        decoded_data = decoder.predict(encoded_data)
    
    return decoded_data
