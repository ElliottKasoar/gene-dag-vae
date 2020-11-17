#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from load import save_h5ad, load_h5ad
#from loss import NB_loglikelihood
from temp import save_figure, plotTSNE

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

import tensorflow as tf
import keras.backend as K
from keras.utils import plot_model
from keras.layers import Layer, Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import multiply, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import scanpy as sc
from time import time

# create directory 'models' if it doesn't exist
base_dir = '.'
plots_dir = base_dir + '/plots'
models_dir = plots_dir + '/models'

from pathlib import Path
for i in [plots_dir, models_dir]:
    Path(i).mkdir(parents=True, exist_ok=True)

if int(tf.__version__.startswith("2.")):
    tf2_flag = True
else:
    tf2_flag = False


# =============================================================================
# Custom Layers
# =============================================================================

# Calculate likelihood of the (input) data conditioned on a model and its params (likelihood_params)    
class ReconstructionLossLayer(Layer):
    
    '''Identity transform layer that adds
    negative log likelihood (reconstruction loss)
    to the objective'''
    
    def __init__(self, rl_func, eps=1e-10):
        #self.is_placeholder = True
        self.rl = rl_func
        self.eps = eps # Prevent NaN loss values
        super(ReconstructionLossLayer, self).__init__()
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'rl_func': self.rl,
            'eps': self.eps
        })
        return config
    
    def call(self, y, inputs):
        params = inputs
        loss = - K.mean(self.rl(y, params, self.eps), axis=-1)
        self.add_loss(loss)
        return inputs


class KLDivergenceLayer(Layer):
    
    '''Identity transform layer that adds 
    KL divergence to the objective'''
    
    def __init__(self, beta_vae, mean, log_var):
        #self.is_placeholder = True
        #self.kld = kld_func
        self.beta = beta_vae
        self.mean = mean
        self.log_var = log_var
        super(KLDivergenceLayer, self).__init__()
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'beta_vae': self.beta,
            'mean': self.mean,
            'log_var': self.log_var
        })
        return config

    # KL divergence between 2 Gaussians
    def gaussian_kl(self, g1, g2):
        mu_1, logvar_1 = g1
        mu_2, logvar_2 = g2
        
        kl = - 0.5 * (1 - logvar_2 + logvar_1) + 0.5 * K.exp(- logvar_2) * ( K.exp(logvar_1) + K.square(mu_1 - mu_2) )
        return kl
    
    def create_reference(self, inputs):
        ones = tf.ones(K.shape(inputs[0]))
        mean_tensor = tf.multiply(self.mean, ones)
        log_var_tensor = tf.multiply(self.log_var, ones)
        
        return [mean_tensor, log_var_tensor]
    
    def call(self, inputs):
    
        reference = self.create_reference(inputs)
        
        loss = self.beta * K.mean(self.gaussian_kl(inputs[0:2], reference), axis=-1)
        self.add_loss(loss)
        return inputs[2]


class SampleLayer(Layer):
    
    '''Reparametrisation trick'''
    
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(SampleLayer, self).__init__()
        
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim
        })
        return config
    
    def sampling(self, args):
        mean, log_var = args
        epsilon_mean, epsilon_std = [0.0, 1.0]
        
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        epsilon = K.random_normal(shape=(batch, dim),
                                  mean=epsilon_mean, stddev=epsilon_std)
        return mean + K.exp(0.5 * log_var) * epsilon
        
    def call(self, params):
        sample = Lambda(self.sampling, output_shape=(self.output_dim,))(params)
        return sample
    

# Multiplies layer with (I-A^T)
class TransMultA(Layer):
    
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        super(TransMultA, self).__init__()
        
        
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'n_nodes': self.n_nodes
        })
        
        return config
    
    
    def build(self, input_shape):
        
        self.A = self.add_weight(name='adj',
                                 shape=(self.n_nodes, self.n_nodes),
                                 initializer='ones',
                                 dtype='float64',
                                 trainable=True)
    
    def call(self, x):
        
        ident = K.eye(K.shape(self.A)[0], dtype='float64')
        I_A = ident - K.transpose(self.A)    
        output = tf.matmul(I_A, x)
        
        return output


# Multiplies layer with (I-A^T)^-1
class TransMultInvA(Layer):
    
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        super(TransMultA, self).__init__()
        
        
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
            'n_nodes': self.n_nodes
        })
        
        return config
    
    
    def build(self, input_shape):
        
        self.A = self.add_weight(name='adj',
                                 shape=(self.n_nodes, self.n_nodes),
                                 initializer='ones',
                                 dtype='float64',
                                 trainable=True)
    
    def call(self, x):
        
        ident = K.eye(K.shape(self.A)[0], dtype='float64')
        I_A = ident - K.transpose(self.A)
        I_A_inv = tf.linalg.inv(I_A)
        
        output = tf.matmul(I_A_inv, x)
        
        return output


# =============================================================================
# Custom losses
# =============================================================================

# weights that maximise loglikelihood of Gaussian model equivalent to weights that minimise MSE
# mu implicitly learned
# may be too small compared to KL divergences?
def MeanSquaredError(y, mu, eps):
    mse = (y-mu)**2
    return -mse


def NB_loglikelihood(y, params, eps=1e-10):   
    
    mu = params[0]
    r = params[1]
    
    if tf2_flag:
        l1 = tf.math.lgamma(y+r+eps) - tf.math.lgamma(r+eps) - tf.math.lgamma(y+1.0)
        l2 = y * tf.math.log((mu+eps)/(r+mu+eps)) + r * tf.math.log((r+eps)/(r+mu+eps))
    else:
        l1 = tf.lgamma(y+r+eps) - tf.lgamma(r+eps) - tf.lgamma(y+1.0)
        l2 = y * tf.log((mu+eps)/(r+mu+eps)) + r * tf.log((r+eps)/(r+mu+eps))
    
    log_likelihood = l1 + l2
    
    return log_likelihood


def ZINB_loglikelihood(y, params, eps=1e-10):
    
    mu = params[0]
    r = params[1]
    pi = params[2]
    
    nb_log_likelihood = NB_loglikelihood(y, params[:-1], eps)
    
    if tf2_flag:
        case_zero = tf.math.log(eps + pi + (1.0 - pi) * tf.math.pow((r/(r+mu+eps)), r))
        case_nonzero = tf.math.log(1.0 - pi + eps) + nb_log_likelihood
    else:
        case_zero = tf.log(pi + (1.0-pi) * tf.pow((r/(r+mu)), r))
        case_nonzero = tf.log(1.0-pi) + nb_log_likelihood
    
    # If count value < 1e-8, use case_zero for the log-likelihood
    zinb_log_likelihood = tf.where(tf.less(y, 1e-8), case_zero, case_nonzero)
    
    return zinb_log_likelihood


tensor = tf.Variable(np.ones([2,2]))
x = TransMultA(2)(tensor)
print(x)
    
