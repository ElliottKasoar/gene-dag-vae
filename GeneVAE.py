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
from keras.layers import Layer, Input, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import multiply, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import scanpy as sc
from time import time

# (Almost reproducible)
# np.random.seed(1337)
# tf.random.set_seed(1234)

if int(tf.__version__.startswith("2.")):
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
# Parameters
# =============================================================================

# Set default parameters. Also defines possible/necessary parameters reading in 
def default_params():
    
    params = {
        
        # Encoder (and symmetric decoder) model structure:
        'AE_params' : {
            'latent_dim' : 32, # Size of encoded representation
            'gene_layers' : 4, # Hidden layers between input and latent layers
            'gene_nodes' : 512, # Size of initial hidden layer
            'gene_flat' : False, # Keep all hidden layers flat (else halve each layer)
            'gene_alpha' : 0.2, # LeakyReLU alpha
            'gene_momentum' : 0.8, # BatchNorm momentum
            'gene_dropout' : 0.2 # Dropout rate
        },
        
        # Size factor model structure:
        'sf_params' : {
            'sf_layers' : 3, # Hidden layers between input and latent layers
            'sf_nodes' : 512, # Size of initial hidden layer (half each layer)
            'sf_alpha' : 0.2, # LeakyReLU alpha
            'sf_momentum' : 0.8, # BatchNorm momentum
            'sf_dropout' : 0.2 # Dropout rate
        },
        
        # Adam optimiser parameters:
        'opt_params' : {
            'lr' : 0.001, # Learning rate (Adam default : 0.001)
            'beta_1' : 0.9, # (Adam default : 0.9)
            'beta_2' : 0.999 # (Adam default : 0.999)
        },
        
        # Model architecture options:
        'arch_params' : {
            'use_sf' : True, # Use size factor in network
            'learn_sf' : True, # Learn size factor using (V)AE network, else input values
            'model' : 'zinb', # Use zero-inflated negative binomial dist
            # 'model' : 'nb', # Use negative binomial dist
            # 'model' : 'gaussian', # Use gaussian dist
            'vae' : True, # Make autoencoder variational
            'beta_vae' : 1 # Change constraint on latent capacity
        },
        
        'training_params' : {
            'train_size' : 0.9, # Fraction of data used in training
            'epochs' : 5,
            'batch_size' : 512
        },
        
        'debugging_params' : {
            'debug' : False
            }
    }
    
    return params


# =============================================================================
# Get and validate parameters
# =============================================================================

# Get parameters from defaults/input and validate
def get_params(params=None):
    
    if params is None:
        params = default_params()
        
    else:
        
        for param_type, value in params.items ():
            assert param_type in default_params(), param_type + ' is not a valid group of keys'
        
        for param_type, value in default_params().items ():
            
            assert param_type in params, 'param should have the ' + param_type + ' group of keys'
            
            for key in params[param_type]:
                assert key in default_params()[param_type], key + ' is not a valid key'
            
            for key in default_params()[param_type]:
                assert key in params[param_type], 'param should have ' + key + ' key'
    
    
    int_keys = ['latent_dim', 'gene_layers', 'gene_nodes', 'sf_layers',
            'sf_nodes', 'epochs', 'batch_size']
    
    for key in int_keys:
        for param_type in params:
            
            if key in params[param_type]:
                val = params[param_type][key]
                
                if not isinstance(val, int):
                    raise TypeError(f' {key} must be an integer.  Current value: {val}')
                    
                elif val < 1:
                    raise ValueError(f' {key} must be greater than 0. Current value: {val}')
    
    
    float_keys = ['gene_alpha', 'gene_momentum', 'gene_dropout', 'sf_alpha',
            'sf_momentum', 'sf_dropout', 'lr', 'beta_1', 'beta_2', 'beta_vae',
            'train_size']
    
    for key in float_keys:
        for param_type in params:
            
            if key in params[param_type]:
                val = params[param_type][key]
                
                if not isinstance(val, float) and not isinstance(val, int):
                    raise TypeError(f' {key} must be a number.  Current value: {val}')
                    
                elif val < 0:
                    raise ValueError(f' {key} must be greater than 0. Current value: {val}')    
    
    
    bool_keys = ['gene_flat', 'use_sf', 'learn_sf', 'vae']
    
    for key in bool_keys:
        for param_type in params:
            
            if key in params[param_type]:    
                val = params[param_type][key]
                
                if not isinstance(val, bool):
                    raise TypeError(f' {key} must be a boolean.  Current value: {val}')
    
    
    assert params['arch_params']['model'] in ['nb', 'zinb', 'gaussian'], "Model must be 'nb', 'zinb' or 'Guassian'"
    
    if not params['arch_params']['use_sf']:
        assert not params['arch_params']['learn_sf'], "'learn_sf' must be False if 'use_sf' is False"
    
    return params


# =============================================================================
# Load data
# =============================================================================

# Loads data from preprocessed file
# Performs limited further processing - scale and split into train/test data
def load_data(train_size):
    
    adata = load_h5ad('preprocessed') # Need to add code to ensure this exists	
    
    # Input shape
    input_dim = adata.X.shape[1]
    
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
    
    return adata, X_train, X_test, sf_train, sf_test, input_dim, gene_scaler


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
    
        
    # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/kullback_leibler.py
    # KL divergence between 2 Gaussians, g1 and g2
    # note: g1[1] (and g2[1]) are the stdev, not log variance
    # def gaussian_kl(self, g1, g2):
        
    #     if tf2_flag:
    #         import tensorflow_probability as tfp
    #         ds = tfp.distributions
    #     else:
    #         ds = tf.contrib.distributions
    #     g1 = ds.Normal(loc=g1[0], scale=g1[1])
    #     g2 = ds.Normal(loc=g2[0], scale=g2[1])
    #     kl = ds.kl_divergence(g1, g2)
        
    #     #return K.mean(kl, axis=-1)
    #     return kl
    
    
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




# =============================================================================
# Encoder Model: count data
# =============================================================================

# def build_encoder(input_dim, arch_params, AE_params):
def build_encoder(count_input, arch_params, AE_params):
    
    x = Dense(AE_params['gene_nodes'])(count_input)
    x = BatchNormalization(momentum=AE_params['gene_momentum'])(x)
    x = LeakyReLU(AE_params['gene_alpha'])(x)
    x = Dropout(AE_params['gene_dropout'])(x)
    
    for i in range(1, AE_params['gene_layers']):
        
        if AE_params['gene_flat']:
            nodes = AE_params['gene_nodes']
        else:
            nodes = AE_params['gene_nodes'] // (2**i)
            if nodes < AE_params['latent_dim']:
                print("Warning: layer has fewer nodes than latent layer")
                print(f"Layer nodes: {nodes}. Latent nodes: {AE_params['latent_dim']}")
        
        x = Dense(nodes)(x)
        x = BatchNormalization(momentum=AE_params['gene_momentum'])(x)
        x = LeakyReLU(AE_params['gene_alpha'])(x)
        x = Dropout(AE_params['gene_dropout'])(x)
    
    if arch_params['vae']:
        z_mean = Dense(AE_params['latent_dim'], name='latent_mean')(x)
        z_log_var = Dense(AE_params['latent_dim'], name='latent_log_var')(x)
        
        z = SampleLayer(AE_params['latent_dim'])([z_mean, z_log_var])
        encoder = Model(count_input, [z_mean, z_log_var, z], name='encoder')
    
    else:
        z = Dense(AE_params['latent_dim'], activation='relu', name='latent')(x)
        encoder = Model(count_input, z, name='encoder')
    
    plot_model(encoder,
               to_file=models_dir + '/' + arch_params['model'] + '_encoder.png',
               show_shapes=True, show_layer_names=True)
    
    return encoder


# =============================================================================
# Size factor model
# =============================================================================

def build_sf_model(count_input, adata, arch_params, sf_params):

    if arch_params['learn_sf']:
        
        x = Dense(sf_params['sf_nodes'])(count_input)
        x = BatchNormalization(momentum=sf_params['sf_momentum'])(x)
        x = LeakyReLU(sf_params['sf_alpha'])(x)
        x = Dropout(sf_params['sf_dropout'])(x)
        
        for i in range(1, sf_params['sf_layers']):
            nodes = sf_params['sf_nodes'] // (2**i)
            x = Dense(nodes)(x)
            x = BatchNormalization(momentum=sf_params['sf_momentum'])(x)
            x = LeakyReLU(sf_params['sf_alpha'])(x)
            x = Dropout(sf_params['sf_dropout'])(x)
        
        if arch_params['vae']:
            sf_mean = Dense(1, name='sf_mean')(x)
            sf_log_var = Dense(1, name='sf_log_var')(x)
            sf = SampleLayer(1)([sf_mean, sf_log_var])
            sf_encoder = Model(count_input, [sf_mean, sf_log_var, sf], name='sf_encoder')
            
        else:
            sf = Dense(1, name='sf_latent')(x)
            sf_encoder = Model(count_input, sf, name='sf_encoder')
        
        plot_model(sf_encoder, to_file=models_dir + '/' + arch_params['model'] + '_sf_encoder.png',
               show_shapes=True, show_layer_names=True)
        
    else:
        sf_encoder = Input(shape=(1,), name='size_factor_input')
    
    
    return sf_encoder


# =============================================================================
# Decoder Model 
# =============================================================================

def build_decoder(input_dim, count_input, AE_params, arch_params):

    # Lossy reconstruction of the input
    lat_input = Input(shape=(AE_params['latent_dim'],))
    
    if AE_params['gene_flat']:
        x = Dense(AE_params['gene_nodes'])(lat_input)
    else:
        nodes = AE_params['gene_nodes'] // (2 ** (AE_params['gene_layers'] - 1))
        x = Dense(nodes)(lat_input)
    
    x = BatchNormalization(momentum=AE_params['gene_momentum'])(x)
    x = LeakyReLU(AE_params['gene_alpha'])(x)
    x = Dropout(AE_params['gene_dropout'])(x)
    
    for i in range(1, AE_params['gene_layers']):
        if AE_params['gene_flat']:
            nodes = AE_params['gene_nodes']
        else:
            nodes = AE_params['gene_nodes'] // (2 ** (AE_params['gene_layers'] - (i+1)))
    
        x = Dense(nodes)(x)
        x = BatchNormalization(momentum=AE_params['gene_momentum'])(x)
        x = LeakyReLU(AE_params['gene_alpha'])(x)
        x = Dropout(AE_params['gene_dropout'])(x)
    
    # Decoder inputs
    decoder_inputs = lat_input
    
    # Decoder outputs
    if arch_params['model'] == 'gaussian': 
        decoder_outputs = Dense(input_dim, activation='sigmoid', name='mu')(x)
    
    elif arch_params['model'] == 'nb' or arch_params['model'] == 'zinb':
    
        # Must ensure all values positive since loss takes logs etc.
        MeanAct = lambda a: tf.clip_by_value(K.exp(a), 1e-5, 1e6)
        DispAct = lambda a: tf.clip_by_value(tf.nn.softplus(a), 1e-4, 1e4)
    
        mu = Dense(input_dim, activation = MeanAct, name='mu')(x)
        disp = Dense(input_dim, activation = DispAct, name='disp')(x)
    
        decoder_outputs = [mu, disp]
        
        if arch_params['model'] == 'zinb':
            # Activation is sigmoid because values restricted to [0,1]
            pi = Dense(input_dim, activation = 'sigmoid', name='pi')(x)
            decoder_outputs.append(pi)    
    
    decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
    
    plot_model(decoder,
               to_file=models_dir + '/' + arch_params['model'] + '_decoder.png',
               show_shapes=True, show_layer_names=True)   

    return decoder      


# =============================================================================
# Autoencoder Model
# =============================================================================

# Connect encoder and decoder models 
def build_autoencoder(count_input, adata, encoder, decoder, sf_encoder, arch_params,
                      opt_params):
    
    # KL Loss (count data)
    if arch_params['vae']:
        z_mean, z_log_var, z = encoder(count_input)
        z = KLDivergenceLayer(arch_params['beta_vae'], 0., 0.)([z_mean, z_log_var, z])
    
    else:
        z = encoder(count_input)
    
    AE_inputs = count_input
    AE_outputs = decoder(z)
    
    if arch_params['use_sf']:
        
        if arch_params['learn_sf']:    
            
            if arch_params['vae']:
                                
                # KL Loss (size factor data)
                sf_mean, sf_log_var, sf = sf_encoder(count_input)
        
                log_counts = np.log(adata.obs['n_counts'])
                
                m = np.float32(np.mean(log_counts))
                v = np.float32(np.var(log_counts))
                
                sf = KLDivergenceLayer(arch_params['beta_vae'], m, v)([sf_mean, sf_log_var, sf])
            else:
                sf = sf_encoder(count_input)

        else:
            sf = sf_encoder
            AE_inputs = [AE_inputs]
            AE_inputs.append(sf)
      
        sfAct = Lambda(lambda a: K.exp(a), name = 'expzsf') 
        sf = sfAct(sf)
        
        if arch_params['model'] == 'gaussian':
            AE_outputs = multiply([AE_outputs, sf]) # Uses broadcasting
        else:
            AE_outputs[0] = multiply([AE_outputs[0], sf]) # Uses broadcasting
            
    else:
        pass
    
    
    if arch_params['model'] == 'gaussian':
        AE_outputs = ReconstructionLossLayer(MeanSquaredError)(count_input, AE_outputs)
    elif arch_params['model'] == 'nb':
        AE_outputs = ReconstructionLossLayer(NB_loglikelihood)(count_input, AE_outputs)
    elif arch_params['model'] == 'zinb':
        AE_outputs = ReconstructionLossLayer(ZINB_loglikelihood)(count_input, AE_outputs)
    
    autoencoder = Model(AE_inputs, AE_outputs, name='autoencoder')
    
    print (f'# losses = {len(autoencoder.losses)}: \n {autoencoder.losses} \n')
    plot_model(autoencoder, to_file=models_dir + '/' + arch_params['model'] + '_autoencoder.png',
               show_shapes=True, show_layer_names=True)         
    
    # =============================================================================
    
    opt = Adam(lr=opt_params['lr'],
               beta_1=opt_params['beta_1'],
               beta_2=opt_params['beta_2'])
    
    autoencoder.compile(optimizer=opt, loss=None)

    return autoencoder


# =============================================================================
# Create Models
# =============================================================================

def create_models(input_dim, adata, params):
    input_shape = (input_dim,)
    count_input = Input(shape=input_shape, name='count_input')
    
    encoder = build_encoder(count_input, params['arch_params'],
                                         params['AE_params'])
    
    if params['arch_params']['use_sf']:
        sf_encoder = build_sf_model(count_input, adata,
                                        params['arch_params'],
                                        params['sf_params'])
    else:
        sf_encoder = None
            
    decoder = build_decoder(input_dim, count_input, params['AE_params'],
                            params['arch_params'])
    
    autoencoder = build_autoencoder(count_input, adata, encoder, decoder, sf_encoder,
                                    params['arch_params'],
                                    params['opt_params'])
    
    return encoder, sf_encoder, decoder, autoencoder


# =============================================================================
# Train model
# =============================================================================

def train_model(X_train, X_test, sf_train, sf_test, autoencoder, arch_params,
                training_params):

    # from tb_callback import MyTensorBoard
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    
    if arch_params['use_sf'] and not arch_params['learn_sf']:
        fit_x = [X_train, sf_train]
        val_x = [X_test, sf_test]
    else:
        fit_x = X_train
        val_x = X_test
        
    if arch_params['model'] == 'gaussian':
        fit_y = X_train
        val_y = X_test
    elif arch_params['model'] == 'nb':
        fit_y = [X_train, X_train]
        val_y = [X_test, X_test]
    elif arch_params['model'] == 'zinb':
        fit_y = [X_train, X_train, X_train]
        val_y = [X_test, X_test, X_test]
    
    # Pass adata.obs['sf'] as an input. 2nd, 3rd elements of y not used
    loss = autoencoder.fit(fit_x, fit_y, epochs=training_params['epochs'],
                           batch_size=training_params['batch_size'],
                           shuffle=False, callbacks=[tensorboard],
                           validation_data=(val_x, val_y))
    
    autoencoder.save('AE.h5')
    
    return loss


# =============================================================================
# Plot loss
# =============================================================================

def plot_loss(loss):
    
    plt.plot(loss.history['loss'])
    plt.plot(loss.history['val_loss'])


# =============================================================================
# Test model
# =============================================================================

def test_model(adata, gene_scaler, encoder, decoder, sf_encoder, arch_params):
    
    if arch_params['vae']:
        encoded_data = encoder.predict(adata.X)[2]
    else:
        encoded_data = encoder.predict(adata.X)
    
    if arch_params['model'] == 'gaussian':
        decoded_data = decoder.predict(encoded_data)
    else:
        decoded_data = decoder.predict(encoded_data)[0]
    
    if arch_params['use_sf']:
        sf_data = test_sf(adata, sf_encoder, arch_params)
        decoded_data = multiply([decoded_data, sf_data])
    
    adata.X = gene_scaler.inverse_transform(decoded_data)
    save_h5ad(adata, 'denoised')
    

def test_sf(adata, sf_encoder, arch_params):
    
    if arch_params['learn_sf']:
        if arch_params['vae']:
            sf_data = sf_encoder.predict(adata.X)[2]
        else:
            sf_data = sf_encoder.predict(adata.X)
    else:
        sf_data = adata.obs['sf'].values
    
    return sf_data


def test_AE(adata, X_train, encoder, decoder, sf_encoder, arch_params, training_params):
    
    if arch_params['vae']:
        encoded_data = encoder.predict(X_train[0:training_params['batch_size']])[2]
    else:
        encoded_data = encoder.predict(X_train[0:training_params['batch_size']])
    
    if arch_params['model'] == 'gaussian':
        decoded_data = decoder.predict(encoded_data)
    else:
        decoded_data = decoder.predict(encoded_data)[0]
        
    if arch_params['use_sf']:
        sf_data = test_sf(adata, sf_encoder, arch_params)[0:training_params['batch_size']]
        decoded_data = multiply([decoded_data, sf_data])
    
    return decoded_data


# =============================================================================
# Main
# =============================================================================

def main():
    
    params = get_params()
    
    if params['debugging_params']['debug']:
        from tensorflow.python import debug as tf_debug
        sess = K.get_session()
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline") #Spyder
        K.set_session(sess)
    
    adata, X_train, X_test, sf_train, sf_test, input_dim, gene_scaler = load_data(params['training_params']['train_size'])
    
    encoder, sf_encoder, decoder, autoencoder = create_models(input_dim, adata, params)
    
    loss = train_model(X_train, X_test, sf_train, sf_test, autoencoder,
                params['arch_params'], params['training_params'])
    
    plot_loss(loss)
    
    test_model(adata, gene_scaler, encoder, decoder, sf_encoder, params['arch_params'])
    
    if params['arch_params']['use_sf']:
        test_sf(adata, sf_encoder, params['arch_params'])
    
    test_AE(adata, X_train, encoder, decoder, sf_encoder, params['arch_params'],
            params['training_params'])
    
    
if __name__ == '__main__':
    main()