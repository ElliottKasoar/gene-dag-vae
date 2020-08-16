#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from load import save_h5ad, load_h5ad

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

from keras.utils import plot_model
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU


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

# Encoder Model
input    = Input(shape=input_shape)
encoded    = Dense(1024, activation='relu')(input)
encoded    = Dense(512, activation='relu')(encoded)
latent    = Dense(encoding_dim, activation='relu')(encoded)

encoder = Model(input, latent, name='encoder')

# Encoded representation of the input (with sparsity contraint via regularizer)
# encoded = Dense(encoding_dim, activation='relu', 
#                 activity_regularizer=regularizers.l1(10e-5))(VAE_input)

# Decoder Model 
# Lossy reconstruction of the input
lat_input = Input(shape=(encoding_dim,))
decoded = Dense(512, activation='relu')(lat_input)
decoded    = Dense(128, activation='relu')(decoded)
output    = Dense(input_dim, activation='sigmoid')(decoded)

decoder = Model(lat_input, output, name='decoder')

# Autoencoder Model
output = decoder(encoder(input))
autoencoder = Model(input, output, name='autoencoder')


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print (autoencoder.summary())
plot_model(autoencoder)         

# =============================================================================
# Train model
# =============================================================================

autoencoder.fit(X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True)

autoencoder.save('AE.h5')

# =============================================================================
# Test model
# =============================================================================

encoded_data = encoder.predict(X_test)
decoded_data = decoder.predict(encoded_data)


