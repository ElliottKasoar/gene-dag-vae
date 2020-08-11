#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from load import save_h5ad, load_h5ad
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

# =============================================================================
# Model parameters
# =============================================================================

# Size of encoded representation
encoding_dim = 256

train_size = 0.7

# =============================================================================
# Load data
# =============================================================================

adata = load_h5ad('preprocessed')

x = adata.X

# Input shape
input_dim = x.shape[1]
input_shape = (input_dim,)

qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
x = qt.fit_transform(x)
scale = x.max(axis=0)
x = np.divide(x, scale)
x_train, x_test = train_test_split(x, train_size=train_size)

# =============================================================================
# Build models
# =============================================================================

# Input placeholder
AE_input = Input(shape=input_shape)

# Encoded representation of the input (with sparsity contraint via regularizer)
# encoded = Dense(encoding_dim, activation='relu', 
#                 activity_regularizer=regularizers.l1(10e-5))(VAE_input)

encoded = Dense(1024, activation='relu')(AE_input)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Lossy reconstruction of the input
decoded = Dense(512, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='tanh')(decoded)

# Model maps an input to its reconstruction
autoencoder = Model(AE_input, decoded)

# Model maps an input to its encoded representation
encoder_model = Model(AE_input, encoded)

# Placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))

# Retrieve the decoder layers from the autoencoder model
decoder = autoencoder.layers[-3](encoded_input)
decoder = autoencoder.layers[-2](decoder)
decoder = autoencoder.layers[-1](decoder)

# Create decoder model
decoder_model = Model(encoded_input, decoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# =============================================================================
# Train model
# =============================================================================

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True)

autoencoder.save('AE.h5')

# =============================================================================
# Test model
# =============================================================================

encoded_data = encoder_model.predict(x_test)
decoded_data = decoder_model.predict(encoded_data)


