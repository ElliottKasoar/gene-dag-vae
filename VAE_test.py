#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:37:58 2020

@author: Elliott
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Model parameters
# =============================================================================

# Size of encoded representation
encoding_dim = 32

# Input shape
input_shape = (784,)

# =============================================================================
# Load data
# =============================================================================

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print (x_test.shape)

# =============================================================================
# Build models
# =============================================================================

# Input placeholder
VAE_input = Input(shape=input_shape)

# Encoded representation of the input (with sparsity contraint via regularizer)
# encoded = Dense(encoding_dim, activation='relu',
#                 activity_regularizer=regularizers.l1(10e-5))(VAE_input)

encoded = Dense(encoding_dim, activation='relu')(VAE_input)

# Lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# Model maps an input to its reconstruction
autoencoder = Model(VAE_input, decoded)

# Model maps an input to its encoded representation
encoder = Model(VAE_input, encoded)

# Placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))

# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# Create decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# =============================================================================
# Train model
# =============================================================================

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# =============================================================================
# Test model
# =============================================================================

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



