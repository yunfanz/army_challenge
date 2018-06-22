# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
# %matplotlib inline
import os,random
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.models import load_model, Model
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
from data_loader import *
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.layers import Conv2D, Input, Softmax, LeakyReLU, BatchNormalization, MaxPooling2D

# load data file
data_file = "/datax/yzhang/training_data/training_data_chunk_0.pkl"
data = LoadModRecData(data_file, .9, .05, .05)
classes = data.modTypes

def conv2d_batchnorm_leaky_relu(layer, name, filters = 64, kernel_size = [1,3], padding = 'valid', kernel_initializer = 'glorot_uniform', alpha = 0):
    layer = Conv2D(filters = filters, kernel_size = kernel_size, padding = padding, activation = 'linear', name = name, kernel_initializer = kernel_initializer)(layer)
    layer = BatchNormalization()(layer)
    return LeakyReLU(alpha = alpha)(layer)

dr = 0.5
in_shp = (2, 1024)
input_tensor = Input(shape = in_shp)
y = Reshape(in_shp+(1,), input_shape=in_shp)(input_tensor)

y = ZeroPadding2D((0, 1))(y)
y = conv2d_batchnorm_leaky_relu(y, 'conv1', filters = 64, kernel_size = [1,3], padding = 'valid', kernel_initializer = 'glorot_uniform', alpha = 0)
y = Dropout(dr)(y)

y = ZeroPadding2D((0, 1))(y)
y = conv2d_batchnorm_leaky_relu(y, 'conv2', filters = 64, kernel_size = [1,3], padding = 'valid', kernel_initializer = 'glorot_uniform', alpha = 0)
y = Dropout(dr)(y)

y = ZeroPadding2D((0, 1))(y)
y = conv2d_batchnorm_leaky_relu(y, 'conv3', filters = 64, kernel_size = [1,3], padding = 'valid', kernel_initializer = 'glorot_uniform', alpha = 0)
y = Dropout(dr)(y)

y = ZeroPadding2D((0, 1))(y)
y = conv2d_batchnorm_leaky_relu(y, 'conv4', filters = 64, kernel_size = [2,3], padding = 'valid', kernel_initializer = 'glorot_uniform', alpha = 0)
y = Dropout(dr)(y)

y = MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid', data_format=None)(y)


y = Flatten()(y)
y = Dense(256, activation='relu', init='he_normal', name="dense1")(y)
y = Dropout(dr)(y)
y = Dense( len(classes), init='he_normal', name="dense2" )(y)
y = Activation('softmax')(y)
y = Reshape([len(classes)])(y)

model = Model(inputs=[input_tensor], outputs=[y])
model.summary()
model = multi_gpu_model(model, gpus=2)
model.compile(loss='categorical_crossentropy', optimizer='adam')

number_of_samples_in_instance = data.instance_shape[1]
data.modTypes

filepath = '/tmp/morads/convmodrecnets_CNN2_0.5.wts.h5'
train_batch_size, number_of_epochs = 256, 40
tsteps = data.train_idx.size//train_batch_size
vsteps = data.val_idx.size//train_batch_size
train_batches = data.batch_iter(data.train_idx, train_batch_size, number_of_epochs, use_shuffle=True)
val_batches = data.batch_iter(data.val_idx, train_batch_size, number_of_epochs, use_shuffle=False)
# model.load_weights(filepath)
history = model.fit_generator(train_batches,
    nb_epoch=number_of_epochs,
    steps_per_epoch=tsteps,
    verbose=2,
    validation_data=val_batches,
    validation_steps=vsteps,
    callbacks = [
         keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
         keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
# # we re-load the best weights once training is finished
# model.load_weights(filepath)

model.save('mod_classifier14.h5')
