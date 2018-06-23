import os,random
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.backend import squeeze
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
from data_loader import *
import tensorflow as tf
from keras.utils import multi_gpu_model
from custom_layers import GlobalAveragePooling1D, get_data
from keras.layers import Input



def inception(input_img, fs=[64,64,64,64,64], with_residual=False):
    tower_1 = Conv1D(fs[0], 1, padding='same', activation='relu')(input_img)
    tower_1 = Conv1D(fs[1], 3, padding='same', activation='relu')(tower_1)
    tower_2 = Conv1D(fs[2], 1, padding='same', activation='relu')(input_img)
    tower_2 = Conv1D(fs[3], 5, padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling1D(3, strides=1, padding='same')(input_img)
    tower_3 = Conv1D(fs[4], 1, padding='same', activation='relu')(tower_3)
    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 2)
    if with_residual and output.shape==input_img.shape:
        output = output+input_img
    return output
def googleNet(x, data_format='channels_last'):
    x = Conv1D(64, 7, strides=2, data_format=data_format, padding='same', activation='relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    for dep in range(2):
        x = Conv1D(192, 3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    for dep in range(2):
        x = inception(x, fs=[32,64,16,32,48])
    for dep in range(2):
        x = inception(x, fs=[32,64,32,64,64])
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    for dep in range(10):
        x = inception(x, fs=[48,96,48,96,96], with_residual=True)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    for dep in range(4):
        x = inception(x, fs=[32,64,32,64,64])
    x = GlobalAveragePooling1D()(x, keepdims=True)
    x = Dropout(0.4)(x)
    output = Flatten()(x)
    out    = Dense(24, activation='softmax')(output)
    return out


if __name__ == "__main__":
    input_img = Input(shape=(1024,2))
    out = googleNet(input_img,data_format='channels_last')
    model = Model(inputs=input_img, outputs=out)
    model.summary()
    
    x_train, y_train, x_val, y_val = get_data()
    filepath = './basemodels/inception.wts.h5'
    model = multi_gpu_model(model, gpus=2)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    train_batch_size, number_of_epochs = 512, 10
    history = model.fit(x_train, y_train,
              batch_size=train_batch_size,
              epochs=20,
              verbose=2,
              validation_data=(x_val, y_val),
              callbacks = [
                  keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
              ])
    score, acc = model.evaluate(x_val, y_val, verbose=0)
    print('Test accuracy:', acc)