# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
# %matplotlib inline
import os,random
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.models import load_model
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


# Set the paths of the dataset, the model, output file
data_path = "/datax/yzhang/training_data/training_data_chunk_0.pkl"
model_path = "mod_classifier.h5"
output_path = "TestSet1Predictions.csv"


# load data file and model
data = LoadModRecData(data_path, 0, 0, 1)
model = load_model(model_path)


dataset = data.signalData

#dataset = dataset[0]
#print('example.shape', example.shape)
preds = model.predict(dataset)

# save with 8 decimals
fmt = '%1.0f' + preds.shape[1] * ',%1.8f'
id_col = np.arange(1, dataset.shape[0] + 1)
preds = np.insert(preds, 0, id_col, axis = 1)


header = 'ID,BPSK,QPSK, 8PSK, 16PSK, QAM16, QAM64, 2FSK_5KHz, 2FSK_75KHz, GFSK_75KHz, GFSK_5KHz, GMSK, MSK, CPFSK_75KHz, CPFSK_5KHz, APSK16_c34, APSK32_c34, QAM32, OQPSK, PI4QPSK, FM_NB, FM_WB, AM_DSB, AM_SSB, NOISE\n'
f=open(output_path, 'w')
f.write(header)
f.close()
f=open(output_path,'ab')
np.savetxt(f, preds, delimiter=',', fmt = fmt)
