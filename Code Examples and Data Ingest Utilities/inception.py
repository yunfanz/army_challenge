import os,random
import numpy as np
from keras.utils import np_utils
from keras.models import Model
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
from custom_layers import GlobalAveragePooling1D
from utils import *
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

def googleNet(x, data_format='channels_last', pdict=None):
    if pdict is None:
        pdict = get_pdict(mode='orig')
    m = pdict['depths']
    f = pdict['features']
    x = Conv1D(32*f[0], 7, strides=2, data_format=data_format, padding='same', activation='relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    for dep in range(m[0]):
        x = Conv1D(64*f[1], 3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    for dep in range(m[1]):
        x = inception(x, fs=(np.array([48,64,16,32,32])//2*f[2]))
    for dep in range(m[2]):
        x = inception(x, fs=np.array([64,96,32,48,64])//2*f[3])
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    for dep in range(m[3]):
        x = inception(x, fs=np.array([48,96,16,48,96])//2*f[4], with_residual=False)
    for dep in range(m[4]):
        x = inception(x, fs=np.array([56,112,16,48,80])//2*f[5], with_residual=False)
    for dep in range(m[5]):
        x = inception(x, fs=np.array([64,128,32,64,96])//2*f[6], with_residual=False)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    for dep in range(m[6]):
        x = inception(x, fs=np.array([32,64,32,64,64])//2*f[7])
    x = GlobalAveragePooling1D()(x, keepdims=True)
    x = Dropout(pdict['dr'])(x)
    output = Flatten()(x)
    out    = Dense(24, activation='softmax')(output)
    return out

def get_pdict(mode='orig'):
    pdict = {}
    if mode == 'orig':
        pdict['depths'] = np.ones(7, dtype=np.int)*2
        pdict['features'] =  np.ones(8, dtype=np.int)*2
        pdict['dr'] = 0.6
    elif mode == 'prior':
        pdict['depths'] = np.random.randint(low=0, high=4, size=7)
        pdict['depths'][[0,1,6]] = np.random.randint(low=1, high=3, size=3)
        pdict['features'] =  np.random.randint(low=1, high=4, size=8)
        pdict['dr'] = np.random.uniform(low=0.2, high=0.8, size=1)[0]
    return pdict

def evaluate(model, data_file):
    acc = {}
    snrs = np.arange(-15,15, 5)
    testdata = LoadModRecData(data_file, .0, .0, 1.)
    classes = testdata.modTypes
    for snr in snrs:

        # extract classes @ SNR
        snrThreshold_lower = snr
        snrThreshold_upper = snr+5
        snr_bounded_test_indicies = testdata.get_indicies_withSNRthrehsold(testdata.test_idx, snrThreshold_lower, snrThreshold_upper)
    
        test_X_i = testdata.signalData[snr_bounded_test_indicies]
        test_X_i = np.transpose(test_X_i, (0,2,1))
        test_Y_i = testdata.oneHotLabels[snr_bounded_test_indicies]    

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
        for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        #plt.figure(figsize=(10,10))
        #plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("SNR", snr, "Overall Accuracy: ", cor / (cor+ncor), "Out of", len(snr_bounded_test_indicies))
        acc[snr] = 1.0*cor/(cor+ncor)
        
if __name__ == "__main__":
    input_img = Input(shape=(1024,2))
    out = googleNet(input_img,data_format='channels_last')
    model = Model(inputs=input_img, outputs=out)
    model.summary()
    
    x_train, y_train, x_val, y_val = get_data(mode='time_series',
                                         BASEDIR="../Data/training_data/",
                                         files=[0])
    print(x_train.shape, y_train.shape)
    filepath = './inception.wts.h5'
    #model = multi_gpu_model(model, gpus=2)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    train_batch_size, number_of_epochs = 256, 10
    history = model.fit(x_train, y_train,
              batch_size=train_batch_size,
              epochs=1,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks = [
                  keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
              ])
    evaluate(model, "../Data/training_data/training_data_chunk_14.pkl")
