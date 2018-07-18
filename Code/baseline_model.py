import os,random
import numpy as np
from keras.utils import np_utils
import keras.models as models
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

data_file = "/home/mshefa/training_data/training_data_chunk_0.pkl"
data = LoadModRecData(data_file, .9, .1, 0.)
test_file = "/home/mshefa/training_data/training_data_chunk_14.pkl"
testdata = LoadModRecData(test_file, 0., 0., 1.)
basedir = "./basemodels/"

in_shp = [2,1024]
dr = 0.5 # dropout rate (%)
classes = data.modTypes
N = 2

def get_params(size=10):
    
    f1, f2, f3, stride1, kernel1 = [], [], [], [], []
    for i in range(size):
        f1.append(random.randint(5,9))
        f2.append(random.randint(5, min(f1[-1],8)))
        f3.append(random.randint(5, 8))
        if np.random.random() > 0.5:
            s1 = (1,2)
        else:
            s1 = (1,1)
        if np.random.random() > 0.5:
            k1 = (1,5)
        else:
            k1 = (1,3)
        stride1.append(s1)
        kernel1.append(k1)
    f1 = 2**np.asarray(f1)
    f2 = 2**np.asarray(f2)
    f3 = 2**np.asarray(f3)
    return f1, f2, f3, kernel1, stride1

def build_model(f1, f2, f3, kernel1, stride1, dr=0.5, in_shp=[2,1024]):
    model = models.Sequential()
    model.add(Reshape(in_shp+[1], input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(f1, kernel_size=kernel1, strides=stride1, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(f2, kernel_size=(2,3), border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(f3, activation='relu', init='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense( len(classes), init='he_normal', name="dense2" ))
    model.add(Activation('softmax'))
    model.add(Reshape([len(classes)]))
    model.summary()
    return model

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
params = get_params(size=N)
print(params)
for i in range(N):
    trial_param = [v[i] for v in params]
    print(trial_param)
    model = build_model(*trial_param)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath = basedir+'convmodrecnets_{}.wts.h5'.format(i)
    train_batch_size, number_of_epochs = 128, 2
    tsteps = data.train_idx.size/train_batch_size
    vsteps = data.val_idx.size/train_batch_size
    train_batches = data.batch_iter(data.train_idx, train_batch_size, number_of_epochs, use_shuffle=True)
    val_batches = data.batch_iter(data.val_idx, train_batch_size, number_of_epochs, use_shuffle=False)
    try:
         history = model.fit_generator(train_batches,
            nb_epoch=number_of_epochs,
            steps_per_epoch=tsteps,
            verbose=2,
            validation_data=val_batches,
            validation_steps=vsteps,
            callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
            ],
            use_multiprocessing=True,
            workers=6)
    except(StopIteration):
        pass
    model.load_weights(filepath)

    # Plot confusion matrix
    acc = {}
    snrs = np.arange(-15,15, 5)
    for snr in snrs:

        # extract classes @ SNR
        snrThreshold_lower = snr
        snrThreshold_upper = snr+5
        snr_bounded_test_indicies = testdata.get_indicies_withSNRthrehsold(data.test_idx, snrThreshold_lower, snrThreshold_upper)

        test_X_i = testdata.signalData[snr_bounded_test_indicies]
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
        fd = open(basedir+'results_{}.dat'.format(i),'wb')
        pickle.dump( (trial_param, acc) , fd )
