import os,random
import numpy as np
from keras.utils import np_utils
import keras.models
from keras.models import model_from_json, Sequential
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv1D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM, GRU
from keras.backend import squeeze
from keras.regularizers import *
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, random, sys, keras
from data_loader import *
from inception import *
import tensorflow as tf
from keras.utils import multi_gpu_model, plot_model
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_dir', type=str, default='./log/model_9/',
                    help='an integer for the accumulator')
parser.add_argument('--load_json', type=bool, default=False)
parser.add_argument('--load_weights', type=bool, default=False)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--model', type=int, default=2,
                    help='an integer for the accumulator')
parser.add_argument('--epochs', type=int, default=10,
                    help='an integer for the accumulator')
parser.add_argument('--num_classes', type=int, default=24,
                    help='an integer for the accumulator')
parser.add_argument('--data_dir', type=str, default='/data2/army_challenge/training_data/',
                    help='an integer for the accumulator')
parser.add_argument('--data_files', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--data_format', type=str, default="channels_last",
                    help='an integer for the accumulator')
parser.add_argument('--sep', type=bool, default=False)
parser.add_argument('--val_file', type=int, default=13)
#parser.add_argument('--', type=int, default=13)
parser.add_argument('--test_file', type=int, default=14)
parser.add_argument('--mod_group', type=int, default=0)
parser.add_argument('--verbose', type=int, default=2)
args = parser.parse_args()
""" Sample usage:
CUDA_VISIBLE_DEVICES=3 python run.py --train_dir=./log/model_277/ --model 2 --data_files 0 1 --epochs 2
"""
if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)
CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
 'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
 'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
 'QAM32', 'QAM64', 'QPSK']

all_mods = [np.arange(24), np.array([1,9,10,11,12,13]), np.array([4,5]), np.array([1,9])]
mods = all_mods[args.mod_group]
num_classes = mods.size

print('Data dir:', args.data_dir, args.data_files, args.data_format)
model_9 = {'depths': np.array([2, 2, 1, 0, 2, 1]), 'features': np.array([3, 2, 3, 3, 3, 3, 3]), 'dr': 0.67561133072930946}
model_44 = {'depths': np.array([2, 2, 0, 3, 0, 2]), 'features': np.array([1, 3, 1, 2, 2, 2, 2]), 'dr': 0.24749480935162974}
model_277 = {'depths': np.array([1, 1, 0, 0, 0, 1]), 'features': np.array([3, 1, 2, 2, 3, 3, 3]), 'dr': 0.54753203788931493}
model_0 = {'depths': np.array([2, 2, 2, 2, 2, 2]), 'features': np.array([2, 2, 2, 2, 2, 2, 2]), 'dr': 0.6}
models = [model_0, model_9, model_44]

input_img = Input(shape=(1024,2))
if args.model<len(models):
    print("Using model")
    print(models[args.model])
    out = googleNet(input_img,data_format='channels_last', pdict=models[args.model], num_classes=num_classes)
    model = Model(inputs=input_img, outputs=out)
elif args.model == 4:
    #input_img = Input(shape=(2,1024))
    out = googleNet_2D(input_img,data_format='channels_last', num_layers=[1,2,4,2], num_classes=num_classes)
    model = Model(inputs=input_img, outputs=out)
elif args.model == 5:
    #input_img = Input(shape=(2,1024))
    out = googleNet_2D(input_img,data_format='channels_last', num_classes=num_classes)
    model = Model(inputs=input_img, outputs=out)
elif args.model == 10:
    x = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(input_img)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256)(x)
    #x = LSTM(150)(x)
    #x = LSTM(150)(x)
    x = Dense(num_classes, activation='sigmoid')(x)
    model = Model(input_img, x)
elif args.model == 11:
    x = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(input_img)
    x = MaxPooling1D(pool_size=2)(x)
    #x = LSTM(150, return_sequences=True)(x)
    x = GRU(256, return_sequences=True)(x)
    x = GRU(256)(x)
    x = Dense(num_classes, activation='sigmoid')(x)
    model = Model(input_img, x)

model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=args.epochs//10, min_lr=0.0001)
t_board = TensorBoard(log_dir=args.train_dir, histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

#print('model compiled')

if args.load_json:
    # load json and create model
    json_file = open(args.train_dir+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
if args.load_weights:
    # load weights into new model
    model.load_weights(args.train_dir+"weights.h5")
if args.train:
    if not args.sep:
        x_train, y_train, x_val, y_val = get_data(mode='time_series', data_format=args.data_format,
                                         load_mods=[CLASSES[mod] for mod in mods],
                                         BASEDIR=args.data_dir,
                                         files=args.data_files)
    for e in range(args.epochs):
        if args.sep:
            x_train, y_train, x_val, y_val = get_data(mode='time_series', data_format=args.data_format,
                                         load_mods=[CLASSES[mod] for mod in mods],
                                         BASEDIR=args.data_dir,
                                         files=[np.random.choice(args.data_files)])
        x_train = augment(x_train)
        model.fit(x_train, y_train,
                  batch_size=256,
                  epochs=1,
                  verbose=args.verbose,
                  validation_data=(x_val, y_val),
                  callbacks=[reduce_lr, t_board])
    model.save(args.train_dir+"model.h5")
    model.save_weights(args.train_dir+"weights.h5")
    model_json = model.to_json()
    with open(args.train_dir+"model.json", "w") as json_file:
        json_file.write(model_json)



test_file = args.data_dir+"training_data_chunk_14.pkl"
testdata = LoadModRecData(test_file, 0., 0., 1., load_mods=[CLASSES[mod] for mod in mods])
# Plot confusion matrix
acc = {}
snrs = np.arange(-15,15, 5)

classes = testdata.modTypes

print("classes ", classes)
for snr in testdata.snrValues:

    # extract classes @ SNR
    snrThreshold_lower = snr
    snrThreshold_upper = snr+5
    snr_bounded_test_indicies = testdata.get_indicies_withSNRthrehsold(testdata.test_idx, snrThreshold_lower, snrThreshold_upper)
    
    test_X_i = testdata.signalData[snr_bounded_test_indicies]
    test_Y_i = testdata.oneHotLabels[snr_bounded_test_indicies]
    test_X_i = np.transpose(test_X_i, (0,2,1))

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
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    plt.savefig(args.train_dir+"ConfusionMatrixSNR=%d"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("SNR", snr, "Overall Accuracy: ", cor / (cor+ncor), "Out of", len(snr_bounded_test_indicies))
    acc[snr] = 1.0*cor/(cor+ncor)
    

