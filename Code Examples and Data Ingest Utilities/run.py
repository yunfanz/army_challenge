import os,random
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.models import model_from_json
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
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
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--model', type=int, default=2,
                    help='an integer for the accumulator')
parser.add_argument('--epochs', type=int, default=10,
                    help='an integer for the accumulator')
parser.add_argument('--data_dir', type=str, default='/data2/army_challenge/training_data/',
                    help='an integer for the accumulator')
parser.add_argument('--data_files', type=int, nargs='+',
                    help='an integer for the accumulator')

args = parser.parse_args()
""" Sample usage:
CUDA_VISIBLE_DEVICES=3 python run.py --train_dir=./log/model_277/ --model 2 --data_files 0 1 --epochs 2
"""
if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)


model_9 = {'depths': np.array([2, 2, 1, 0, 2, 1]), 'features': np.array([3, 2, 3, 3, 3, 3, 3]), 'dr': 0.67561133072930946}
model_44 = {'depths': np.array([2, 2, 0, 3, 0, 2]), 'features': np.array([1, 3, 1, 2, 2, 2, 2]), 'dr': 0.24749480935162974}
model_277 = {'depths': np.array([1, 1, 0, 0, 0, 1]), 'features': np.array([3, 1, 2, 2, 3, 3, 3]), 'dr': 0.54753203788931493}
models = [model_9, model_44, model_277]

input_img = Input(shape=(1024,2))
if args.model<3:
    out = googleNet(input_img,data_format='channels_last', pdict=models[args.model])
elif args.model == 5:
    out = googleNet_2D(input_img,data_format='channels_last')
model = Model(inputs=input_img, outputs=out)

model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)
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
    x_train, y_train, x_val, y_val = get_data(mode='time_series',
                                         BASEDIR='/data2/army_challenge/training_data/',
                                         files=args.data_files)
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=args.epochs,
              verbose=2,
              validation_data=(x_val, y_val),
              callbacks=[reduce_lr, t_board])
    model.save_weights(args.train_dir+"weights.h5")
    model_json = model.to_json()
    with open(args.train_dir+"model.json", "w") as json_file:
        json_file.write(model_json)



test_file = args.data_dir+"training_data_chunk_14.pkl"
testdata = LoadModRecData(test_file, 0., 0., 1.)
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
    

