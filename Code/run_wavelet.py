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
from resnet2D import ResnetBuilder


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
parser.add_argument('--num_classes', type=int, default=24,
                    help='an integer for the accumulator')
parser.add_argument('--data_dir', type=str, default='/data2/army_challenge/training_data/',
                    help='an integer for the accumulator')
parser.add_argument('--data_files', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--data_format', type=str, default="channels_last",
                    help='an integer for the accumulator')
parser.add_argument('--sep', type=bool, default=False)
args = parser.parse_args()

if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)

print('Data dir:', args.data_dir)

img_channels, img_rows, img_cols = 2, 1024, 20
model = ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), 5)

x_train, y_train, x_val, y_val = get_data(mode='wavelet',
                                          load_mods=['CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB', 'GFSK_5KHz'],
                                         BASEDIR=args.data_dirs,
                                         files=[0])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')
hist = model.fit(x_train, y_train,
          batch_size=32,
          nb_epoch=20,
          validation_data=[x_val, y_val],
          shuffle=True,
          verbose=1,
          callbacks=[lr_reducer, early_stopper])