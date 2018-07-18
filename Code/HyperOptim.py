from __future__ import print_function
from data_loader import *
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.regularizers import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import np_utils
import keras.models as models
from hyperas import optim
from hyperas.distributions import choice, uniform
import tensorflow as tf
from keras.utils import multi_gpu_model
import keras

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    BASEDIR = "/data2/army_challenge/training_data/"
    data_file = BASEDIR+"training_data_chunk_0.pkl"
    data = LoadModRecData(data_file, .9, .1, 0.)
    test_file = BASEDIR+"training_data_chunk_14.pkl"
    testdata = LoadModRecData(test_file, 0., 0., 1.)
    
    x_train = data.signalData[data.train_idx]
    y_train = data.oneHotLabels[data.train_idx] 
    x_test = testdata.signalData[testdata.test_idx]
    y_test = testdata.oneHotLabels[testdata.test_idx] 
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    classes = list(range(24))
    in_shp = [2,1024]
    if True:
        model = models.Sequential()
        model.add(Reshape(in_shp+[1], input_shape=in_shp))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Convolution2D({{choice([128, 256, 512])}}, kernel_size={{choice([(1,3),(1,5)])}}, strides={{choice([(1,1),(1,2)])}}, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(ZeroPadding2D((0, 2)))
        model.add(Convolution2D({{choice([64, 128, 256])}}, kernel_size=(2,3), border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
        model.add(Dropout({{uniform(0, 1)}}))
        # If we choose 'four', add an additional fourth layer
        if {{choice(['three', 'four'])}} == 'four':
            model.add(Convolution2D({{choice([64, 128])}}, kernel_size=(1,3), strides=(1,2), border_mode="valid", activation="relu", name="conv3", init='glorot_uniform'))
            # We can also choose between complete sets of layers
            model.add({{choice([Dropout(0.5), Activation('linear')])}})
            #model.add(Activation('relu'))
        if {{choice([0,1])}} >0:
                rshape = model.output_shape
                model.add(Reshape([rshape[2],rshape[3]]))
                model.add(LSTM({{choice([64,128])}}, return_sequences=True, name='lstm1'))
             
        model.add(Flatten())
        model.add(Dense({{choice([32, 64, 128])}}, activation='relu', init='he_normal', name="dense1"))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense( len(classes), init='he_normal', name="dense2" ))
        model.add(Activation('softmax'))
        model.add(Reshape([len(classes)]))
        model.summary()
    #model = multi_gpu_model(model, gpus=2)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='adam')

    model.fit(x_train, y_train,
              batch_size={{choice([128, 256])}},
              epochs=20,
              verbose=2,
              validation_data=(x_test[:1000], y_test[:1000]),
              callbacks = [
                  keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
              ])
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
