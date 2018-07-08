import numpy as np
from data_loader import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure


import keras
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Dense
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model



BASEDIR = '/datax/yzhang/training_data/'
model_path = '/datax/yzhang/morad_classifier1.h5'

data = []
for i in range(13):
    data_file = BASEDIR + "training_data_chunk_" + str(i) + ".pkl"
    data.append(LoadModRecData(data_file, 1., 0., 0.))
    

    


data_file = BASEDIR + "training_data_chunk_13.pkl"
valdata = LoadModRecData(data_file, 1., 0., 0.)


data_file = BASEDIR + "training_data_chunk_14.pkl"
testdata = LoadModRecData(data_file, 0., 0., 1.)


#global conv_index
#conv_index=0
def inception(input_img, height = 1, fs=[64,64,64,64,64], with_residual=False):
    tower_1 = Conv2D(filters=fs[0], kernel_size=[height, 1], padding='same', activation='relu')(input_img)
    #conv_index += 1
    tower_2 = Conv2D(filters=fs[2], kernel_size=[height, 1], padding='same', activation='relu')(input_img)
    #conv_index += 1
    tower_2 = Conv2D(filters=fs[3], kernel_size=[height, 8], padding='same', activation='relu')(tower_2)
    #conv_index += 1
    tower_3 = Conv2D(filters=fs[2], kernel_size=[height, 1], padding='same', activation='relu')(input_img)
    #conv_index += 1
    tower_3 = Conv2D(filters=fs[3], kernel_size=[height, 4], padding='same', activation='relu')(tower_3)
    #conv_index += 1
    # tower_5 = Conv2D(filters=fs[2], kernel_size=[height, 1], padding='same', activation='relu')(input_img)
    # tower_5 = Conv2D(filters=fs[3], kernel_size=[height, 2], padding='same', activation='relu')(tower_5)
    tower_4 = MaxPooling2D(3, strides=1, padding='same')(input_img)
    tower_4 = Conv2D(filters=fs[4], kernel_size=1, padding='same', activation='relu')(tower_4)
    output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)
    if with_residual and output.shape==input_img.shape:
        output = output+input_img
    #print("end inception",conv_index)
    print()
    return output

def googleNet(x, data_format='channels_last'):
#     num_layers = [2,4,10,4]
    num_layers = [1,2,2,1]
    x = Reshape(in_shp + (1,), input_shape=in_shp)(x)
    x = Conv2D(filters = 64, kernel_size=[2,7], strides=[2,2], data_format=data_format, padding='same', activation='relu')(x)
    x = MaxPooling2D([1, 3], strides=[1,2], padding='same')(x)
    for dep in range(num_layers[0]):
        x = Conv2D(filters = 192, kernel_size=[1, 3], strides=[1,1], padding='same', activation='relu')(x)
    x = MaxPooling2D([1,3], strides=[1,2], padding='same')(x)
    for dep in range(num_layers[1]):
        x = inception(x, height=2, fs=[32,32,32,32,32])
    x = MaxPooling2D([1,3], strides=2, padding='same')(x)
    for dep in range(num_layers[2]):
        x = inception(x, height=2, fs=[48,96,48,96,96], with_residual=True)
    x = MaxPooling2D([2,3], strides=2, padding='same')(x)
    for dep in range(num_layers[3]):
        x = inception(x, height=1,fs=[32,32,32,32,32])
#     x = GlobalAveragePooling1D()(x)
#     x = Conv2D(filters=64, kernel_size=[1,1], padding='same', activation='relu')(x) # optional dim reduction

    x = Dropout(0.45)(x)
    output = Flatten()(x)
    out    = Dense(24, activation='softmax')(output)
    return out

in_shp = (2, 1024)
input_img = Input(shape=in_shp)
out = googleNet(input_img,data_format='channels_last')
model = Model(inputs=input_img, outputs=out)
model.summary()








#plot_model(model, to_file='model.png', show_shapes = True)







train_batch_size, number_of_epochs = 512, 100

val_batches = valdata.batch_iter(valdata.train_idx, train_batch_size, number_of_epochs, use_shuffle=False)
vsteps = valdata.train_idx.size//train_batch_size


generators = []
tsteps = 0
for d in data:
    generators.append(d.batch_iter(d.train_idx, train_batch_size, number_of_epochs, use_shuffle=False))
    tsteps += d.train_idx.size

tsteps = tsteps//train_batch_size 



def train_batches():
    while True:
        batches_x, batches_y = [], []

        for gen in generators:
            batch_x, batch_y = next(gen)
            batches_x.append(batch_x)
            batches_y.append(batch_y)
            
        batches_x = np.concatenate(batches_x)
        batches_y = np.concatenate(batches_y)
        
        idx = np.random.permutation(batches_x.shape[0])
        
        batches_x = batches_x[idx]
        batches_y = batches_y[idx]
        
        for i in range(len(generators)):
            beg = i * train_batch_size
            end = beg + train_batch_size
            yield batches_x[beg:end], batches_y[beg:end]
        

train_batches = train_batches()




model = multi_gpu_model(model, gpus=2)
model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath = '/datax/yzhang/morad_model.h5'

# model.load_weights(filepath)
history = model.fit_generator(train_batches,
    nb_epoch=number_of_epochs,
    steps_per_epoch=tsteps,
    verbose=2,
    validation_data=val_batches,
    validation_steps=vsteps,
     callbacks = [
          keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss'    , verbose=0, save_best_only=True, mode='auto'),
          keras.callbacks.EarlyStopping(monitor='val_loss', patience=8    , verbose=0, mode='auto')
     ]) 
model.save(model_path)  

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
    # plt.figure(figsize=(10,10))
    # plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("SNR", snr, "Overall Accuracy: ", cor / (cor+ncor), "Out of", len(snr_bounded_test_indicies))
    acc[snr] = 1.0*cor/(cor+ncor)



message = client.messages.create(
                              body='Finished Training!',
                              from_='+14156104527',
                              to='+19257322858'
                          )

print("Done")
