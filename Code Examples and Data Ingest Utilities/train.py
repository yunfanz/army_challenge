import numpy as np
from data_loader import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import keras
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Dense
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model
import argparse

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_dir', type=str, default='/datax/yzhang/models/',
                    help='an integer for the accumulator')
parser.add_argument('--load_json', type=bool, default=False)
parser.add_argument('--load_weights', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--noise', type=float, default=-1.)
parser.add_argument('--crop_to', type=int, default=1024)
parser.add_argument('--num_models', type=int, default=1)
parser.add_argument('--verbose', type=int, default=2)
parser.add_argument('--val_file', type=int, default=13)
#parser.add_argument('--', type=int, default=13)
parser.add_argument('--test_file', type=int, default=14)
parser.add_argument('--mod_group', type=int, default=0)
parser.add_argument('--data_dir', type=str, default='/datax/yzhang/training_data/',
                    help='an integer for the accumulator')
parser.add_argument('--data_files', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--data_format', type=str, default="channels_last",
                    help='an integer for the accumulator')
parser.add_argument('--sep', type=bool, default=False)
args = parser.parse_args()


CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
 'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
 'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
 'QAM32', 'QAM64', 'QPSK']

all_mods = [np.arange(24), np.array([1,9,10,11,12,13]), np.array([4,5]), np.array([1,9])]
mods = all_mods[args.mod_group]
num_classes = mods.size
BASEDIR = args.data_dir

if not os.path.exists(args.train_dir):
     os.makedirs(args.train_dir)
data = []
for i in range(15):
    if i in [ args.test_file]: continue
    data_file = BASEDIR + "training_data_chunk_" + str(i) + ".pkl"
    data.append(LoadModRecData(data_file, 1., 0., 0., load_mods=[CLASSES[mod] for mod in mods]))


data_file = BASEDIR + "training_data_chunk_" + str(args.test_file) + ".pkl"
testdata = LoadModRecData(data_file, 0., 0., 1., load_mods=[CLASSES[mod] for mod in mods])


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

def googleNet(x, data_format='channels_last', num_classes=24,num_layers=[1,2,2,1]):
#     num_layers = [2,4,10,4]
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

    x = Dropout(0.45)(x)
    output = Flatten()(x)
    out    = Dense(num_classes, activation='softmax')(output)
    return out

train_batch_size, number_of_epochs = args.batch_size, args.epochs



def get_train_batches(generators):
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
            bx, by = batches_x[beg:end], batches_y[beg:end]
            if False and np.random.random()>0.5:
                bx = bx[...,::-1]
            if args.noise > 0:
                shp0, shp1, shp2 = bx.shape
                bx += args.noise * np.random.randn(shp0, shp1, shp2)
            if args.crop_to < 1024:
                c_start = np.random.randint(low=0, high=1024-args.crop_to)
                bx = bx[...,c_start:c_start+args.crop_to]
                assert bx.shape[-1] == args.crop_to
            yield bx, by
        
def get_val_batches(gen):
    while True:
        bx, by = next(gen)
        if args.crop_to < 1024:
            c_start = np.random.randint(low=0, high=1024-args.crop_to)
            bx = bx[...,c_start:c_start+args.crop_to]
            assert bx.shape[-1] == args.crop_to
        yield bx, by

for m in range(args.num_models):
    
    valdata = data[m]
    
    val_gen = valdata.batch_iter(valdata.train_idx, train_batch_size, number_of_epochs, use_shuffle=False)
    vsteps = valdata.train_idx.size//train_batch_size


    generators = []
    tsteps = 0
    for i, d in enumerate(data):
        if i == m:
            continue
        generators.append(d.batch_iter(d.train_idx, train_batch_size, number_of_epochs, use_shuffle=True))
        tsteps += d.train_idx.size
    tsteps = tsteps//train_batch_size 
    train_batches = get_train_batches(generators)
    val_batches = get_val_batches(val_gen)

    in_shp = (2, args.crop_to)
    input_img = Input(shape=in_shp)
    out = googleNet(input_img,data_format='channels_last', num_classes=num_classes)
    model = Model(inputs=input_img, outputs=out)
    model_path = args.train_dir+'model{}.h5'.format(m)
    if args.ngpu > 1:
        model = multi_gpu_model(model, gpus=args.ngpu)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath = args.train_dir+'checkpoints{}.h5'.format(m)

    try:
        history = model.fit_generator(train_batches,
            nb_epoch=number_of_epochs,
            steps_per_epoch=tsteps,
            verbose=args.verbose,
            validation_data=val_batches,
            validation_steps=vsteps,
            callbacks = [
              keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
              #keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  #patience=args.epochs//10, min_lr=0.0001),
              keras.callbacks.TensorBoard(log_dir=args.train_dir+'/logs{}'.format(m), histogram_freq=0, batch_size=args.batch_size, write_graph=False),
              keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,verbose=0, mode='auto')
             ]) 
    except(StopIteration):
        pass
    model.load_weights(filepath)
    model.save(model_path)  
    
    
    #Print test accuracies

    acc = {}
    scores = {}
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
        
        #sc, ac = model.evaluate(test_X_i, test_Y_i, batch_size=256)
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



    print("Done model {} out of {}".format(m, args.num_models))
