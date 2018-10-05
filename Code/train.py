import numpy as np
from data_loader import *
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Activation, Lambda, Average, Input, Reshape, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Dense
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model
import argparse

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_dir', type=str, default='/datax/yzhang/models/',
                    help='an integer for the accumulator')
parser.add_argument('--load_json', type=bool, default=False)
parser.add_argument('--load_weights', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--resample', type=int, default=None)
parser.add_argument('--m0', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--noise', type=float, default=-1.)
parser.add_argument('--confireg', type=float, default=-1.)
parser.add_argument('--crop_to', type=int, default=1024)
parser.add_argument('--num_models', type=int, default=1)
parser.add_argument('--verbose', type=int, default=2)
parser.add_argument('--perturbp', type=float, default=-1.)
parser.add_argument('--val_file', type=int, default=13)
parser.add_argument('--lrpatience', type=int, default=8)
parser.add_argument('--stoppatience', type=int, default=8)
parser.add_argument('--minlr', type=float, default=0.00001)
parser.add_argument('--noiseclip', type=float, default=100.)
parser.add_argument('--test_file', type=int, default=-1)
parser.add_argument('--num_files', type=int, default=10,
                    help='total number of files to use for training')
parser.add_argument('--mod_group', type=int, default=0)
parser.add_argument('--data_dir', type=str, default='/datax/yzhang/training_data/',
                    help='an integer for the accumulator')
parser.add_argument('--data_files', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--data_format', type=str, default="channels_last",
                    help='an integer for the accumulator')
parser.add_argument('--sep', type=bool, default=False)
parser.add_argument('--classifier_name', type=str, default="sub_classifer.h5")
args = parser.parse_args()

#facebook linear scaling rule. 
args.lr  = args.lr * args.ngpu * (args.batch_size / 512)
args.batch_size = args.batch_size * args.ngpu

# CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
#  'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
#  'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
#  'QAM32', 'QAM64', 'QPSK']
CLASSES = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4',
       'QAM16', 'QAM64', 'QPSK', 'WBFM']
all_mods = [np.arange(len(CLASSES)), np.array([1,9,10,11,12,13]), 
            np.array([4,5]), np.array([1,9]), np.array([6,7,20,21,22]), np.array([0,3]), np.array([0,3,6,7,20,21,22])]
mods = all_mods[args.mod_group]
num_classes = mods.size
BASEDIR = args.data_dir
model_path = args.train_dir+args.classifier_name

if not os.path.exists(args.train_dir):
     os.makedirs(args.train_dir)
data = []

load_mods = [CLASSES[mod] for mod in mods]
for i in range(args.num_files):
    if i in [ args.test_file]: continue
    data_file = BASEDIR + "part_" + str(i) + ".dat"
    #data_file = BASEDIR + "training_data_chunk_" + str(i) + ".pkl"
    if not args.sep:
        data.append(LoadModRecData(data_file, 1., 0., 0., load_mods=load_mods))
    else:
        data.append(data_file)

testdata = None
if args.test_file > 0:
    #data_file = BASEDIR + "training_data_chunk_" + str(args.test_file) + ".pkl"
    data_file = BASEDIR + "part_" + str(i) + ".dat"
    testdata = LoadModRecData(data_file, 0., 0., 1., load_mods=load_mods)


#global conv_index
#conv_index=0
def inception(input_img, height = 1, fs=[64,64,64,64,64], with_residual=False, tw_tower=False):
    tower_1 = Conv2D(filters=fs[0], kernel_size=[height, 1], padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(filters=fs[2], kernel_size=[height, 1], padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(filters=fs[3], kernel_size=[height, 9], padding='same', activation='relu')(tower_2)
    tower_3 = Conv2D(filters=fs[2], kernel_size=[height, 1], padding='same', activation='relu')(input_img)
    tower_3 = Conv2D(filters=fs[3], kernel_size=[height, 4], padding='same', activation='relu')(tower_3)
    if tw_tower:
        tower_5 = Conv2D(filters=fs[2], kernel_size=[height, 1], padding='same', activation='relu')(input_img)
        tower_5 = Conv2D(filters=fs[3], kernel_size=[height, 12], padding='same', activation='relu')(tower_5)
    tower_4 = MaxPooling2D(3, strides=1, padding='same')(input_img)
    tower_4 = Conv2D(filters=fs[4], kernel_size=1, padding='same', activation='relu')(tower_4)
    if tw_tower:
        output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4, tower_5], axis = 3)
    else:
        output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)
    if with_residual and output.shape==input_img.shape:
        output = output+input_img
    print()
    return output

def out_tower(x, dr=0.5, reg=-1):
    x = Dropout(dr)(x)
    output = Flatten()(x)
    logits    = Dense(num_classes)(output)
    if reg > 0:
        #logits = reg * Activation('tanh')(logits)
        logits = Lambda(lambda x: reg*K.tanh(logits))(logits)
    out = Activation('softmax')(logits)
    return out

def googleNet(x, data_format='channels_last', num_classes=24,num_layers=[1,1,2,1], features=[1,1,1,1,1]):
    
    x = Reshape(in_shp + (1,), input_shape=in_shp)(x)
    x = Lambda(lambda x: x * 1.e3)(x)
    x = Conv2D(filters=64*features[0], kernel_size=[2,7], strides=[2,2], data_format=data_format, padding='same', activation='relu')(x)
    x = MaxPooling2D([1, 3], strides=[1,2], padding='same')(x)
    for dep in range(num_layers[0]):
        x = Conv2D(filters=192*features[1], kernel_size=[1, 3], strides=[1,1], padding='same', activation='relu')(x)
    x = MaxPooling2D([1,3], strides=[1,2], padding='same')(x)
    for dep in range(num_layers[1]):
        x = inception(x, height=2, fs=np.array([32,32,32,32,32])*features[2], tw_tower=True)
    x = MaxPooling2D([1,3], strides=2, padding='same')(x)
    for dep in range(num_layers[2]):
        x = inception(x, height=2, fs=np.array([48,96,48,96,96])*features[3], with_residual=True)
    #out_mid = out_tower(x, dr=0.3)
    #for dep in range(num_layers[3]):
    #    x = inception(x, height=2, fs=np.array([48,96,48,96,96])*features[4], with_residual=True)
    x = MaxPooling2D([2,3], strides=2, padding='same')(x)
    for dep in range(num_layers[3]):
        x = inception(x, height=1,fs=np.array([32,32,32,32,32])*features[4])
    out = out_tower(x, dr=0.5, reg=args.confireg)
    #out = Average()([out_mid, out_late])
    return out




for m in range(args.m0, args.m0+args.num_models):
    
    

    if not args.sep:
        
        valdata = data[m]
    
        val_gen = valdata.batch_iter(valdata.train_idx, args.batch_size, args.epochs, use_shuffle=False)
        vsteps = valdata.train_idx.size//args.batch_size
    
        val_batches = get_val_batches(val_gen)
    
    
        generators = []
        tsteps = 0
        for i, d in enumerate(data):
            if i == m:
                continue
            generators.append(d.batch_iter(d.train_idx, args.batch_size, args.epochs, use_shuffle=True, yield_snr=True))
            tsteps += d.train_idx.size
        tsteps = tsteps//args.batch_size 
        train_batches = get_train_batches(generators, train_batch_size=args.batch_size, noise=args.noise, perturbp=args.perturbp)
    else:
        
        valdata = data[m]
        valdata = LoadModRecData(valdata, 1., 0., 0., load_mods=load_mods)
        val_gen = valdata.batch_iter(valdata.train_idx, args.batch_size, args.epochs, use_shuffle=False)
        vsteps = valdata.train_idx.size//args.batch_size
    
        val_batches = get_val_batches(val_gen)
        train_data = data.copy() 
        train_data.pop(m)
        tsteps = len(train_data) * 22000 * num_classes //args.batch_size 
        tsteps_per_file = 22000 * num_classes * 1 // args.batch_size
        train_batches = get_train_batches_small_memory(train_data, train_batch_size=args.batch_size, number_of_epochs=args.epochs,
                                                       tsteps_per_file=tsteps_per_file,load_mods=load_mods,
                                                       noise=args.noise, perturbp=args.perturbp)
        #import IPython; IPython.embed()
        
    in_shp = (2, args.crop_to)
    model_path = args.train_dir+'model{}.h5'.format(m)
    if args.ngpu > 1:
        with tf.device("/cpu:0"):
            input_img = Input(shape=in_shp)
            out = googleNet(input_img,data_format='channels_last', num_classes=num_classes)
            model = Model(inputs=input_img, outputs=out)
        model = multi_gpu_model(model, gpus=args.ngpu)
    else:
        input_img = Input(shape=in_shp)
        out = googleNet(input_img,data_format='channels_last', num_classes=num_classes)
        model = Model(inputs=input_img, outputs=out)
    loss_func = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
    model.compile(loss=loss_func, optimizer=Adam(lr=args.lr))
    filepath = args.train_dir+'checkpoints{}.h5'.format(m)

    try:
        history = model.fit_generator(train_batches,
            nb_epoch=args.epochs,
            steps_per_epoch=tsteps,
            verbose=args.verbose,
            validation_data=val_batches,
            validation_steps=vsteps,
            callbacks = [
              keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_weights_only=False, save_best_only=True, mode='auto'),
              #keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=args.lrpatience, min_lr=args.minlr),
              keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.stoppatience,verbose=0, mode='auto'),

              #keras.callbacks.TensorBoard(log_dir=args.train_dir+'/logs{}'.format(m), histogram_freq=0, batch_size=args.batch_size, write_graph=False)
             ]) 
    except(StopIteration):
        pass
    model.load_weights(filepath)
    model.save(model_path)  
    
    if args.test_file < 0: continue 
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
