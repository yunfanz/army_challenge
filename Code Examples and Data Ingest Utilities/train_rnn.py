import numpy as np
from data_loader import *
from utils import *
from keras.layers.recurrent import LSTM, GRU
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import keras
from keras.layers import Input, Reshape, Conv1D, MaxPooling1D, ZeroPadding2D, Flatten, Dropout, Dense
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model
import argparse

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_dir', type=str, default='/datax/yzhang/models/',
                    help='an integer for the accumulator')
parser.add_argument('--load_json', type=bool, default=False)
parser.add_argument('--load_weights', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--noise', type=float, default=0.4)
parser.add_argument('--crop_to', type=int, default=1024)
parser.add_argument('--nperseg', type=int, default=512)
parser.add_argument('--noverlap', type=int, default=256)
parser.add_argument('--num_models', type=int, default=1)
parser.add_argument('--verbose', type=int, default=2)
parser.add_argument('--val_file', type=int, default=13)
parser.add_argument('--reducelr', type=int, default=8)
parser.add_argument('--test_file', type=int, default=14)
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


CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
 'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
 'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
 'QAM32', 'QAM64', 'QPSK']

all_mods = [np.arange(24), np.array([1,9,10,11,12,13]), np.array([4,5]), np.array([1,9])]
mods = all_mods[args.mod_group]
num_classes = mods.size
BASEDIR = args.data_dir
model_path = args.train_dir+args.classifier_name

if not os.path.exists(args.train_dir):
     os.makedirs(args.train_dir)
data = []
for i in range(15):
    if i in [ args.test_file]: continue
    data_file = BASEDIR + "training_data_chunk_" + str(i) + ".pkl"
    data.append(LoadModRecData(data_file, 1., 0., 0., load_mods=[CLASSES[mod] for mod in mods]))


data_file = BASEDIR + "training_data_chunk_" + str(args.test_file) + ".pkl"
testdata = LoadModRecData(data_file, 0., 0., 1., load_mods=[CLASSES[mod] for mod in mods])




print("STARTING NPERSEG {}, NOVERLAP {}".format(args.nperseg, args.noverlap))

train_batch_size, number_of_epochs = args.batch_size, args.epochs



generators = []
tsteps = 0
for d in data:
    generators.append(d.batch_iter(d.train_idx, train_batch_size, number_of_epochs, use_shuffle=True))
    tsteps += d.train_idx.size

tsteps = tsteps//train_batch_size 



def get_train_batches(generators):
    while True:
        batches_x, batches_y, batches_snr = [], [], []

        for gen in generators:
            batch_x, batch_y, batch_labels = next(gen)
            batches_x.append(batch_x)
            batches_y.append(batch_y)
            #print(batch_labels)
            batches_snr.append(10**((batch_labels[:,1]).astype(np.float)/10.))
            
        batches_x = np.concatenate(batches_x)
        batches_y = np.concatenate(batches_y)
        batches_snr = np.concatenate(batches_snr)
        idx = np.random.permutation(batches_x.shape[0])
        
        
        if args.noise > 0:
            shp0, shp1, shp2 = batches_x.shape
            batches_x += args.noise * np.random.randn(shp0, shp1, shp2)/batches_snr[:,np.newaxis, np.newaxis]
                
                
        batches_x = batches_x[idx]
        batches_y = batches_y[idx]
        batches_snr = batches_snr[idx]
        
        for i in range(len(generators)):
            beg = i * train_batch_size
            end = beg + train_batch_size
            bx, by, bs = batches_x[beg:end], batches_y[beg:end], batches_snr[beg:end]
            if False and np.random.random()>0.5:
                bx = bx[...,::-1]
            
            if args.crop_to < 1024:
                c_start = np.random.randint(low=0, high=1024-args.crop_to)
                bx = bx[...,c_start:c_start+args.crop_to]
                assert bx.shape[-1] == args.crop_to
            bx = get_welch(bx, window='hann', nperseg=args.nperseg, 
                           noverlap=args.noverlap, remove_DC=False, in_format='channels_first')
            #print(bx.shape)
            yield bx, by
        

train_batches = get_train_batches(generators)



def get_val_batches(gen):
    while True:
        bx, by = next(gen)
        if args.crop_to < 1024:
            c_start = np.random.randint(low=0, high=1024-args.crop_to)
            bx = bx[...,c_start:c_start+args.crop_to]
            assert bx.shape[-1] == args.crop_to
        bx = get_welch(bx, window='hann', nperseg=args.nperseg, 
                           noverlap=args.noverlap, remove_DC=False, in_format='channels_first')
        yield bx, by

for m in range(args.num_models):
    
    valdata = data[m]
    
    val_gen = valdata.batch_iter(valdata.train_idx, train_batch_size, number_of_epochs, use_shuffle=False)
    vsteps = valdata.train_idx.size//train_batch_size
    
    val_batches = get_val_batches(val_gen)


    generators = []
    tsteps = 0
    for i, d in enumerate(data):
        if i == m:
            continue
        generators.append(d.batch_iter(d.train_idx, train_batch_size, number_of_epochs, use_shuffle=True, yield_snr=True))
        tsteps += d.train_idx.size
    tsteps = tsteps//train_batch_size 
    train_batches = get_train_batches(generators)
    val_batches = get_val_batches(val_gen)

    in_shp = (args.nperseg, 1)
    input_img = Input(shape=in_shp)
    x = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(input_img)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256)(x)
    #x = LSTM(150)(x)
    #x = LSTM(150)(x)
    x = Dense(num_classes, activation='sigmoid')(x)
    model = Model(input_img, x)
    if m == 0:
        model.summary()
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
              #keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=args.reducelr, min_lr=0.0001),
              keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,verbose=0, mode='auto'),

              keras.callbacks.TensorBoard(log_dir=args.train_dir+'/logs{}'.format(m), histogram_freq=0, batch_size=args.batch_size, write_graph=False)
             ]) 
    except(StopIteration):
        pass
    #model.load_weights(filepath)
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
        test_X_i = get_welch(test_X_i, window='hann', nperseg=args.nperseg, 
                           noverlap=args.noverlap, remove_DC=False, in_format='channels_first')
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
