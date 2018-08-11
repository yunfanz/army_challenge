import numpy as np
from data_loader import *
from utils import *
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model, multi_gpu_model
import argparse

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_dir', type=str, default='/datax/yzhang/models/',
                    help='an integer for the accumulator')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--nhidden', type=int, default=32)
parser.add_argument('--test_thresh', type=float, default=0.85)
parser.add_argument('--resample', type=int, default=None)
parser.add_argument('--m0', type=int, default=0)
parser.add_argument('--startdraw', type=int, default=20000,
                     help="step number to start drawing from test set 1")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dislr', type=float, default=0.0005)
parser.add_argument('--ganlr', type=float, default=0.0002)
parser.add_argument('--noise', type=float, default=-1.)
parser.add_argument('--confireg', type=float, default=-1.)
parser.add_argument('--crop_to', type=int, default=1024)
parser.add_argument('--num_models', type=int, default=1)
parser.add_argument('--verbose', type=int, default=2)
parser.add_argument('--val_file', type=int, default=13)
parser.add_argument('--test_file', type=int, default=-1)
parser.add_argument('--mod_group', type=int, default=4)
parser.add_argument('--test_dir', type=str, default='/datax/yzhang/test_data/')
parser.add_argument('--data_dir', type=str, default='/datax/yzhang/training_data/',
                    help='an integer for the accumulator')
parser.add_argument('--data_files', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--classifier_name', type=str, default="sub_classifer.h5")
args = parser.parse_args()

#facebook linear scaling rule. 
args.lr  = args.lr * args.ngpu * (args.batch_size / 512)
args.batch_size = args.batch_size * args.ngpu
args.test_thresh = args.test_thresh + np.random.uniform(-0.02,0.02)
CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
 'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
 'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
 'QAM32', 'QAM64', 'QPSK']

all_mods = [np.arange(24), np.array([1,9,10,11,12,13]), 
            np.array([4,5]), np.array([1,9]), np.array([6,7,20,21,22]), np.array([0,3]), np.array([0,3,6,7,20,21,22])]
mods = all_mods[args.mod_group]
num_classes = mods.size
BASEDIR = args.test_dir
model_path = args.train_dir+args.classifier_name

if not os.path.exists(args.train_dir):
     os.makedirs(args.train_dir)
data = []
for i in range(15):
    if i in [ args.test_file]: continue
    data_file = args.data_dir + "training_data_chunk_" + str(i) + ".pkl"
    data.append(LoadModRecData(data_file, 1., 0., 0., load_mods=[CLASSES[mod] for mod in mods]))

testdata = None
if args.test_file > 0:
    data_file = args.data_dir + "training_data_chunk_" + str(args.test_file) + ".pkl"
    testdata = LoadModRecData(data_file, 0., 0., 1., load_mods=[CLASSES[mod] for mod in mods])

#print('!!!', mods)
target_file = BASEDIR+"Test_Set_1_Army_Signal_Challenge.pkl"
target_file2 = BASEDIR+"Test_Set_2_Army_Signal_Challenge.pkl"
target_pred = BASEDIR+"TestSet1Predictions.csv"
target_pred2 = BASEDIR+"TestSet2Predictions.csv"
f = open(target_file, 'rb')
targetdata1 = pickle.load(f, encoding='latin1')
targetdata1 = np.stack([targetdata1[i+1] for i in range(len(targetdata1.keys()))], axis=0)
targetdata1 = targetdata1[get_mod_group(target_pred, mods, thresh=args.test_thresh)]
f = open(target_file2, 'rb')
targetdata2 = pickle.load(f, encoding='latin1')
targetdata2 = np.stack([targetdata2[i+1] for i in range(len(targetdata2.keys()))], axis=0)
targetdata2 = targetdata2[get_mod_group(target_pred2, mods, thresh=args.test_thresh)]
targetdata = np.concatenate([targetdata1, targetdata2], axis=0)
print('Got {} instances from set1, {} from set 2'.format(targetdata1.shape[0],targetdata2.shape[0]))
print()
#import IPython; IPython.embed()

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
    return output

def out_tower(x, dr=0.5, reg=-1):
    output = Dropout(dr)(x)
    logits    = Dense(num_classes)(output)
    if reg > 0:
        #logits = reg * Activation('tanh')(logits)
        logits = Lambda(lambda x: reg*K.tanh(logits))(logits)
    out = Activation('softmax')(logits)
    return out

def googleNet(x, nhidden=128, data_format='channels_last', num_classes=24,num_layers=[1,1,2,1], features=[1,1,1,1,1]):
    
    x = Reshape(in_shp + (1,), input_shape=in_shp)(x)
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
    
    x = Flatten()(x)
    x = Dense(nhidden)(x)
    #x = Dropout(0.5)(x)
    x = Lambda(lambda  x: K.l2_normalize(x), name='l2_normalize')(x)
    out = out_tower(x, dr=0.1, reg=args.confireg)
    #out = Average()([out_mid, out_late])
    return out, x

def discriminate(x, nhidden=128, dr=0.5):
    x = Reshape((nhidden, 1))(x)
    H = Conv1D(filters=512, kernel_size=5, strides=2, activation='relu')(x)
    #H = LeakyReLU(0.2)(H)
    H = Dropout(dr)(H)
    #H = Conv1D(filters=512, kernel_size=3, strides=2,  activation='relu')(H)
    # H = LeakyReLU(0.2)(H)
    #H = Dropout(dr)(H)
    H = Flatten()(x)
    H = Dense(256, activation='relu')(H)
    #H = LeakyReLU(0.2)(H)
    H = Dropout(dr)(H)
    d_V = Dense(2,activation='softmax')(H)
    return d_V


train_batch_size = args.batch_size

generators = []
tsteps = 0
for d in data:
    generators.append(d.batch_iter(d.train_idx, train_batch_size, args.epochs, use_shuffle=True))
    tsteps += d.train_idx.size

tsteps = tsteps//train_batch_size 


def get_train_batches(generators):
    while True:
        batches_x, batches_y, batches_snr = [], [], []

        for gen in generators:
            batch_x, batch_y, batch_labels = next(gen)
            batches_x.append(batch_x)
            batches_y.append(batch_y)
            batches_snr.append(10**((batch_labels[:,1]).astype(np.float)/10.))
            
        batches_x = np.concatenate(batches_x)
        batches_y = np.concatenate(batches_y)
        batches_snr = np.concatenate(batches_snr)
        idx = np.random.permutation(batches_x.shape[0])

        #
        if args.noise > 0:
            shp0, shp1, shp2 = batches_x.shape
            noisestd = args.noise/batches_snr[:,np.newaxis, np.newaxis]
            batches_x += noisestd * np.random.randn(shp0, shp1, shp2)
                
        if args.resample is not None and np.random.random()>0.8:
            batches_x = resample(batches_x, f=args.resample)
              
        batches_x = batches_x[idx]
        batches_y = batches_y[idx]
        batches_snr = batches_snr[idx]
        
        for i in range(len(generators)):
            beg = i * train_batch_size
            end = beg + train_batch_size
            bx, by, bs = batches_x[beg:end], batches_y[beg:end], batches_snr[beg:end]

            yield bx, by
        

train_batches = get_train_batches(generators)

def get_val_batches(gen):
    while True:
        bx, by = next(gen)
        if args.crop_to < 1024:
            c_start = np.random.randint(low=0, high=1024-args.crop_to)
            bx = bx[...,c_start:c_start+args.crop_to]
            assert bx.shape[-1] == args.crop_to
        yield bx, by

for m in range(args.m0, args.m0+args.num_models):
    
    valdata = data[m]
    val_gen = valdata.batch_iter(valdata.train_idx, train_batch_size, args.epochs, use_shuffle=False)
    vsteps = valdata.train_idx.size//train_batch_size
    val_batches = get_val_batches(val_gen)


    generators = []
    tsteps = 0
    for i, d in enumerate(data):
        if i == m:
            continue
        generators.append(d.batch_iter(d.train_idx, train_batch_size, args.epochs, use_shuffle=True, yield_snr=True))
        tsteps += d.train_idx.size
    tsteps = tsteps//train_batch_size 
    train_batches = get_train_batches(generators)
    #val_batches = get_val_batches(val_gen)

    in_shp = (2, args.crop_to)
    input_img = Input(shape=in_shp); input_img_ = Input(shape=in_shp)
    out, emb = googleNet(input_img,nhidden=args.nhidden,data_format='channels_last', num_classes=num_classes)
    model = Model(inputs=input_img, outputs=out)
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.lr))

    d_in = Input(shape=(args.nhidden,))
    d_V = discriminate(d_in, nhidden=args.nhidden)
    discriminator = Model(d_in, d_V)
    discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.dislr))

    make_trainable(discriminator, False)

    # Build stacked GAN model
    emb_model = Model(inputs=input_img, outputs=emb)  #not compiled??
    gan_input = Input(shape=in_shp)
    EMB = emb_model(gan_input)
    gan_V = discriminator(EMB)
    GAN = Model(gan_input, gan_V)
    GAN.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.ganlr))


    d_loss = 0; g_loss = 0; c_loss = 0 
    losses = {"d":[], "g":[], "c":[], "v":[]}
    v_loss_min = 2.
    for step in range(args.epochs*(targetdata.shape[0]//args.batch_size)*10):  #actually num_batches
        try:
            # Make generative images
            bx, by = next(train_batches)
        except(StopIteration):
            break
        if step < args.startdraw:
            bx_ = targetdata2[np.random.randint(0, high=targetdata2.shape[0], size=args.batch_size)]   
        else:
            bx_ = targetdata[np.random.randint(0, high=targetdata.shape[0], size=args.batch_size)]
        if step == args.startdraw:
            print('Start drawing from test set 1')
        if step % 400 == 0 and step % 1000 != 0:
            vx, vy = next(val_batches)
            v_loss = model.test_on_batch(vx, vy)
            #losses["v"].append(v_loss)
            print("Step {}, classification loss {}, discriminator loss {}, GAN loss {}, validation loss {}".format(step, c_loss, d_loss, g_loss, v_loss))
        elif step % 1000 == 0:
            vls = []
            for b in range(10):
                vx, vy = next(val_batches)
                vls.append(model.test_on_batch(vx, vy))
            v_loss = np.mean(vls)
            if v_loss < v_loss_min:
                print("saving checkpoint, loss=", v_loss)
                model.save_weights(args.train_dir+'checkpoint{}.h5'.format(m))
                v_loss_min = v_loss
        #import IPython; IPython.embed()
        #make_trainable(discriminator,False)
        c_loss = model.train_on_batch(bx, by)
        losses["c"].append(c_loss)
        
        #make_trainable(discriminator,True)
        emb = emb_model.predict(bx)
        emb_ = emb_model.predict(bx_)
        # Train discriminator on generated images
        domain_X = np.concatenate((emb, emb_), axis=0)
        domain_y = np.zeros([2*args.batch_size,2])
        domain_y[0:args.batch_size,1] = 1
        domain_y[args.batch_size:,0] = 1 #0 for target domain
        if c_loss > 1.0 : #warm up classifier
            d_loss  = 0.
        else:
            d_loss  = discriminator.train_on_batch(domain_X,domain_y)
        losses["d"].append(d_loss)
        
        #make_trainable(discriminator,False)
        y2 = np.zeros([args.batch_size,2])
        y2[:,1] = 1  #1 for target domain
        if c_loss > 0.9:
            g_loss = GAN.test_on_batch(bx_, y2 )
        else:
            g_loss = GAN.train_on_batch(bx_, y2 )
        losses["g"].append(g_loss)

        if True and step>args.startdraw+8000 and step % 15000== 0: # one epoch of test data
            epochc_loss = np.mean(losses["c"][-1000:])
            meanc_loss = np.mean(losses["c"][-15000:])
            lr = K.eval(model.optimizer.lr)
            dislr = K.eval(discriminator.optimizer.lr)
            ganlr = K.eval(GAN.optimizer.lr)
            if meanc_loss <= epochc_loss:
                if (lr >= 1.e-4 and dislr > 1.e-5 and ganlr > 1.e-5):
                    print("reducing learning rate from", lr, dislr, ganlr)
                    K.set_value(model.optimizer.lr, 0.1*lr)
                    if ganlr < 1.e-4:
                        ganlr *= 2
                        dislr *= 2
                    K.set_value(discriminator.optimizer.lr, 0.1*dislr)
                    K.set_value(GAN.optimizer.lr, 0.1*ganlr)
                else:
                    print ("Stopping early")
                    break


    model_path = args.train_dir+'model{}.h5'.format(m)
    model.load_weights(args.train_dir+'checkpoint{}.h5'.format(m))
    model.save(model_path)  
    print("Done {}/{}".format(m, args.num_models)) 
