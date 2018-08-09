import numpy as np
from data_loader import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import pickle
import keras
import tensorflow as tf
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Dense
from keras.models import Model, load_model
from multiprocessing import Manager, Process, Queue, Pool
from functools import partial
import argparse
from time import sleep

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_dir', type=str, default='./log/model_9/',
                    help='an integer for the accumulator')
parser.add_argument('--all_snr', type=bool, default=False)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--test', type=int, default=1)
parser.add_argument('--eps', type=float, default=1.e-15)
parser.add_argument('--subweight', type=float, default=0.8)
parser.add_argument('--qampskweight', type=float, default=1.)
parser.add_argument('--qamweight', type=float, default=1.)
parser.add_argument('--pskweight', type=float, default=1.)
parser.add_argument('--model', type=str, default=None,
                    help='group 0 model')
parser.add_argument('--submodel', type=str, default=None,
                    help='an integer for the accumulator')
parser.add_argument('--qammodel', type=str, default=None)
parser.add_argument('--pskmodel', type=str, default=None)
parser.add_argument('--qampskmodel', type=str, default=None)
parser.add_argument('--num_classes', type=int, default=24,
                    help='an integer for the accumulator')
parser.add_argument('--data_dir', type=str, default='/data2/army_challenge/training_data/',
                    help='an integer for the accumulator')
parser.add_argument('--data_file', type=str, default=None,
                    help='an integer for the accumulator')
parser.add_argument('--data_format', type=str, default="channels_last",
                    help='an integer for the accumulator')
parser.add_argument('--mode', type=str, default='test')
args = parser.parse_args()

EPS = args.eps#1.e-15
CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
 'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
 'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
 'QAM32', 'QAM64', 'QPSK']

mods = np.array([1,9,10,11,12,13])
AMmods = np.array([4,5])
QAMmods = np.array([6,7,20,21,22])
PSKmods = np.array([0,3])
QAMPSKmods = np.concatenate([QAMmods, PSKmods])
BASEDIR = args.train_dir
DATABASE = args.data_dir
if args.model is None:
    m_path = [BASEDIR+'morad_classifier1.h5']
elif os.path.isdir(args.model):
    m_path = find_files(args.model, pattern="model*.h5")
    print(m_path)
else:
    m_path = [args.model]
if args.submodel is None:
    s_path = None
elif os.path.isdir(args.submodel):
    s_path = find_files(args.submodel, pattern="model*.h5")
    print(s_path)
else:
    s_path = [args.submodel]
if args.qammodel is None:
    q_path = None
elif os.path.isdir(args.qammodel):
    q_path = find_files(args.qammodel, pattern="model*.h5")
    print(q_path)
else:
    q_path = [args.qammodel]

if args.qampskmodel is None:
    qp_path = None
elif os.path.isdir(args.qampskmodel):
    qp_path = find_files(args.qampskmodel, pattern="model*.h5")
    print(qp_path)
else:
    qp_path = [args.qampskmodel]

if args.pskmodel is None:
    p_path = None
elif os.path.isdir(args.pskmodel):
    p_path = find_files(args.pskmodel, pattern="model*.h5")
    print(p_path)
else:
    p_path = [args.pskmodel]
output_path = BASEDIR+"TestSet{}Predictions.csv".format(args.test)

if args.mode == 'test':
    if args.data_file is None:
        data_file = DATABASE + "training_data_chunk_14.pkl"
    else:
        data_file = args.data_file
    testdata = LoadModRecData(data_file, 0., 0., 1.)
else:
    f = open(args.data_file, 'rb')
    testdata = pickle.load(f, encoding='latin1')
    testdata = np.stack([testdata[i] for i in range(1, len(testdata.keys())+1)], axis=0)
#model = load_model(m_path)
#submodel = load_model(s_path) if s_path is not None else None
def _init(queue):
    global idx
    idx = queue.get()

def _get_prediction(path, test_X):
    global idx
    print(idx, "Model: ", path)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    with tf.device('/gpu:{}'.format(0)):
        model = load_model(path)
        pred = model.predict(test_X)
    return pred
def _print_id(num):
    global idx
    print(idx, num)
    sleep(5)
def ens_predictions(paths, test_X):
    if paths is None:
        return None
    if args.ngpu>1:
        ids = range(min(len(m_path), args.ngpu))
        manager = Manager()
        idQueue = manager.Queue()
        for i in ids:
            idQueue.put(i)
        p = Pool(args.ngpu, _init, (idQueue,))
        #_ = p.map(_print_id, np.arange(20))
        preds = p.map(partial(_get_prediction, test_X=test_X), paths)
        p.close()
        p.join()
    else:
        preds = []
        for mp in paths:
            print("Model: "+mp)
            model = load_model(mp)
            preds.append(model.predict(test_X))
        
    preds = np.stack(preds, axis=0)
    preds = np.mean(preds, axis=0)
    return preds

def get_logloss(test_Y_i_hat, test_Y_i, EPS, round_to=None):
    if round_to is not None:
        test_Y_i_hat = np.around(test_Y_i_hat, decimals=round_to)
    test_Y_i_hat = np.where(test_Y_i_hat>EPS, test_Y_i_hat, EPS)
    test_Y_i_hat = np.where(test_Y_i_hat<1-EPS, test_Y_i_hat, 1-EPS)
    test_Y_i_hat /= np.sum(test_Y_i_hat, axis=1, keepdims=True)
    logloss = - np.sum(test_Y_i*np.log(test_Y_i_hat))/test_Y_i.shape[0]
    return logloss

def hiarch_update(Y_i_hat, sub_hat, mods, weight=1.):
    sub_sum = np.sum(Y_i_hat[mods])
    Y_i_hat[mods] = weight * sub_sum * sub_hat + (1-weight)*Y_i_hat[mods]
    k = int(np.argmax(Y_i_hat))
    return Y_i_hat, k

if args.mode == 'test':
    acc = {}
    snrs = np.arange(-15,15, 5)

    classes = testdata.modTypes
    print("classes ", classes)
    for snr in testdata.snrValues:

        # extract classes @ SNR
        snrThreshold_lower = snr
        snrThreshold_upper = snr+5
        if args.all_snr: 
            print('evaluating all snr')
            snrThreshold_upper += 20
        snr_bounded_test_indicies = testdata.get_indicies_withSNRthrehsold(testdata.test_idx, snrThreshold_lower, snrThreshold_upper)
    
        test_X_i = testdata.signalData[snr_bounded_test_indicies]
        test_Y_i = testdata.oneHotLabels[snr_bounded_test_indicies]    

        # estimate classes
        test_Y_i_hat = ens_predictions(m_path, test_X_i)#model.predict(test_X_i) # shape (batch, nmods)
        sub_Y_i_hat = ens_predictions(s_path, test_X_i)#submodel.predict(test_X_i)
        qampsk_Y_i_hat = ens_predictions(qp_path, test_X_i)
        qam_Y_i_hat = ens_predictions(q_path, test_X_i)#submodel.predict(test_X_i)
        psk_Y_i_hat = ens_predictions(p_path, test_X_i)#submodel.predict(test_X_i)
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])

        sublist = []
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            if s_path is not None and k in mods:
                test_Y_i_hat[i], k = hiarch_update(test_Y_i_hat[i], sub_Y_i_hat[i], mods)
            elif qp_path is not None and k in QAMPSKmods:
                test_Y_i_hat[i], k = hiarch_update(test_Y_i_hat[i], qampsk_Y_i_hat[i], QAMPSKmods)
            elif q_path is not None and k in QAMmods:
                test_Y_i_hat[i], k = hiarch_update(test_Y_i_hat[i], qam_Y_i_hat[i], QAMmods)
            elif p_path is not None and k in PSKmods:
                test_Y_i_hat[i], k = hiarch_update(test_Y_i_hat[i], psk_Y_i_hat[i], PSKmods)
            elif k in AMmods and False:
                sub_sum = np.sum(test_Y_i_hat[i,AMmods])
                sub_hat = 0.5*np.ones_like(test_Y_i_hat[i,AMmods])
                test_Y_i_hat[i,AMmods] = sub_sum * sub_hat
                k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
        for eps in [EPS]:
            logloss = get_logloss(test_Y_i_hat.copy(), test_Y_i, eps)
        for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        #plt.figure(figsize=(10,10))
        #plot_confusion_matrix(confnorm, labels=classes, title="hiearchical (SNR=%d)"%(snr))
        #plt.savefig(BASEDIR+"ConfusionMatrixSNR=%d"%(snr))
    
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("SNR", snr, "Accuracy: ", cor / (cor+ncor), "LogLoss", logloss, "Out of", len(snr_bounded_test_indicies))
        if args.all_snr:
            break
        acc[snr] = 1.0*cor/(cor+ncor)


else:
    preds = ens_predictions(m_path,testdata) 
    preds[:,19] = 0 #set PI4QPSK to 0
    if args.test == 1:
        preds[:,11]=0  #set the FM to 0
        preds[:,12]=0
    subpreds = ens_predictions(s_path,testdata) 
    qampskpreds = ens_predictions(qp_path,testdata)
    qampreds = ens_predictions(q_path,testdata) 
    pskpreds = ens_predictions(p_path,testdata) 
    for i in range(0,preds.shape[0]):
        k = int(np.argmax(preds[i,:]))
        if s_path is not None and k in mods:
            preds[i], k = hiarch_update(preds[i], subpreds[i], mods, args.subweight)
        elif qp_path is not None and k in QAMPSKmods:
            preds[i], k = hiarch_update(preds[i], qampskpreds[i], QAMPSKmods, args.qampskweight)
        elif q_path is not None and k in QAMmods:
            preds[i], k = hiarch_update(preds[i], qampreds[i], QAMmods, args.qamweight)
        elif p_path is not None and k in PSKmods:
            preds[i], k = hiarch_update(preds[i], pskpreds[i], PSKmods, args.pskweight)
    preds = np.where(preds>EPS, preds, EPS)
    preds = np.where(preds<1-EPS, preds, 1-EPS)
    preds[:,19] = 0 #set PI4QPSK to 0
    if args.test == 1:
        preds[:,11]=0  #set the FM to 0
        preds[:,12]=0
    preds /= np.sum(preds, axis=1, keepdims=True)
    # save with 15 decimals
    fmt = '%1.0f' + preds.shape[1] * ',%1.15f'
    id_col = np.arange(1, testdata.shape[0] + 1)
    preds = np.insert(preds, 0, id_col, axis = 1)
    
    header = "Index,"
    for i in range(len(CLASSES) - 1):
        header += CLASSES[i]+','
    header += CLASSES[-1]
    f = open(output_path, 'w')
    f.write(header+'\n')
    f.close()
    f = open(output_path,'ab')
    np.savetxt(f, preds, delimiter=',', fmt = fmt)
print("Done")
