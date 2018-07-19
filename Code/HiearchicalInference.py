import numpy as np
from data_loader import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure

import keras
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Dense
from keras.models import Model, load_model
import argparse

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_dir', type=str, default='./log/model_9/',
                    help='an integer for the accumulator')
parser.add_argument('--all_snr', type=bool, default=False)
parser.add_argument('--load_weights', type=bool, default=False)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--model', type=int, default=2,
                    help='an integer for the accumulator')
parser.add_argument('--submodel', type=str, default=None,
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
parser.add_argument('--save_csv', type=bool, default=False)
args = parser.parse_args()

EPS = 1.e-15
CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
 'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
 'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
 'QAM32', 'QAM64', 'QPSK']

mods = np.array([1,9,10,11,12,13])
AMmods = np.array([4,5])
BASEDIR = '/datax/yzhang/models/'
DATABASE = '/datax/yzhang/training_data/'
m_path = BASEDIR+'morad_classifier1.h5'
if args.submodel is None:
    s_path = BASEDIR+'sub_classifier1.h5'
else:
    s_path = args.submodel
output_path = BASEDIR+"TestSet1Predictions.csv"

data_file = DATABASE + "training_data_chunk_14.pkl"
testdata = LoadModRecData(data_file, 0., 0., 1.)

model = load_model(m_path)
submodel = load_model(s_path)

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
    test_Y_i_hat = model.predict(test_X_i) # shape (batch, nmods)
    sub_Y_i_hat = submodel.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])

    sublist = []
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        if k in mods:
            sub_sum = np.sum(test_Y_i_hat[i,mods])
            sub_hat = sub_Y_i_hat[i]
            test_Y_i_hat[i,mods] = sub_sum * sub_hat
            k = int(np.argmax(test_Y_i_hat[i,:]))
        elif k in AMmods:
            sub_sum = np.sum(test_Y_i_hat[i,AMmods])
            sub_hat = 0.5*np.ones_like(test_Y_i_hat[i,AMmods])
            test_Y_i_hat[i,AMmods] = sub_sum * sub_hat
            k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    test_Y_i_hat = np.where(test_Y_i_hat>EPS, test_Y_i_hat, EPS)
    test_Y_i_hat = np.where(test_Y_i_hat<1-EPS, test_Y_i_hat, 1-EPS)
    test_Y_i_hat /= np.sum(test_Y_i_hat, axis=1, keepdims=True)
    logloss = - np.sum(test_Y_i*np.log(test_Y_i_hat))/test_Y_i.shape[0]
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(confnorm, labels=classes, title="hiearchical (SNR=%d)"%(snr))
    plt.savefig(BASEDIR+"ConfusionMatrixSNR=%d"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("SNR", snr, "Accuracy: ", cor / (cor+ncor), "LogLoss", logloss, "Out of", len(snr_bounded_test_indicies))
    if args.all_snr:
        break
    acc[snr] = 1.0*cor/(cor+ncor)


if args.save_csv:
    # save with 4 decimals
    fmt = '%1.0f' + preds.shape[1] * ',%1.4f'
    id_col = np.arange(1, dataset.shape[0] + 1)
    preds = np.insert(preds, 0, id_col, axis = 1)
    
    header = "Index,"
    for i in range(len(CLASSES) - 1):
        header += CLASSES[i]+','
    header += CLASSES[-1]
    f = open(output_path, 'w')
    f.write(header)
    f.close()
    f = open(output_path,'ab')
    np.savetxt(f, preds, delimiter=',', fmt = fmt)
print("Done")
