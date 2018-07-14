import matplotlib.pyplot as plt
from data_loader import *
from scipy.signal import *
import numpy as np

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def augment(x, data_format='channels_last'):
    if data_format == 'channels_last':
        x_complex = x[:,:,0] + x[:,:,1]*1.j
    else:
        x_complex = x[:,0] + x[:,1]*1.j
    phs = np.random.random(x_complex.shape[0])
    x_complex *= np.exp(2*np.pi*phs*1.j)[:,np.newaxis]
    if data_format == 'channels_last':
        x_complex = np.stack([x_complex.real, x_complex.imag], axis=2)
    else:
        x_complex = np.stack([x_complex.real, x_complex.imag], axis=1)    
    return x_complex

def cwt_ricker(x, widths):
    return cwt(x, ricker, widths)
cwt_ricker = np.vectorize(cwt_ricker, signature='(m),(n)->(n,m)')
def cwt_morlet(x, widths):
    return cwt(x, morlet, widths)
cwt_morlet = np.vectorize(cwt_morlet, signature='(m),(n)->(n,m)')

def get_wavelet(x, widths = np.arange(1,21)):
    x = x[...,0]+x[...,1]*1.j
    cwtricker_x = cwt_ricker(x,widths)
    cwtmorlet_x = cwt_morlet(x,widths)
    cwts = np.stack([cwtricker_x, cwtmorlet_x], axis=-1)
    cwts = np.transpose(cwts, (0,2,1,3))
    return cwts

def get_data(data_format='channels_last', mode='time_series', load_mods=None, BASEDIR="/home/mshefa/training_data/", files=[0], window='hann', nperseg=256, noverlap=200):
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
#     if not mode.is_instance(list):
#         mode = [mode]
    x_train, y_train, x_val, y_val = [], [], [], []
    for f in files:
        data_file = BASEDIR+"training_data_chunk_0.pkl"
        data = LoadModRecData(data_file, .9, .1, 0., load_mods=load_mods)
    
        x_train.append(data.signalData[data.train_idx])
        y_train.append(data.oneHotLabels[data.train_idx]) 
        x_val.append(data.signalData[data.val_idx])
        y_val.append(data.oneHotLabels[data.val_idx])
    x_train = np.vstack(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.vstack(x_val)
    y_val = np.concatenate(y_val)
    #x_train = augment(x_train)
    if data_format == "channels_last":
        x_train = np.transpose(x_train, (0,2,1))
        x_val = np.transpose(x_val, (0,2,1))
    if mode == 'time_series':
        return x_train, y_train, x_val, y_val
    elif mode == 'fourier':
        return get_fourier(x_train, window=window, nperseg=nperseg, noverlap=noverlap), y_train, get_fourier(x_val, window=window, nperseg=nperseg, noverlap=noverlap), y_val
    elif mode == 'welch':
        return get_welch(x_train, window=window, nperseg=nperseg, noverlap=noverlap), y_train, get_welch(x_val, window=window, nperseg=nperseg, noverlap=noverlap), y_val
    elif mode == 'wavelet':
        return get_wavelet(x_train), y_train, get_wavelet(x_val), y_val
       
    elif mode == "both":
        return x_train, get_fourier(x_train, window=window, nperseg=nperseg, noverlap=noverlap), y_train, x_val, get_fourier(x_val), y_val
    
def get_fourier(cdata, window='hann', nperseg=256, noverlap=220):
    """input must be (batch_size, 1024, 2) real time series"""
    if len(cdata.shape) == 3:
        cdata = cdata[...,0] + cdata[...,1]*1.j
        cdata = cdata.astype(np.complex64)
    batch_size = cdata.shape[0]
    print(cdata.shape)
#     fold = cdata.squeeze().reshape((batch_size, window, 1024//window))
#     if with_hamming:
#         fold *= np.hamming(1024//window)
#     ft = np.fft.fftshift(np.fft.fft(fold, axis=-1))
    fts = []
    for i in range(batch_size):
        _, _, ft = stft(cdata[i], window=window, nperseg=nperseg, noverlap=noverlap)
        #ft = np.fft.fftshift(ft, axes=-1)
        fts.append(ft)
    fts = np.asarray(fts)
    fts = np.stack([fts.real, fts.imag], axis=-1)
    return fts

def get_welch(cdata, window='hann', nperseg=256, noverlap=220, remove_DC=True):
    """input must be (batch_size, 1024, 2) real time series"""
    if len(cdata.shape) == 3:
        cdata = cdata[...,0] + cdata[...,1]*1.j
        cdata = cdata.astype(np.complex64)
    batch_size = cdata.shape[0]
    print(cdata.shape)
    fts = []
    for i in range(batch_size):
        tdata = cdata[i]
        if remove_DC:
            fdata = np.fft.fft(tdata)
            fdata[0] = 0.
            tdata = np.fft.ifft(fdata)
        _,ft = welch(tdata, window=window, nperseg=nperseg, noverlap=noverlap)
        fts.append(ft)
    fts = np.asarray(fts)
    return fts[...,np.newaxis]
