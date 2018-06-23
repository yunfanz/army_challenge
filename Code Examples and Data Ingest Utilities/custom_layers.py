from keras.layers import Layer
from keras import backend as K
from keras.engine.base_layer import InputSpec
from data_loader import *
class _GlobalPooling1D(Layer):
    """Abstract class for different global pooling 1D layers.
    """

    def __init__(self, **kwargs):
        super(_GlobalPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, inputs):
        raise NotImplementedError
class GlobalAveragePooling1D(_GlobalPooling1D):
    """Global average pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    """

    def call(self, inputs, keepdims=True):
        return K.mean(inputs, axis=1, keepdims=keepdims)
    
    
    
def get_fourier(cdata, window=16, with_hamming=True):
    """input must be (batch_size, 1024, 2) real time series"""
    if len(cdata.shape) == 3:
        cdata = cdata[...,0] + cdata[...,1]*1.j
        cdata = cdata.astype(np.complex64)
    batch_size = cdata.shape[0]
    fold = cdata.squeeze().reshape((batch_size, window, 1024//window))
    if with_hamming:
        fold *= np.hamming(1024//window)
    ft = np.fft.fftshift(np.fft.fft(fold, axis=-1))
    ft = np.stack([ft.real, ft.imag], axis=-1)
    return ft

def get_data(data_format='channel_last', mode='time_series'):
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    BASEDIR = "/home/mshefa/training_data/"
    data_file = BASEDIR+"training_data_chunk_0.pkl"
    data = LoadModRecData(data_file, .9, .1, 0.)
    
    x_train = data.signalData[data.train_idx]
    y_train = data.oneHotLabels[data.train_idx] 
    x_val = data.signalData[data.val_idx]
    y_val = data.oneHotLabels[data.val_idx]
    if data_format == "channel_last":
        x_train = np.transpose(x_train, (0,2,1))
        x_val = np.transpose(x_val, (0,2,1))
    if mode == 'time_series':
        return x_train, y_train, x_val, y_val
    elif mode == 'fourier':
        return get_fourier(x_train), y_train, get_fourier(x_val), y_val
    elif mode == "both":
        return x_train, get_fourier(x_train), y_train, x_val, get_fourier(x_val), y_val


