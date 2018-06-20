'''
DISTRIBUTION STATEMENT F: Further dissemination only as directed by Army Rapid Capabilities Office, or higher DoD authority.

    Data Rights Notice

        This software was produced for the U. S. Government under Basic Contract No. W15P7T-13-C-A802, and is subject to the Rights in Noncommercial Computer Software and Noncommercial Computer Software Documentation Clause 252.227-7014 (FEB 2012)
        Copyright 2017 The MITRE Corporation. All Rights Reserved.

    :author: Bill Urrego - wurrego@mitre.org
    :date: 12/11/17

'''
from license import DATA_RIGHTS
from version import __version__


__author__ = "Bill Urrego - wurrego@mtire.org"
__license__ = DATA_RIGHTS
__version__ = __version__
__last_modified__ = '12/11/17'

""" INCLUDES"""
import os, sys, time
import numpy as np
from data_loader import LoadModRecData


def parse_args():
    ''' This simply parses the command line arguments

        .. note::
            Usage: python example.py <number of epochs> <batch size> <train_data_set>

    '''

    TAG = '[Parse Args] - '
    # Input Arguments <number of epochs> <batch_size>
    if len(sys.argv) != 4:
        print(TAG + 'Invalid number of arguments. \n Usage: python example.py <number of epochs> <batch size> <train_data_set>')
        exit()

    print(TAG + 'Current Working Directory: ' + os.getcwd())

    print(TAG + 'Number of epochs to run:' + sys.argv[1])
    if not sys.argv[1].isdigit():
        print(TAG + 'Number of epochs is not a number. Exiting...')
        exit()

    print(TAG + 'Batch Size: ' + sys.argv[2])
    if not sys.argv[2].isdigit():
        print(TAG + 'Batch Size is not a number. Exiting...')
        exit()

    print(TAG + 'Training Data file to load: ' + sys.argv[3])
    if not os.path.exists(sys.argv[3]):
        print(TAG + 'Training Data File Not Found.. Exiting...')
        exit()

    return int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def example_1( data, number_of_epochs,  train_batch_size ):
    ''' Example 1

            This example shows how to use :func:`data_loader.LoadModRecData.batch_iter` method
    '''

    # number of classes
    number_of_classes = len(data.modTypes)

    # Tensor input shape
    input_shape = [data.instance_shape[0], data.instance_shape[1], 1]

    # generate batch iterator
    train_batches = data.batch_iter(data.train_idx, train_batch_size, number_of_epochs, use_shuffle=True)

    # iterate for ( ( number_of_examples / batch_size ) + 1 ) x number_of_ephocs
    for batch in train_batches:

        # get the batch
        train_batch_x, train_batch_y, train_batch_y_labels = zip(*batch)

        # determine batch size
        batch_size = len(train_batch_x)

        # determine min and max SNR of this test batch
        min_SNR = min(np.asarray(train_batch_y_labels)[:,1].astype(np.int))
        max_SNR = max(np.asarray(train_batch_y_labels)[:,1].astype(np.int))

        # reshape the training batch to a 4D Tensor
        train_batch_x = np.reshape(train_batch_x, [batch_size, input_shape[0], input_shape[1], input_shape[2]])
        train_batch_y = np.reshape(train_batch_y, [batch_size, number_of_classes])

        # do stuff ...


def example_2(data, number_of_epochs, batch_size):
    ''' Example 2

            This example shows how to use :func:`data_loader.LoadModRecData.get_indicies_withSNRthrehsold`
    '''

    # number of classes
    number_of_classes = len(data.modTypes)

    # Tensor input shape
    input_shape = [data.instance_shape[0], data.instance_shape[1], 1]

    snrThreshold_lower = -5
    snrThreshold_upper = 20
    snr_bounded_test_indicies = data.get_indicies_withSNRthrehsold(data.test_idx, snrThreshold_lower, snrThreshold_upper)


    # generate batch iterator
    test_batches = data.batch_iter(snr_bounded_test_indicies, batch_size, number_of_epochs, use_shuffle=False)

    # iterate for ( ( number_of_examples / batch_size ) + 1 ) x number_of_ephocs
    for batch in test_batches:

        # get the batch
        test_batch_x, test_batch_y, test_batch_y_labels = zip(*batch)

        # determine batch size
        batch_size = len(test_batch_x)

        # determine min and max SNR of this test batch
        min_SNR = min(np.asarray(test_batch_y_labels)[:, 1].astype(np.int))
        max_SNR = max(np.asarray(test_batch_y_labels)[:, 1].astype(np.int))

        # reshape the training batch to a 4D Tensor
        test_batch_x = np.reshape(test_batch_x, [batch_size, input_shape[0], input_shape[1], input_shape[2]])
        test_batch_y = np.reshape(test_batch_y, [batch_size, number_of_classes])


def main(argv):
    ''' a simple application that leverages the :class:`data_loader.LoadModRecData class` '''

    TAG = '[Main Example] - '

    # Training Parameters
    number_of_epochs,  batch_size, train_data_file_path = parse_args()

    ''' Load Dataset '''
    # start timing
    tick = time.time()
    print ('\n' + TAG + 'Loading Mod Rec Dataset...\n')

    # Load data
    data = LoadModRecData(train_data_file_path, .9, .06, .04)

    # stop timing
    tock = time.time()
    print (TAG + 'Loading Mod Rec Dataset took: ' + str(tock-tick) + ' seconds.')


    ''' Run Example 1 '''
    # start timing
    tick = time.time()

    # Run example 1
    example_1(data,number_of_epochs, batch_size)

    # stop timing
    tock = time.time()
    print (TAG + 'Running Example 1 took: ' + str(tock-tick) + ' seconds.')


    ''' Run Example 2 '''
    # start timing
    tick = time.time()

    # Run example 2
    example_2(data, number_of_epochs, batch_size)

    # stop timing
    tock = time.time()
    print (TAG + 'Running Example 2 took: ' + str(tock-tick) + ' seconds.')



if __name__ == "__main__":
    main(0)