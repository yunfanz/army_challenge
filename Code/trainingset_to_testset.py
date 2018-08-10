import numpy as np
from data_loader import *
import pickle

import argparse

parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_file', type=str, default='/datax/yzhang/training_data/training_data_chunk_0.pkl',
                    help='Name and path of training file.')
parser.add_argument('--test_file', type=str, default=None,
                    help='Name and path of sudo testfile.')
parser.add_argument('--end_ind', type=int, default=288000)
parser.add_argument('--start_ind', type=int, default=0)   
args = parser.parse_args()



data_file = args.train_file
trainData = LoadModRecData(data_file, 1., 0., 0.)
trainData = trainData.signalData[args.start_ind:args.end_ind]


testDict = {}
for i, sig in enumerate(trainData):
    testDict[i+1] = sig

with open(args.test_file, 'wb') as f:
    pickle.dump(testDict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
