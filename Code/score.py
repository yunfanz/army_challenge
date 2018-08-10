import numpy as np
from data_loader import *
import os
from utils import find_files
import argparse


def load_label(data_file):

#    csv_file = args.csv_file
#    data_file = args.data_file

    data = LoadModRecData(data_file, 1, 0, 0)
    labels = data.oneHotLabels

    # make sure all PI4QPSK is not scored
    for i in range(labels.shape[0]):
        labels[i][19] = 0
    return labels

def load_pred(csv_file):    
    with open(csv_file) as f:
        f.readline() # get rid of header
        preds = f.read()


    # split numbers
    preds = str.replace(preds, '\n', ',')[:-1] # get rid of \n at the end of file
    preds = preds.split(',')

    # turn into float np array
    preds = np.array([float(dat) for dat in preds])

    # reshape and remove index column
    preds = preds.reshape((-1,25))
    preds = preds[:,1:]
    return preds

def evaluate(preds, labels):
    # avoid log extremes
    x,y = preds.shape
    for i in range(x):
        for j in range(y):
            preds[i,j] = max(1e-15, min(preds[i,j], 1-1e-15))

    # rescale preds they sum to 1
    for i, row in enumerate(preds):
        preds[i] = row / np.sum(row)

    preds = np.log(preds)

    preds = preds*labels#[:preds.shape[0]]

    sample_size = (preds.shape[0]*23./24)  #exclude PIQPSK

    logloss = -np.sum(preds) / sample_size
    score = 100/(1+logloss)
    
    return logloss, score

def get_pred(csv_path):
    if os.path.isdir(csv_path):
        csv_files = find_files(csv_path, pattern="*.csv")
        csv_files = sorted(csv_files)
    else:
        csv_files = [csv_path]
    print(csv_files)
    preds = np.concatenate([load_pred(fname) for fname in csv_files], axis=0)
    print(preds.shape)
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process')
    parser.add_argument('--csv_file', type=str, default=None,
                    help='Name and path of csv prediction file.')
    parser.add_argument('--data_file', type=str, default=None,
                    help='Name and path of data file that has the labels.')
    args = parser.parse_args()
    
    if os.path.isdir(args.csv_file):
        csv_files = find_files(args.csv_file, pattern="*.csv")
        csv_files = sorted(csv_files)
    else:
        csv_files = [args.csv_file]
    print(csv_files)
    preds = np.concatenate([load_pred(fname) for fname in csv_files], axis=0)
    print(preds.shape)
    labels = load_label(args.data_file)
    logloss, score = evaluate(preds, labels)
    print("Logloss:", logloss)
    print("Score:", score)
    print()
