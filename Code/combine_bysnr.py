from score import *
import argparse
parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--csv_path1', type=str, default='./anom/test2_1262gp0.csv',
                    help='Name and path of csv prediction file.')
parser.add_argument('--csv_path2', type=str, default='./anom/test2_1121gp0.csv',
                    help='Name and path of csv prediction file.')
parser.add_argument('--snr_path', type=str, nargs='+',
                    help='Name and path of data file that has the labels.')
parser.add_argument('--out_path', type=str, default='./anom/TestSet2Predictions.csv',
                    help='Name and path of data file that has the labels.')
args = parser.parse_args()

snr = np.concatenate([np.load(path) for path in args.snr_path], axis=0)
#snr = np.argmax(snr, axis=1)

snr_inds = [np.argwhere(snr==S).squeeze() for S in range(6)]
pred1 = get_pred(args.csv_path1)
pred2 = get_pred(args.csv_path2)
lambs = [0.75,0.75,0.85,0.9,0.95,0.95]


res = np.empty(pred1.shape)
for i,lmbda in enumerate(lambs):
    preds1 = pred1[snr_inds[i]]
    preds2 = pred2[snr_inds[i]]
    res[snr_inds[i]] = lmbda * preds1 + (1-lmbda) * preds2


    
fmt = '%1.0f' + res.shape[1] * ',%1.8f'
id_col = np.arange(1, res.shape[0] + 1)
preds = np.insert(res, 0, id_col, axis = 1)    
    
    
    
CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
    'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
    'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
    'QAM32', 'QAM64', 'QPSK']

header = "Index,"
for i in range(len(CLASSES) - 1):
    header += CLASSES[i]+','
header += CLASSES[-1] + '\n'

output_path = args.out_path

f=open(output_path, 'w')
f.write(header)
f.close()
f=open(output_path,'ab')
np.savetxt(f, preds, delimiter=',', fmt=fmt) 






