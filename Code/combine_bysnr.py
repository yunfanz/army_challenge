from score import *

csv_path1 = '/home/yzhang/Projects/army_challenge/Code/anom/test2_1262gp0.csv'
csv_path2 = '/home/yzhang/Projects/army_challenge/Code/anom/test2_1121gp0.csv'
#csv_path2 = './anom/train0_1121_model0'

# snr = np.load('/datax/yzhang/test_data/train0_snr.npy')
snr = np.load('/datax/yzhang/test_data/snr_preds2.npy')
snr = np.argmax(snr, axis=1)

snr_inds = [np.argwhere(snr==S).squeeze() for S in range(6)]
pred1 = get_pred(csv_path1)
pred2 = get_pred(csv_path2)
#labels = load_label('/datax/yzhang/training_data/training_data_chunk_0.pkl')
lambs = [0.75,0.75,0.85,0.9,0.95,0.95]

#for S in range(6):
#    inds = snr_inds[S]
#    print()
#    print(S,':')
#    for L in np.arange(0,1.05,0.05):
#        logloss, score = evaluate(pred1[inds]*L+pred2[inds]*(1-L), labels[inds])
#        print(L, logloss, score)
#print(snr.shape, pred1.shape, pred2.shape)


res = np.empty(pred1.shape)
for i,lmbda in enumerate(lambs):
    preds1 = pred1[snr_inds[i]]
    preds2 = pred2[snr_inds[i]]
    res[snr_inds[i]] = lmbda * preds1 + (1-lmbda) * preds2


    
fmt = '%1.0f' + res.shape[1] * ',%1.15f'
id_col = np.arange(1, res.shape[0] + 1)
preds = np.insert(res, 0, id_col, axis = 1)    
    
    
    
CLASSES = ['16PSK', '2FSK_5KHz', '2FSK_75KHz', '8PSK', 'AM_DSB', 'AM_SSB', 'APSK16_c34',
    'APSK32_c34', 'BPSK', 'CPFSK_5KHz', 'CPFSK_75KHz', 'FM_NB', 'FM_WB',
    'GFSK_5KHz', 'GFSK_75KHz', 'GMSK', 'MSK', 'NOISE', 'OQPSK', 'PI4QPSK', 'QAM16',
    'QAM32', 'QAM64', 'QPSK']

header = "ID,"
for i in range(len(CLASSES) - 1):
    header += CLASSES[i]+','
header += CLASSES[-1] + '\n'

output_path = "TestSet2Predictions.csv"

f=open(output_path, 'w')
f.write(header)
f.close()
f=open(output_path,'ab')
np.savetxt(f, preds, delimiter=',', fmt=fmt) 






