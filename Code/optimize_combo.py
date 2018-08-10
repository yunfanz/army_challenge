from score import *

csv_path1 = './anom/train0_large_model0'
csv_path2 = './anom/train0_1121_model0'

snr = np.load('/datax/yzhang/test_data/train0_snr.npy')
snr_inds = [np.argwhere(snr==S).squeeze() for S in range(6)]
print(snr_inds[0].shape)
pred1 = get_pred(csv_path1)
pred2 = get_pred(csv_path2)
labels = load_label('/datax/yzhang/training_data/training_data_chunk_0.pkl')
lambs = np.ones(6)

for S in range(6):
    inds = snr_inds[S]
    print()
    print(S,':')
    for L in np.arange(0,1.05,0.05):
        logloss, score = evaluate(pred1[inds]*L+pred2[inds]*(1-L), labels[inds])
        print(L, logloss, score)
#print(snr.shape, pred1.shape, pred2.shape)
