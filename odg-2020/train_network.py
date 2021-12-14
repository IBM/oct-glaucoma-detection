"""
Network training
"""
from __future__ import print_function

import sys
import numpy as np
import sklearn.metrics as sm

from common import *
from nutsflow.common import StableRandom
from keras import backend as K
from network import create_network
from keras.metrics import categorical_accuracy
from datetime import datetime


def compute_aucs(network, samples, cache=None):
    y_pred = (samples >> read_cube >> flatten_cube >> resize_cube >>
              make_enface >> If(cache, cache) >> build_pred_batch >>
              network.predict() >> Collect())
    aucs = {}
    idxs = range(len(LABELS))
    ijs = [(i, j) for i in idxs for j in idxs if i > j]
    for i, j in ijs:
        inclass = Filter(lambda s: s[0][1] in {i, j})
        fsamples, fpreds = zip(samples, y_pred) >> inclass >> Unzip()
        y_true = [1 if s[1] == i else 0 for s in fsamples]
        y_score = [p[i] for p in fpreds]
        name = LABELS[j] + '-' + LABELS[i]
        aucs[name] = sm.roc_auc_score(y_true, y_score)
    return aucs

    
def prepare_cache():
    if CLEARCACHE:
        cachetrain.clear()
        cacheval.clear()
        cachetest.clear()
    else:
        print('CACHE NOT CLEARED!')


def train(fold):
    has_params = len(sys.argv) > 1
    print('SYS.ARGV', sys.argv)
    if has_params:
        params = [int(a) for a in sys.argv[1:]]
        print('PARAMS', params)
        hp.BN, hp.ORDER, hp.ALGO, hp.LR, hp.FLIPEYE, hp.MIXUP = params[:6]
        hp.P_AUG = params[6:10]
        n, layer_params = len(params[10:]) // 3, params[10:]
        hp.N_FILTER, hp.N_CONV, hp.N_STRIDE = (layer_params[0:n],
                                               layer_params[n:2 * n],
                                               layer_params[2 * n:3 * n])
                                               
    network = create_network('best_weights.h5')
    network.print_layers()
    if USE_PRETRAINED:
        print('LOADING PRETRAINED NET!')
        network.load_weights()

    do_augment = sum(hp.P_AUG) > 0
    rand = StableRandom(fold)
    same_sid = lambda s: s[2].split('-')[0]
    split_data = SplitRandom((0.8, 0.1, 0.1), constraint=same_sid, rand=rand)
    train_samples, val_samples, test_samples = \
        read_samples() >> split_data >> Collect()
    train_samples = train_samples >> Head(N_TRAIN)    
    traindist = train_samples >> CountValues(1)
    
    print('#samples', len(train_samples), len(val_samples), len(test_samples))
    print('train-dist', traindist)
    print('val-dist', val_samples >> CountValues(1))
    print('test-dist', test_samples >> CountValues(1))

    best_val_auc = 0    
    for epoch in range(N_EPOCHS):
        print('EPOCH:', epoch)

        # build_batch returns [epoch0, epoch1...]
        # where epochN is [[numpy],[numpy]]
        # network train expects [numpy, numpy]
        # this is a hack to remove the extrta []
        def epch_hack(ep):
            return (ep[0][0], ep[1][0]) 

        t_loss, t_acc = (train_samples >> PrintProgress(train_samples) >>
                         read_cube >> flatten_cube >> resize_cube >>
                         make_enface >> Stratify(1, traindist) >>
                         build_batch >> Map(epoch_hack) >> network.train() >> Unzip())

        t_loss_mean = t_loss >> Mean()
        t_acc_mean = t_acc >> Mean()
        train_time = datetime.now() - start_time
        print("train loss : {:.5f}".format(t_loss_mean))
        print("train acc  : {:.5f}".format(t_acc_mean))
        print("train time : {:.1f} min".format(train_time.seconds / 60.0))

        aucs = compute_aucs(network, val_samples, cacheval)
        v_auc = aucs.values() >> Mean()
        print("val auc    : {:.5f}".format(v_auc))
        for name, auc in aucs.items():
            print('  val', name, round(auc, 3))

        plot_epoch((t_acc_mean, v_auc))
        plot_aucs([v_auc] + list(aucs.values()))

        v_auc_mean = aucs.values() >> Mean()
        best_val_auc = max(best_val_auc, v_auc_mean)
        if SAVE_BEST:
            network.save_best(v_auc_mean, isloss=False)

    print('test -----------------------------------------')
    network.load_weights()
    aucs = compute_aucs(network, test_samples, cachetest)
    for name, auc in aucs.items():
        print('  test', name, round(auc, 3))
    test_auc = round(aucs.values() >> Mean(), 3)
    best_val_auc = round(best_val_auc, 3)
    print('test     auc', test_auc)
    print('best val auc', best_val_auc)

    if has_params:
        f = open('hyperparams6.csv', 'a')
        pad = [0] * (6 - len(hp.N_FILTER))
        n_filter = hp.N_FILTER + pad
        n_conv = hp.N_CONV + pad
        n_stride = hp.N_STRIDE + pad
        results = ', '.join(map(str, [test_auc, best_val_auc] +
                                [hp.BN, hp.ORDER, hp.ALGO, hp.LR, hp.FLIPEYE, hp.MIXUP] +
                                hp.P_AUG + n_filter + n_conv + n_stride))
        print('results', results)
        f.write(results + '\n')
        f.close()
    return best_val_auc, test_auc


if __name__ == "__main__":
    val_aucs, test_aucs = [], []
    for fold in range(N_FOLDS):
        print('FOLD', fold+1, 'of', N_FOLDS)
        val_auc, test_auc = train(fold)
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)
    print('***', N_FOLDS, 'fold cross-validation')    
    print('*** mean val  AUC %.3f %.3f' % (val_aucs >> MeanStd()))
    print('*** mean test AUC %.3f %.3f' % (test_aucs >> MeanStd()))

