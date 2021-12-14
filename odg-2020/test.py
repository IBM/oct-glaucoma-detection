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
    print('EPOCH:', 0)

    start_time = datetime.now()        

    zz = (train_samples >> PrintProgress(train_samples) >>
          read_cube >> flatten_cube >> resize_cube >>
          make_enface >> Stratify(1, traindist) >>
          build_batch >> Map(f) >> network.train() >> Collect())
    return zz

zz = train(0)

