"""
Evaluate network performance on test set
"""

import numpy as np
import sklearn.metrics as sm

from common import *
from nutsflow.common import StableRandom
from network import create_network


def predict(network, samples):
    y_score = (samples >> read_cube >> flatten_cube >> resize_cube >>
               make_enface >> build_pred_batch >> network.predict() >>
               Get(1) >> Collect())
    return y_score


def read_fold_samples(fold):
    rand = StableRandom(fold)
    same_sid = lambda s: s[2].split('-')[0]
    split_data = SplitRandom((0.8, 0.1, 0.1), constraint=same_sid, rand=rand)
    _, val_samples, test_samples = read_samples() >> split_data >> Collect()
    return val_samples[:20], test_samples[:20]
    # return val_samples, test_samples


def evaluate_fold(network, threshold, fold):
    print('evaluting fold ', fold)
    val_samples, test_samples = read_fold_samples(fold)

    y_true = test_samples >> Get(1) >> Collect()
    y_score = predict(network, test_samples)
    y_pred = y_score >> Map(lambda x: 1 if x > threshold else 0) >> Collect()

    dist = test_samples >> CountValues(1)
    auc = sm.roc_auc_score(y_true, y_score)
    mcc = sm.matthews_corrcoef(y_true, y_pred)
    recall = sm.recall_score(y_true, y_pred)
    precision = sm.precision_score(y_true, y_pred)
    f1 = sm.f1_score(y_true, y_pred)

    return auc, mcc, recall, precision, f1


def evaluate():
    network = create_network('best_weights.h5')
    network.load_weights()

    val_samples, test_samples = read_fold_samples(0)
    metrics = []
    y_score = predict(network, val_samples)
    y_val = val_samples >> Get(1) >> Collect()
    for threshold in np.linspace(0, 1, 100):
        y_pred = y_score >> Map(lambda x: 1 if x > threshold else 0) >> Collect()
        f1 = sm.f1_score(y_val, y_pred)
        metrics.append((threshold, f1))
    threshold,f1 = max(metrics, key=lambda t: t[1])
    print('best threshold and F1', threshold, f1)

    np.set_printoptions(precision=3)
    M = np.array([evaluate_fold(network, threshold, fold) for fold in range(5)])
    print(M)
    print('AUC MCC Recall Precision F1')
    print(M.mean(axis=0))
    print(M.std(axis=0))


if __name__ == '__main__':
    evaluate()
