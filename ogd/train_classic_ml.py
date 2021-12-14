"""
Train classic Machine Learning algorithms on handcrafted/traditional features
extracted from ONH OCT scans.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import Random, choice
from tabulate import tabulate
from common import read_samples
from nutsflow import GetCols, Collect, Map, MapCol, Head, PrintProgress, Consume
from nutsml import SplitRandom

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler

FOLDS = 5  # x-fold cross validation
TRIALS = 100  # Trials for hyper-parameter optimization

# ONH features
CLOCKHOURS = ['clockhour%d' % i for i in range(1, 13)]
QUADS = ['quad_%s' % c for c in 'tsni']
DISCCUP = ['avgthickness', 'rimarea', 'discarea', 'avg_cd_ratio',
           'vert_cd_ratio', 'cupvol']
FEATURES = CLOCKHOURS + QUADS + DISCCUP


def read_feature_table():
    path = r'c:\Maet\Data\NYU\longitudinal\cirrus_onh.xlsx'
    df = pd.read_excel(path)
    return df[['uid'] + FEATURES]


def create_samples():
    df = read_feature_table()
    uid2features = {r[1]: r[2:] for r in df.itertuples()}
    samples = (read_samples() >> GetCols((2, 1, 2)) >>
               MapCol(2, uid2features.get) >> Collect())
    return samples


def samples2mats(samples):
    make_vector = Map(lambda s: list(s[2]) + [s[1]])
    M = samples >> make_vector >> Collect(np.vstack)
    X, y = M[:, :-1], M[:, -1].astype(int)
    return X, y


def create_naive_bayes():
    return GaussianNB()


def create_log_reg():
    C = choice(np.logspace(-1, 1, 100))
    penalty = choice(['l1', 'l2'])
    return LogisticRegression(C=C, penalty=penalty,
                              class_weight='balanced')


def create_linear_svm():
    C = choice(np.logspace(-3, 3, 100))
    return SVC(C=C, kernel='linear', class_weight='balanced', probability=True)


def create_poly_svm():
    C = choice(np.logspace(-3, 3, 100))
    degree = choice([2, 3])
    return SVC(C=C, degree=degree, kernel='poly', class_weight='balanced',
               probability=True)


def create_rbf_svm():
    C = choice(np.logspace(-3, 3, 100))
    gamma = choice(np.logspace(-3, 3, 10).tolist() + ['auto'])
    return SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced',
               probability=True)


def create_gboost():
    learning_rate = choice(np.logspace(-3, 0, 100))
    n_estimators = choice([100, 200, 500, 1000])
    max_depth = choice(list(range(2, 10)))
    min_samples_split = choice([2, 4, 6, 8, 10])
    min_samples_leaf = choice([1, 3, 5, 7, 9])
    return GradientBoostingClassifier(learning_rate=learning_rate,
                                      n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf)


def create_randforest():
    max_features = choice(np.linspace(0.1, 1.0, 100))
    n_estimators = choice([10, 50, 100, 500, 1000])
    min_samples_split = choice([2, 4, 6, 8, 10, 20, 40, 60, 100])
    min_samples_leaf = choice([1, 3, 5, 7, 9])
    return RandomForestClassifier(n_estimators=n_estimators,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  max_features=max_features,
                                  class_weight='balanced')


def create_extratree():
    max_features = choice(np.linspace(0.1, 1.0, 100))
    n_estimators = choice([10, 50, 100, 500, 1000])
    min_samples_split = choice([2, 4, 6, 8, 10, 20, 40, 60, 100])
    min_samples_leaf = choice([1, 3, 5, 7, 9])
    return ExtraTreesClassifier(n_estimators=n_estimators,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                max_features=max_features,
                                class_weight='balanced')


CLFS = [
    ('Logistic Regression', create_log_reg),
    ('Naive Bayes', create_naive_bayes),
    ('Random Forest', create_randforest),
    ('Gradient Boosting', create_gboost),
    ('Extra Trees', create_extratree),
    ('SVM (linear)', create_linear_svm),
    ('SVM (poly)', create_poly_svm),
    ('SVM (rbf)', create_rbf_svm),
]


def eval_classifier(clf, X_train, y_train, X_val, y_val):
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_score)


def opt_classifier(create_clf, X_train, y_train, X_val, y_val):
    """Return classifier with best AUC after random hyperparam search"""
    best_clf, best_auc = None, 0
    for _ in range(TRIALS) >> PrintProgress(TRIALS):
        clf = create_clf()
        auc = eval_classifier(clf, X_train, y_train, X_val, y_val)
        if auc > best_auc:
            best_clf, best_auc = clf, auc
    return best_clf, best_auc


def eval_classifiers(X_train, y_train, X_val, y_val, X_test, y_test):
    test_aucs, val_aucs = [], []
    for name, cc in CLFS:
        print('evaluating:', name)
        clf, val_auc = opt_classifier(cc, X_train, y_train, X_val, y_val)
        val_aucs.append(val_auc)
        test_auc = eval_classifier(clf, X_train, y_train, X_test, y_test)
        test_aucs.append(test_auc)
    return test_aucs, val_aucs


def create_fold(samples, k):
    same_sid = lambda s: s[0].split('-')[0]
    split = SplitRandom((0.8, 0.1, 0.1), constraint=same_sid,
                        rand=Random(k))
    train_samples, val_samples, test_samples = samples >> split

    # Samples to matrices
    X_train, y_train = samples2mats(train_samples)
    X_val, y_val = samples2mats(val_samples)
    X_test, y_test = samples2mats(test_samples)

    # normalize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def eval_folds():
    print('reading data...')
    samples = create_samples()
    testmat, valmat = [], []
    for k in range(FOLDS):
        print('\nFold', k + 1, 'of', FOLDS)
        X_train, y_train, X_val, y_val, X_test, y_test = create_fold(samples, k)
        test_aucs, val_aucs = eval_classifiers(X_train, y_train, X_val, y_val,
                                               X_test, y_test)
        testmat.append(test_aucs)
        valmat.append(val_aucs)

    testmat = np.array(testmat)
    tmeans, tstds = np.mean(testmat, axis=0), np.std(testmat, axis=0)
    valmat = np.array(valmat)
    vmeans, vstds = np.mean(valmat, axis=0), np.std(valmat, axis=0)
    names = [n for n, c in CLFS]
    diffs = vmeans - tmeans

    print('\n')
    header = ['Classifier', 'AUC(val)', 'STD(val)', 'AUC(test)', 'STD(test)',
              'val-test']
    data = list(zip(names, vmeans, vstds, tmeans, tstds, diffs))
    data.sort(key=lambda r: r[3], reverse=True)
    table = tabulate(data, header, floatfmt='.3f')
    print(table)

    print('\n\n')
    print(r'Classifier & AUC(val) & AUC(test) & AUC(val-test) \\')
    print(r'\hline')
    for result in data:
        fmtstr = r'{:20} & {:.2f}$\pm${:.3f} & {:.2f}$\pm${:.3f} & {:.3f} \\'
        print(fmtstr.format(*result))


def evaluate_classifier(samples, name, clf):
    aucs = []
    for k in range(FOLDS):
        same_sid = lambda s: s[0].split('-')[0]
        split = SplitRandom((0.9, 0.1), constraint=same_sid, rand=Random(k))
        samples_train, samples_test = samples >> split

        # Samples to matrices
        X_train, y_train = samples2mats(samples_train)
        X_test, y_test = samples2mats(samples_test)

        # normalize data
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # train and eval classifier
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
        aucs.append(auc)

    print(name, round(np.mean(aucs), 2), round(np.std(aucs), 3))


def evaluate_classifiers():
    clfs = [
        # ('Random Forest', RandomForestClassifier()),
        # ('Logistic Regression', LogisticRegression(class_weight='balanced')),
        # ('Naive Bayes', GaussianNB()),
        # ('SVM (linear)', SVC(kernel='linear')),
        ('SVM (poly)', SVC(kernel='poly')),
        # ('SVM (RBF)', SVC(kernel='rbf')),
        # ('GradientBoosting', GradientBoostingClassifier()),
    ]

    samples = create_samples()
    for name, clf in clfs:
        evaluate_classifier(samples, name, clf)


def feature_importances():
    print('reading samples...')
    samples = create_samples()

    print('training...')
    aucs, imps = [], []
    for k in range(FOLDS):
        print('fold:', k, 'of', FOLDS)
        X_train, y_train, X_val, y_val, X_test, y_test = create_fold(samples, k)
        clf, _ = opt_classifier(create_extratree, X_train, y_train, X_val,
                                y_val)
        imps.append(clf.feature_importances_)
        auc = roc_auc_score(y_test, clf.predict(X_test))
        aucs.append(auc)
    print('AUC:', np.mean(aucs), '+/-', np.std(aucs))

    print('plotting...')
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Feature importances')
    y_pos = np.arange(len(FEATURES))
    errs = np.std(imps, axis=0)
    means = np.mean(imps, axis=0)
    ax.barh(y_pos, means, xerr=errs, color="gray", align="center")
    ax.yaxis.set_ticks(y_pos)
    ax.set_yticklabels(FEATURES)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    plt.show()


def view_clusters():
    from sklearn.manifold import TSNE

    print('reading samples...')
    samples = create_samples()
    X, y = samples2mats(samples)
    colors = ['r' if l == 1 else 'g' for l in y]

    print('running TSNE')
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    pts = model.fit_transform(X)

    print('plotting')
    fig = plt.figure(figsize=(14, 7))
    plt_pts = fig.add_subplot(121)
    sc = plt_pts.scatter(pts[:, 0], pts[:, 1], facecolor=colors, marker='.',
                         alpha=0.5, picker=5)
    plt_pts.set_yticklabels([])
    plt_pts.set_xticklabels([])
    plt_pts.tick_params(direction='in')
    plt.show()


def plot_features():
    print('reading samples...')
    samples = create_samples()
    X, y = samples2mats(samples)
    # X = X / X.max(axis=0)   # Normalize cols
    fig, ax = plt.subplots(1, 1)
    ax.boxplot(X)
    ax.set_xticklabels(FEATURES, rotation=90)
    plt.show()


def evaluate_logistic_regression():
    import sklearn.metrics as sm

    print('reading samples...')
    samples = create_samples()

    print('training...')
    X_train, y_train, X_val, y_val,_, _ = create_fold(samples, 0)
    clf, _ = opt_classifier(create_log_reg, X_train, y_train, X_val, y_val)

    metrics = []
    y_score = clf.predict_proba(X_val)[:,1]
    for threshold in np.linspace(0,1,100):
        y_pred = np.where(y_score > threshold, 1, 0)
        f1 = sm.f1_score(y_val, y_pred)
        metrics.append((threshold, f1))
    threshold, f1 = max(metrics, key=lambda t: t[1])
    print('best threshold and F1', threshold, f1)

    print('evaluating...')
    M = []
    for k in range(5):
        _, _, _, _, X_test, y_test = create_fold(samples, k)
        y_score = clf.predict_proba(X_test)[:, 1]
        y_pred = np.where(y_score > threshold, 1, 0)
        auc = sm.roc_auc_score(y_test, y_score)
        mcc = sm.matthews_corrcoef(y_test, y_pred)
        recall = sm.recall_score(y_test, y_pred)
        precision = sm.precision_score(y_test, y_pred)
        f1 = sm.f1_score(y_test, y_pred)
        M.append((auc,mcc,recall,precision, f1))
    M = np.array(M)
    np.set_printoptions(precision=3)
    print(M)
    print('AUC MCC Recall Precision F1')
    print(M.mean(axis=0))
    print(M.std(axis=0))



if __name__ == '__main__':
    #evaluate_logistic_regression()
    # evaluate_classifiers()
    eval_folds()
    # feature_importances()
    # view_clusters()
    # plot_features()
