"""
Visualize hyper parameter settings and effect on classification accuracy.
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


from nutsflow import *
from tabulate import tabulate

N = 8  ## Padding rows to length n

@nut_processor
def ChunkStats(chunks, n):
    for chunk in chunks:
        if len(chunk) != n:
            continue
        for i in range(n):
            assert chunk[0][2:] == chunk[i][2:], 'Wrong number of folds!'


        t_mean, t_std = chunk >> Get(0) >> MeanStd()
        v_mean, v_std = chunk >> Get(1) >> MeanStd()
        yield [t_mean, v_mean, t_std, v_std] + chunk[0][2:]


def load_data(filepath, n_folds):
    header, data = [], []
    with open(filepath) as f:
        line = next(f).strip()
        header = [e for e in line.split(',')]
        for line in f:
            row = [float(e) for e in line.split(',')]
            row += [0.0] * (N - len(row))
            data.append(row)

    if n_folds:
        data = data >> Chunk(n_folds, list) >> ChunkStats(n_folds) >> Collect()
        header = header[:2] + ['t_std', 'v_std'] + header[2:]
    data.sort(key=lambda r: r[0], reverse=True)
    return header, data


def print_data(header, data, n=100):
    table = tabulate(data >> Head(n), header)
    print(table)


def print_median(header, data, n=9):
    X = np.array(data)[:n]
    M = [np.median(X, axis=0)]
    print(tabulate(X, header))
    print()
    print(tabulate(M, header))


def show_heatmap(header, data):
    X = np.array(data)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    X = np.transpose(X)
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Experiments')
    ax.imshow(X, cmap='Blues', interpolation='nearest', aspect='auto')
    ax.yaxis.set_ticks(np.arange(0, len(header), 1.0))
    ax.set_yticklabels(header)


def show_correlations(header, data):
    X = np.array(data)
    R, pval = ss.spearmanr(X)
    R = np.abs(R)
    R[R < 0.3] = 0
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('|Spearman| > 0.3')
    ax.imshow(R, cmap='hot', interpolation='nearest')
    ticks = np.arange(0, len(header), 1.0)
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(header)
    ax.xaxis.set_ticks(ticks)
    ax.set_xticklabels(header, rotation='vertical')


def main():
    #header, data = load_data('hyperparams7.csv', n_folds=3)
    #header, data = load_data('hyperparams8.csv', n_folds=5)
    #header, data = load_data('hyperparams9.csv', n_folds=5)
    header, data = load_data('hyperparams10.csv', n_folds=5)

    #print_data(header, data)
    print_median(header, data)
    #show_heatmap(header, data)
    #show_correlations(header, data)
    #plt.show()


if __name__ == '__main__':
    main()

