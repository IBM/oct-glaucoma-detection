"""
t-sne plot of embedding to potentially discover Glaucoma subtypes.
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from common import *
from constants import LABELS, MODE, hp
from network import create_network
from sklearn.manifold import TSNE


@nut_function
def GetActivations(batch, network, layername='CAM', istrain=0):
    import keras.backend as K
    model = network.model
    f = K.function([model.input, K.learning_phase()],
                   [model.get_layer(layername).output])
    batch = batch[0]
    return batch, f([batch, istrain])[0]


@nut_processor
def ToVectors(act_batches):
    for act_batch in act_batches:
        for act in act_batch[1]:
            yield act.flatten()


def compute_activations(n):
    network = create_network('best_weights.h5')
    network.load_weights()

    samples = read_samples() >> Shuffle(10000) >> Head(n)
    avs = (samples >> PrintProgress(samples) >> read_cube >>
           flatten_cube >> resize_cube >> make_enface >>
           build_pred_batch >> GetActivations(network) >>
           ToVectors() >> Collect())
    labels = samples >> Get(1) >> Collect()
    return labels, avs


def show_embedding(n):
    print('computing activations')
    labels, avs = compute_activations(n)
    colors = ['r' if l == 1 else 'g' for l in labels]

    print('running TSNE')
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    pts = model.fit_transform(np.stack(avs))

    print('plotting')
    fig = plt.figure(figsize=(14, 7))
    plt_pts = fig.add_subplot(121)
    sc = plt_pts.scatter(pts[:, 0], pts[:, 1], facecolor=colors, marker='.',
                         alpha=0.5, picker=5)
    plt_pts.set_yticklabels([])
    plt_pts.set_xticklabels([])
    plt_pts.tick_params(direction='in')
    plt.show()


if __name__ == '__main__':
    show_embedding(1000)
