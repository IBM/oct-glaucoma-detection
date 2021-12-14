"""
Network architecture and loss function
"""

from math import pow
from keras.regularizers import l1
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.layers.normalization import BatchNormalization
from constants import INPUTSHAPE, N_CLASSES, hp

from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import (Dense, Dropout, Activation, Flatten,
                          GlobalAveragePooling2D, GlobalAveragePooling3D)
from keras.layers import Conv3D, Convolution2D, MaxPooling2D, MaxPooling3D
from keras.layers.core import Lambda
from keras import backend as K

from nutsml import KerasNetwork


def create_optimizer():
    lr = 1/pow(10, hp.LR)
    optimizers = {
        0: RMSprop(lr=lr),
        1: Adam(lr=lr),
        2: Nadam(lr=lr),
        3: SGD(lr=lr, momentum=0.9, nesterov=True)
    }
    return optimizers[hp.ALGO]


def ___create_network(weightspath):
    model = Sequential()
    # model.add(Dropout(0.7, input_shape=INPUTSHAPE));
    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=INPUTSHAPE))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu', name='CAM'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(N_CLASSES, name='CWGT'))
    # model.add(Dropout(0.5))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=1e-4)
    # optimizer = SGD(lr=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return KerasNetwork(model, weightspath)


def ___create_network(weightspath):
    print('create_network: MOST ARCHITECTURE PARAMS IGNORED', hp.BN)

    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), padding='same', input_shape=INPUTSHAPE))
    if hp.BN: model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv3D(64, (3, 3, 3)))
    if hp.BN: model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3)))
    if hp.BN: model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, (3, 3, 3)))
    if hp.BN: model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu', name='CAM'))
    model.add(GlobalAveragePooling3D())
    model.add(Dense(N_CLASSES, name='CWGT'))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return KerasNetwork(model, weightspath)


def create_network(weightspath):
    print('CREATE NETWORK:', hp.BN, hp.ORDER, hp.FLIPEYE, hp.MIXUP,
            hp.N_FILTER, hp.N_CONV, hp.N_STRIDE)
    model = Sequential()

    cam_i = len(hp.N_FILTER) - 1
    params = zip(hp.N_FILTER, hp.N_CONV, hp.N_STRIDE)
    for i, (n_filter, n_conv, n_stride) in enumerate(params):
        if i == 0:
            model.add(Conv3D(n_filter, n_conv, strides=n_stride,
                             padding='same', input_shape=INPUTSHAPE))
        else:
            model.add(Conv3D(n_filter, n_conv, strides=n_stride,
                             padding='same'))
        if hp.BN:
            model.add(BatchNormalization(axis=-1))
        name = 'CAM' if i == cam_i else 'layer' + str(i)
        model.add(Activation('relu', name=name))

    model.add(GlobalAveragePooling3D())
    model.add(Dense(N_CLASSES, name='CWGT'))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return KerasNetwork(model, weightspath)


if __name__ == '__main__':
    network = create_network('best_weights.h5')
    network.model.summary()