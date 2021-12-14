"""
Common functions and nuts.
"""
from __future__ import print_function

from constants import *
from glob import glob

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress tensorflow warnings
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

from nutsflow import *
from nutsml import *
import preprocessing as pp
import numpy as np
import os.path as osp
import nutsml.imageutil as ni

CFG = load_config('nutseye.yaml')


def read_dx_samples(path=CFG.nyu_dx_onh_cubes):
    anno = lambda cf: osp.basename(cf).split('-')
    label = lambda cf: anno(cf)[0]
    classid = lambda cf: LABELS.index(label(cf))
    uid = lambda cf: '-'.join(anno(cf)[1:])[:-4]
    cubefiles = glob(path)
    return [(cf, classid(cf), uid(cf)) for cf in cubefiles if label(cf) in LABELS]
    
    
def read_sur_samples(path=CFG.nyu_sur_onh_cubes):
    assert LABELS == ['SLOW', 'FAST']
    anno = lambda cf: osp.basename(cf).split('-')
    #classid = lambda cf: 0 if anno(cf)[0] == '1' else 1
    classid = lambda cf: LABELS.index(anno(cf)[0])
    uid = lambda cf: '-'.join(anno(cf)[2:4])
    cubefiles = glob(path)
    return [(cf, classid(cf), uid(cf)) for cf in cubefiles]    



@nut_function
def ResizeCube(sample, shape=(C, H, W)):
    cube, label = sample
    cube = pp.resize_cube(cube, shape)
    return cube, label    
    
    
@nut_function
def MakeEnFace(sample):
    cube, label = sample
    cube = np.transpose(cube, (1, 0, 2))
    cube = np.expand_dims(cube, 3)
    return cube, label


@nut_function
def FlattenCube(sample, order=hp.ORDER):
    cube, label = sample
    if order != -1:
        cube = pp.flatten_cube(cube, order)
    return cube, label


@nut_function
def ReadCube(sample):
    filepath, label, uid = sample
    cube = np.load(filepath)
    flat_cube = pp.flatten_cube(cube, hp.ORDER)
    return flat_cube, label


@nut_function
def ReadCube3d(sample):
    filepath, label, uid = sample
    cube = np.load(filepath)
    #flat_cube = pp.flatten_cube(cube, hp.ORDER) if hp.ORDER else cube
    #enface_cube = np.transpose(flat_cube, (1, 0, 2))
    #ext_cube = np.expand_dims(enface_cube, 3)
    return cube, label


@nut_function
def ReadCubeStitched(sample):
    filepath, label, uid = sample
    cube = np.load(filepath)
    flat_cube = pp.flatten_cube(cube, hp.ORDER)
    enface_cube = np.transpose(flat_cube, (1, 0, 2))
    rs = range(0, 8)
    rows = [enface_cube[r * 16:(r + 1) * 16] for r in rs]
    stitched = np.hstack(np.concatenate(rows, axis=1))
    # stitched = np.hstack(s for s in enface_cube)
    return stitched, label


@nut_function
def ReadCubeEnface(sample):
    filepath, label, uid = sample
    cube = np.load(filepath)
    flat_cube = pp.flatten_cube(cube, hp.ORDER)
    enface_cube = np.transpose(flat_cube, (1, 0, 2))
    if hp.FLIPEYE and '-OD' in uid:
        enface_cube = np.flip(cube, 2)
    return enface_cube, label


@nut_processor
def AugmentEnfaceCubes(samples):
    idxs = tuple(range(H))
    pflip, ptrans, pocc, prot = hp.P_AUG
    augment = (AugmentImage(idxs)
               .by('identical', 1.0)
               .by('fliplr', pflip)
               .by('translate', ptrans, [-5, 5], [-5, 5])
               .by('occlude', pocc, [0, 1], [0, 1], [0.1, 0.5], [0.1, 0.5])
               .by('rotate', prot, [-10, +10])
               )
    for sample in samples:
        cube, label = sample
        assert cube.shape[0] == H, 'Expect enface cube as input'
        for enfaces in [cube] >> augment:
            yield np.stack(enfaces), label

            
@nut_function
def Mixup(batch):
    alpha = hp.MIXUP / 10.0
    if alpha <= 0:
        return batch

    images = batch[0][0]
    labels = batch[1][0]

    ri = np.arange(len(images))
    np.random.shuffle(ri)
    lam = np.random.beta(alpha, alpha)

    images = lam * images + (1 - lam) * images[ri]
    labels = lam * labels + (1 - lam) * labels[ri]

    return [[images], [labels]]
    
            
read_samples = read_sur_samples if LABELS == ['SLOW', 'FAST'] else read_dx_samples
plot_epoch = PlotLines((0, 1), layout=(2, 1), filepath='plot_epoch.png')
n_aucs = (N_CLASSES * (N_CLASSES - 1)) // 2 + 1
plot_aucs = PlotLines(tuple(range(n_aucs)), layout=(n_aucs, 1),
                      filepath='plot_aucs.png')
cachetrain = Cache('cache_train', clearcache=CLEARCACHE)
cacheval = Cache('cache_val', clearcache=CLEARCACHE)
cachetest = Cache('cache_test', clearcache=CLEARCACHE)

if MODE == 'Default':
    read_cube = ReadCube()
    build_batch = (BuildBatch(BATCH_SIZE)
                   .input(0, 'tensor', 'float32', axes=(1, 2, 0))
                   .output(1, 'one_hot', 'uint8', N_CLASSES))
    build_pred_batch = (BuildBatch(BATCH_SIZE, False)
                        .input(0, 'tensor', 'float32', axes=(1, 2, 0)))
elif MODE == 'Stitched':
    read_cube = ReadCubeStitched()
    build_batch = (BuildBatch(BATCH_SIZE)
                   .input(0, 'image', 'float32')
                   .output(1, 'one_hot', 'uint8', N_CLASSES))
    build_pred_batch = (BuildBatch(BATCH_SIZE, False)
                        .input(0, 'image', 'float32'))
elif MODE == 'Enface':
    read_cube = ReadCubeEnface()
    build_batch = (BuildBatch(BATCH_SIZE)
                   .input(0, 'tensor', 'float32', axes=(1, 2, 0))
                   .output(1, 'one_hot', 'uint8', N_CLASSES))
    build_pred_batch = (BuildBatch(BATCH_SIZE, False)
                        .input(0, 'tensor', 'float32', axes=(1, 2, 0)))
elif MODE == '3D':
    read_cube = ReadCube3d()
    flatten_cube = FlattenCube()
    resize_cube = ResizeCube()
    make_enface = MakeEnFace()
    build_batch = (BuildBatch(BATCH_SIZE)
                   .input(0, 'tensor', 'float32')
                   .output(1, 'one_hot', 'uint8', N_CLASSES))
    build_pred_batch = (BuildBatch(BATCH_SIZE, False)
                        .input(0, 'tensor', 'float32'))
