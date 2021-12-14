"""
View class activation maps.
"""
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import os.path as osp
import nutsml.imageutil as ni
import numpy as np
import preprocessing as pp

from common import *
from constants import LABELS, MODE, hp
from network import create_network


@nut_function
def GetActivations(batch, network, layername='CAM', istrain=0):
    import keras.backend as K
    model = network.model
    f = K.function([model.input, K.learning_phase()],
                   [model.get_layer(layername).output])
    batch = batch[0]
    return batch, f([batch, istrain])[0]


def get_weights(network, layername='CWGT'):
    model = network.model
    return model.get_layer(layername).get_weights()[0]


def overlay(img, cam):
    """Return gray-scale image and image overlayed with cam"""
    img = ni.gray2rgb(img)
    overlay = img.copy()
    overlay[:, :, 0] = overlay[:, :, 0] + cam
    return img, overlay


def to_rgb(cube):
    cube = cube[..., np.newaxis]
    return np.concatenate(3 * (cube,), axis=-1)


@nut_processor
def CAM(batches, class_wgts, classid, hard=False):
    for cube_batch, act_batch in batches:
        w = class_wgts[:, classid]
        cam_batch = np.matmul(act_batch, w)
        cam_batch[cam_batch < 0.3 * np.max(cam_batch)] = 0  # ignore weak
        if hard: cam_batch[cam_batch > 0] = 100
        for cube, cam in zip(cube_batch, cam_batch):
            h, w = cube.shape[:2]
            img = np.sum(cube, axis=2)  # Enface image from cube
            img = ni.rerange(img, np.min(img), np.max(img), 0, 155, 'uint8')
            cam = ni.rerange(cam, np.min(cam), np.max(cam), 0, 100, 'uint8')
            cam = ni.resize(cam, w, h, order=1)
            img_gray, overlay = overlay(img, cam)
            yield img_gray, overlay, img, cam


@nut_processor
def CAM3D(batches, class_wgts, classid, hard=False):
    for cube_batch, act_batch in batches:
        w = class_wgts[:, classid]
        cam_batch = np.matmul(act_batch, w)
        cam_batch[cam_batch < 0.3 * np.max(cam_batch)] = 0  # ignore weak
        if hard: cam_batch[cam_batch > 0] = 100
        for cube, cam in zip(cube_batch, cam_batch):
            cube = np.squeeze(cube)
            cube = ni.rerange(cube, np.min(cube), np.max(cube), 0, 155, 'uint8')
            cam = ni.rerange(cam, np.min(cam), np.max(cam), 0, 100, 'uint8')

            # enface cube resized to equal dimensions
            shape = (128, 128, 128)
            cam1 = pp.resize_cube(cam, shape)
            cube1 = pp.resize_cube(cube, shape)

            idxs = (1, 0, 2)  # side view
            cube2 = np.transpose(cube1, idxs)
            cam2 = np.transpose(cam1, idxs)

            views = zip(cube1, cam1, cube2, cam2)
            for i, (im1, ca1, im2, ca2) in enumerate(views):
                im1, over1 = overlay(im1, ca1)
                over1[i, :, :] = (155, 155, 0)
                im2, over2 = overlay(im2, ca2)
                over2[i, :, :] = (155, 155, 0)
                yield im1, over1, im2, over2


@nut_processor
def ComputeCam(batches, class_wgts, classid, hard=True):
    for cube_batch, act_batch in batches:
        w = class_wgts[:, classid]
        cam_batch = np.matmul(act_batch, w)
        cam_batch[cam_batch < 0.3 * np.max(cam_batch)] = 0  # ignore weak
        if hard: cam_batch[cam_batch > 0] = 100
        for cam in cam_batch:
            yield cam


@nut_function
def OverlayCam(sample):
    cube, cam = sample
    cube = np.squeeze(cube)
    cube = ni.rerange(cube, np.min(cube), np.max(cube), 0, 155, 'uint8')
    cam = ni.rerange(cam, np.min(cam), np.max(cam), 0, 100, 'uint8')

    cam = pp.resize_cube(cam, cube.shape)
    cube = to_rgb(cube)
    cube[:, :, :, 0] = cube[:, :, :, 0] + cam

    return cube


@nut_processor
def WriteCam(cams, samples, outpath):
    for cam, sample in zip(cams, samples):
        filename = osp.basename(sample[0])
        filepath = osp.join(outpath, filename)
        np.save(filepath, cam, allow_pickle=False)
        yield filepath


def view_cam(dx):
    network = create_network('best_weights.h5')
    network.load_weights()
    wgts = get_weights(network)

    classid = LABELS.index(dx)
    filter_disease = Filter(lambda s: dx in s[0])

    (read_samples() >> filter_disease >> read_cube >>
     build_pred_batch >> GetActivations(network) >> CAM3D(wgts, classid) >>
     # ViewImage((1,), layout=(1, 1), pause=1000) >> Consume())    # Stitched
     ViewImage((1, 3), layout=(1, 2), pause=0.01) >> Consume())


def write_camcubes(n, dx, inpath, outpath):
    network = create_network('best_weights.h5')
    network.print_layers()
    network.load_weights()
    wgts = get_weights(network)
    classid = LABELS.index(dx)

    filter_dx = Filter(lambda s: dx in s[0])
    samples = read_dx_samples(inpath) >> filter_dx >> Head(n)
    cubes1, cubes2 = samples >> read_cube >> flatten_cube >> Tee(2)
    cams = (cubes1 >> resize_cube >> make_enface >> build_pred_batch >>
            GetActivations(network) >> ComputeCam(wgts, classid))
    (cubes2 >> make_enface >> PrintProgress(samples) >> Get(0) >> Zip(cams) >>
     OverlayCam() >> WriteCam(samples, outpath) >> Consume())


if __name__ == "__main__":
    # view_cam('POAG')
    n_normal = 1
    n_disease = 10

    inpath = CFG.nyu_dx_onh_cubes_large
    outpath = r"c:\Maet\Data\NYU\cam_browser\new"
    write_camcubes(n_disease, 'POAG', inpath, outpath)
    write_camcubes(n_normal, 'Normal', inpath, outpath)
