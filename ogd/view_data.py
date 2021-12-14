"""
View bscans in cubes
"""
import numpy as np
import nutsml.imageutil as ni

from nutsflow import *
from nutsml import *
from common import *
from preprocessing import flatten_layers


@nut_processor
def ReadCubeScans(samples):
    for sample in samples:
        filepath, label, uid = sample
        cube = np.load(filepath)
        for image in cube:
            flat_img, mask = flatten_layers(image, verbose=True)
            yield ni.gray2rgb(image), flat_img, mask


def view_cubes():
    (read_samples() >> Print() >>
     ReadCubeEnface() >> Prefetch() >> FlattenCol((0, 1)) >>
     #ReadCubeStitched() >> PrintColType() >>
     ViewImageAnnotation(0, 1, pause=0.01) >> Consume())


def view_flattening():
    #path = CFG.nyu_dx_onh_cubes
    path = CFG.nyu_dx_onh_cubes_large

    filter = Filter(lambda s: 'POAG' in s[0])
    (read_dx_samples(path) >> filter >> Print() >> ReadCubeScans() >>
     Prefetch() >> ViewImage((0, 1, 2), pause=0.01) >> Consume())


if __name__ == '__main__':
    view_cubes()
    #view_flattening()
