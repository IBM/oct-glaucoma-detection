"""
Preprocessing functions.
"""

import numpy as np
import scipy as sp
import nutsml.imageutil as ni

from nutsflow import *
from constants import C, H, W, H_TOP, H_BOTTOM


def flatten_layers(image, order=2, verbose=False):
    """
    :param ndarray image: gray scale image.
    :param int order: Order of polynom fitted [0..n].
    :param bool verbose: True: return additional data,
                         False: return image with flattend layers
    :return: Image with height h2 * 2 and flattened, centered layer
    :rtype: ndarray of shape (h2*2, w)
    """
   # build mask with bright pixels (layers)
    threshold = np.mean(image) * 1.5
    mask = (image > threshold).astype(int)
    nrows, ncols = mask.shape
    idxs = np.tile(np.array([range(nrows)]).transpose(), (1, ncols))
    idxs = np.multiply(mask, idxs)

    # get coordinates of vertical median of bright pixels
    ys = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, idxs)
    ys[np.isnan(ys)] = 0
    xs = np.array(range(ncols))

    # fit polynom of degree order but ignore center (fovea, optic cup)
    d = mask.shape[1] // 3  # ignore 1/3 image width center for polynom fit
    rxs = np.hstack([xs[0:d], xs[ncols - d:ncols]])
    rys = np.hstack([ys[0:d], ys[ncols - d:ncols]])
    f = np.poly1d(np.polyfit(rxs, rys, order))

    # copy image into vertically padded image
    r,c = image.shape
    padded = np.zeros((r*2,c))
    offset = r//4
    padded[offset:offset+r] = image

    # extract flattened, vertically centered stripe from padded image
    flat_img = []
    ht, hb = int(r*H_TOP), int(r*H_BOTTOM)
    for x in xs:
        y = offset + int(f(x))
        stripe = padded[y - ht:y + hb, x]
        stripe = np.pad(stripe, (ht+hb-stripe.shape[0], 0), 'constant')
        flat_img.append(stripe)
    flat_img = np.transpose(np.vstack(flat_img))
    flat_img = ni.resize(flat_img, c, r, order=1)

    if verbose:
        flat_img = ni.gray2rgb(flat_img)
        fit_img = ni.gray2rgb(mask).astype('uint8') * 255
        for x in range(fit_img.shape[1]):
            y = int(f(x))
            fit_img[y-2:y+2, x] = (255, 100, 0)

    return (flat_img, fit_img) if verbose else flat_img


def flatten_cube(cube, order):
    return np.stack(flatten_layers(bscan, order) for bscan in cube)

    
def resize_cube(cube, shape):
    """Return resized cube with the define shape"""
    zoom = [float(x) / y for x, y in zip(shape, cube.shape)]
    resized = sp.ndimage.zoom(cube, zoom)
    assert resized.shape == shape
    return resized    


