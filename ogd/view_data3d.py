"""
View data cubes in 3d
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from common import read_cube, read_samples


def view():
    paths = read_samples()
    path, label, uid = paths[3]
    cube = np.load(path)
    print(cube.shape)
    xs, ys, zs = np.where(cube > 130)
    print(len(xs))

    ax = plt.axes(projection='3d')
    ax.scatter(xs, ys, zs, c=(0, 0, 0), marker='.', alpha=0.3, linewidth=0.0)
    plt.show()


if __name__ == '__main__':
    view()
