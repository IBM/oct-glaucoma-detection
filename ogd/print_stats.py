"""
Print some data statistics
"""

import os.path as osp

from nutsflow import *
from nutsml import *
from common import *


def uid2sid(uid):
    return '-'.join(uid.split('-')[:-1])

def dx(sample):
    fname = osp.basename(sample[0])
    return fname.split('-')[0]


def print_stats():
    samples = read_samples() >> Collect()
    print('Example samples:')
    samples >> Take(3) >> Print() >> Consume()
    print('Stats--------------------------------------------------------------')
    print('#samples', samples >> Count())
    print('#patients', samples >> Get(2) >> Map(uid2sid) >> Dedupe() >> Count())
    print('dist diagnosis', samples >> Map(dx) >> CountValues())


if __name__ == '__main__':
    print_stats()
