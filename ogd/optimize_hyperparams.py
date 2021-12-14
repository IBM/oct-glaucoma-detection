"""
Randomly pick hyper parameters and run training
"""

import traceback

from constants import hp, N_EPOCHS, N_TRAIN
from subprocess import check_call
from numpy import prod
from random import choice


C_NL = [4, 5]  # number of layers
C_FS = [16, 32, 64] #[8, 16, 32, 64, 96, 128]  # Filter bank sizes
C_CS = [1, 3, 5, 7] #[1, 3, 5, 7, 9]  # Convolution size
C_NS = [1, 2] #[1, 2, 3]  # Convolution stride
C_OS = [-1, 0, 1, 2] #[-1, 0, 1, 2]  # Order for flattening
C_BN = [1] #[0, 1]  # BatchNorm on/off
C_AS = [2] #[0, 1, 2, 3]  # Training algorithm
C_LS = [4] #[5, 4, 3, 2, 1]  # Learning rates 1e-x
C_FE = [0] # Flip eye
C_MU = [0] #[0, 2, 5, 8, 10]  # Mixup, alpha = C_MU/10.0
C_PA = [0] # [0,1] # Augmentation


def execute(params):
    cmdline = ['python', 'train_network.py'] + [str(p) for p in params]
    try:
        print('CMDLINE', cmdline)
        check_call(cmdline)
    except:
        traceback.print_exc()

        
def run_random():   
    assert N_EPOCHS == 50
    assert N_TRAIN == 2000

    while True:
        n_layers = choice(C_NL)
        n_filters = [choice(C_FS) for _ in range(n_layers)]
        n_conv = [choice(C_CS) for _ in range(n_layers)]
        n_stride = [choice(C_NS) for _ in range(n_layers)]
        order = choice(C_OS)
        bn = choice(C_BN)
        algo = choice(C_AS)
        lr = choice(C_LS)
        fe = choice(C_FE)
        mu = choice(C_MU)
        paug = [choice(C_PA) for _ in range(4)]
        if prod(n_stride) > 4:
            continue
        params = [bn, order, algo, lr, fe, mu] + paug + n_filters + n_conv + n_stride
        execute(params)

        
def run_selected():
    assert N_EPOCHS == 50
    assert N_TRAIN == 2000
        
    trials = [
      # [bn, order, algo, lr, eyeflip, mixup] + [flip, trans, occ, rot] + filters + conv + stride            
      [1,-1,2,4,0,0] + [0,0,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],
      [1,0,2,4,0,0]  + [0,0,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],
      [1,1,2,4,0,0]  + [0,0,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],
      [1,2,2,4,0,0]  + [0,0,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],

      [1,-1,2,4,0,0]  + [0,0,0,1] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],
      [1,-1,2,4,0,0]  + [0,0,1,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],
      [1,-1,2,4,0,0]  + [0,1,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],
      [1,-1,2,4,0,0]  + [1,0,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],

      [1,-1,2,4,1,0]  + [0,0,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],
      [1,-1,2,4,0,1]  + [0,0,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],
      [1,-1,2,4,0,5]  + [0,0,0,0] + [16,32,32,32,64] + [7,3,3,3,7] + [2,1,1,1,1],

      [1,-1,2,4,0,0]  + [0,0,0,0] + [64,32,32,32,16] + [7,3,3,3,5] + [1,2,1,1,2],

      [1,-1,2,4,0,0]  + [0,0,0,0] + [32,64,32,16] + [9,5,3,3] + [2,1,1,1],
    ]

    for i, params in enumerate(trials):
        print('TRIAL:', i+1, 'of', len(trials), '*'*80)
        print('params:', params)
        execute(params)



if __name__ == "__main__":
    #run_random()
    run_selected()