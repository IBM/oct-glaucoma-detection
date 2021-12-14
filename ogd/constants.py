"""
Global constants
"""


class Hyperparams(object):
    BN = 1  # Batchnormalization
    ORDER = -1  # Polynom order for layer flattening
    ALGO = 2  # Algorithm 0:RMSProp, 1:Adam, 2:Nadam, 3:Nesterov+Momentum
    LR = 4  # Learning rate 1e-x
    FLIPEYE = 0  # Flip OD to OS
    MIXUP = 0  # Mix up batch augmentation, alpha = MIXUP/10.0
    N_FILTER = [32,32,32,32,32]  # Number of filter banks per layer
    N_CONV = [7,5,3,3,3]  # Size of convolutional filter per layer
    N_STRIDE = [2, 1, 1, 1, 1]  # Stride per layer
    P_AUG = [0,0,0,0]  # Augmentation: flip, trans, occ, rot


hp = Hyperparams()

# LABELS = ['SLOW', 'FAST']
# LABELS = ['Normal', 'GS', 'POAG']
LABELS = ['Normal', 'POAG']

N_FOLDS = 1
N_EPOCHS = 3  # 40
C, H, W = 64, 128, 64
H_TOP, H_BOTTOM = 0.25, 0.6
N_CLASSES = len(LABELS)
BATCH_SIZE = 8
N_TRAIN = 200  # 200  # Max number training samples
CLEARCACHE = True
USE_PRETRAINED = False
SAVE_BEST = True
DEVICE = "1"  # 0,1,... for GPU or -1 for CPU

# MODE = 'Default'
# MODE = 'Enface'
# MODE = 'Stitched'
MODE = '3D'

if MODE == 'Default':
    INPUTSHAPE = (H, W, C)
elif MODE == 'Stitched':
    INPUTSHAPE = (C * 8, W * 16, 1)
elif MODE == 'Enface':
    INPUTSHAPE = (C, W, H)
elif MODE == '3D':
    INPUTSHAPE = (H, W, C, 1)
