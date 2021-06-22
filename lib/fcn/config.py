import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.RNG_SEED = 3
__C.MODE = 'TRAIN'
__C.gpu_id = 0

__C.TRAIN = edict()
__C.TRAIN.CLASSES = (0,8,14,21)
__C.TRAIN.VERTEX_REG = False

# Scales to compute real features
__C.TRAIN.SCALES_BASE = (1.0,)

__C.TRAIN.MAX_ITERS_PER_EPOCH = 1000000
__C.TRAIN.IMS_PER_BATCH = 2

# parameters for data augmentation
__C.TRAIN.CHROMATIC = True
__C.TRAIN.ADD_NOISE = False

__C.TEST = edict()
__C.TEST.CLASSES = (0,8,14,21)
__C.TEST.VISUALIZE = False
__C.TEST.IMS_PER_BATCH = 2