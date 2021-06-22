from .imdb import imdb
from .ycb_video import YCBVideo

import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

print("root dir -->:", ROOT_DIR)