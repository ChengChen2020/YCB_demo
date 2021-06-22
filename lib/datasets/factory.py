__sets = {}

import datasets.ycb_video
import numpy as np

# ycb video dataset
for split in ['train', 'val', 'keyframe', 'trainval', 'debug']:
    name = 'ycb_video_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split: datasets.YCBVideo(split))

def get_dataset(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_datasets():
    """List all registered imdbs."""
    return __sets.keys()