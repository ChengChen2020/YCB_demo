#!/usr/bin/env python3

import torch

import time, os, sys
import os.path as osp
import numpy as np
import argparse
import pprint

import _init_paths
from datasets.factory import get_dataset
from fcn.config import cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--dataset_background', dest='dataset_background_name',
                        help='background dataset to train on',
                        default='background_nvidia', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    print('GPU device {:d}'.format(args.gpu_id))

	# prepare dataset
    if cfg.TEST.VISUALIZE:
        shuffle = True
        np.random.seed()
    else:
        shuffle = False

    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)
    print(dataset)
    worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=shuffle,
    	num_workers=0, worker_init_fn=worker_init_fn)

    print(iter(dataloader).next()['label'].shape)

