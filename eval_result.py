# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(Kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
import src.core.loss as loss

import numpy as np
from src import model
from src import dataset
from src.core import validate
from src.utils.misc import get_config, save_infer_result

def main(args):
    print("Reading configuration file")
    cfg = get_config(args.cfg, type = 'infer')

    print("Loading Training Data")
    infer_data = eval('dataset.' + cfg.DATASET.NAME)(cfg.DATASET)

    preds_path = 'output/Jan 30 13:55:50_train_ICCV17_Combine3D/best/preds.npy'
    preds = np.load(preds_path)
    print (preds.shape)
    infer_data.post_infer(cfg, torch.tensor(preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='configure file',
                        type=str)
    args = parser.parse_args()
    main(args)