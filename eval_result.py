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

import pickle 
import numpy as np
from src import model
from src import dataset
from src.utils.misc import get_config, save_infer_result
from src.core.evaluate import AUC, calc_auc
def main(args):
    print("Reading configuration file")
    cfg = get_config(args.cfg, type = 'train')

    print("Loading Training Data")
    infer_data = eval('dataset.' + cfg.VALID.DATASET.NAME)(cfg.VALID.DATASET)

    preds_path = 'output/Mar  8 19:37:07_train_Weakly_direct_regression_Combine2D(RHD2D)_valid_Combine2D(RHD2D)/best/preds.pickle'

    preds = pickle.load(open(preds_path))['pose3d']
    print (preds.shape)
    if os.path.exists('data/RHD/gt3d.torch'):
        gt = torch.load('data/RHD/gt3d.torch')
    else:
        gt = torch.zeros(preds.shape)
        for idx in range(len(infer_data)):
            print(idx)
            gt[idx] = infer_data[idx]['coor3d']
        torch.save(gt, 'data/RHD/gt3d.torch')

    dist = torch.norm(gt - preds, dim = -1).view(-1) * 1000
    # print ()
    auc = calc_auc(dist)
    auc00_50 = calc_auc(dist,  0, 50)
    auc20_50 = calc_auc(dist, 20, 50)
    auc30_50 = calc_auc(dist, 30, 50)
    print('AUC: ', auc)
    print('AUC  0 - 50: ', auc00_50)
    print('AUC 20 - 50: ', auc20_50)
    print('AUC 30 - 50: ', auc30_50)
    print('average: ', dist.mean())
    print('median:', dist.median())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='configure file',
                        type=str)
    args = parser.parse_args()
    main(args)