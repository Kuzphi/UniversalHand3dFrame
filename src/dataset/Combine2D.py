# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import os
import cv2
import json
import pickle
import torch
import numpy as np
from easydict import EasyDict as edict

from src.dataset.twoDimension import *
from src.dataset.BaseDataset import JointsDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image
from src.core.evaluate import get_preds_from_heatmap


class Combine2D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(Combine2D, self).__init__(cfg)
        self.datasets = []
        self.len = 0
        for key in cfg.CONTAINS:
            # print (key)
            cfg.CONTAINS[key].HEATMAP = cfg.HEATMAP
            cfg.CONTAINS[key].IS_TRAIN = cfg.IS_TRAIN
            cfg.CONTAINS[key].TRANSFORMS = cfg.TRANSFORMS
            cfg.CONTAINS[key].NUM_JOINTS = cfg.NUM_JOINTS
            self.datasets.append( eval(key)(cfg.CONTAINS[key]))
            self.len += len(self.datasets[-1])
    def __len__(self):
        # return 100
        return self.len

    def _get_db(self):
        pass

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)

    def eval_result(self, outputs, batch, cfg = None):
        preds = get_preds_from_heatmap(outputs['heatmap'][-1])
        # preds = get_preds_from_heatmap(batch['heatmap'])
        # print (preds[0][:5,:], batch['coor'][0][:5])
        diff = batch['coor'] - preds
        dis = torch.norm(diff, dim = -1)
        PcK_Acc = (dis < self.cfg.THR).float().mean()
        return {"dis": dis.mean(), "PcKAcc":PcK_Acc}

    def get_preds(self, outputs, batch):
        return get_preds_from_heatmap(outputs['heatmap'][-1])
        
    def post_infer(self, cfg, pred):
        # print (self[0]['coor'] - pred[0])
        dist = np.array([torch.norm(self[i]['coor'] - pred[i], dim = -1).mean() for i in range(len(self))])
        print (dist.mean())
        median = np.median(dist)
        x, y = AUC(dist)
        auc = calc_auc(dist)
        auc00_50 = calc_auc(dist,  0, 50)
        auc20_50 = calc_auc(dist, 20, 50)
        auc30_50 = calc_auc(dist, 30, 50)
        print('AUC: ', auc)
        print('AUC  0 - 50: ', auc00_50)
        print('AUC 20 - 50: ', auc20_50)
        print('AUC 30 - 50: ', auc30_50)
        print('median:', median)
        import matplotlib.pyplot as plt
        fig = plt.figure('AUC')
        plt.plot(x, y)
        fig.savefig(os.path.join(cfg.CHECKPOINT,'AUC.png'))
        res = {
            'x':x,
            'y':y,
            'AUC':auc,
            'AUC00_50': auc00_50,
            'AUC30_50': auc30_50,
        }
        pickle.dump(res, open(os.path.join(cfg.CHECKPOINT,'dist.pickle'),'w'))
        