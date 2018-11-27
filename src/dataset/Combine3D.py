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

from src.dataset.threeDimension import *
from src.dataset.BaseDataset import JointsDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image
from src.core.evaluate import get_preds_from_heatmap


class Combine3D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(Combine3D, self).__init__(cfg)
        self.datasets = []
        for key in cfg.CONTAINS:
            cfg.CONTAINS[key].HEATMAP = cfg.HEATMAP
            cfg.CONTAINS[key].IS_TRAIN = cfg.IS_TRAIN
            cfg.CONTAINS[key].TRANSFORMS = cfg.TRANSFORMS
            self.datasets.append( eval('%{}2D(cfg.%{})'.format(key, key) ))
            self.len += len(self.datasets[-1])
    def __len__(self):
        return self.len

    def _get_db(self):
        pass

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)

    def eval_result(self, outputs, batch, cfg = None):
        gt_coor = batch['coor']
        # print(outputs['pose3d'].size(), batch['index_bone_length'].size())
        pred_coor = outputs['pose3d'] * batch['index_bone_length'].view(-1,1,1).repeat(1,21,3)
        dis = torch.norm(gt_coor - pred_coor, dim = -1)
        
        dis = torch.mean(dis)
        # AUC = calc_auc(dis, 0, 50)
        # median = torch.median(dis)
        return {"dis": dis}
        # return {"dis": dis, 'median':median, 'AUC':AUC }

    def get_preds(self, outputs, batch):
        return outputs['pose3d'] * batch['index_bone_length'].view(-1,1,1).repeat(1,21,3)

    def post_infer(self, cfg, pred):
        # print (self[0]['coor'] - pred[0])
        dist = np.array([torch.norm(self[i]['coor'] - pred[i], dim = -1).mean() for i in range(len(self))])
        median = np.median(dist)
        x, y = AUC(dist)
        auc = calc_auc(dist)
        auc00_30 = calc_auc(dist,  0, 50)
        auc30_50 = calc_auc(dist, 30, 50)
        print('AUC: ', auc)
        print('AUC  0 - 30: ', auc)
        print('AUC 30 - 50: ', auc)
        print('median:', median)
        import matplotlib.pyplot as plt
        fig = plt.figure('AUC')
        plt.plot(x, y)
        fig.savefig(os.path.join(cfg.CHECKPOINT,'AUC.png'))
        