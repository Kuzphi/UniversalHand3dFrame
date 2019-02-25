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
from src.core.evaluate import calc_auc, AUC

class Combine3D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(Combine3D, self).__init__(cfg)
        self.datasets = []
        self.len = 0
        for key in cfg.CONTAINS:
            print(key)
            cfg.CONTAINS[key].IS_TRAIN = cfg.IS_TRAIN
            cfg.CONTAINS[key].TRANSFORMS = cfg.TRANSFORMS
            cfg.CONTAINS[key].NUM_JOINTS = cfg.NUM_JOINTS
            self.datasets.append( eval(key)(cfg.CONTAINS[key]))
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
        # print (self[0]['coor'].shape)
        dist = np.array([torch.norm(self[i]['coor'] - pred[i], dim = -1).mean() for i in range(len(self))])
        dist = dist * 1000
        median = np.median(dist)
        auc = calc_auc(dist)
        auc00_50 = calc_auc(dist,  0, 50)
        auc30_50 = calc_auc(dist, 30, 50)
        print('AUC: ', auc)
        print('AUC  0 - 50: ', auc00_50)
        print('AUC 30 - 50: ', auc30_50)
        print('median:', median)
        # x,y = AUC(dist)
        # import matplotlib.pyplot as plt
        # fig = plt.figure('AUC')
        # plt.plot(x, y)
        # fig.show();
        # fig.savefig(os.path.join('~/Frame','AUC.png'))
        