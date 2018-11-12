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

from src.dataset.BaseDataset import JointsDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image

from .SHP import SHP
from .G import G
class G_SHP(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(G_SHP, self).__init__(cfg)
        self.shp = SHP(cfg.SHP)
        self.G = G(cfg.G)

    def __len__(self):
        # return 100
        return len(self.shp) + len(self.G)

    def _get_db(self):
        pass

    def __getitem__(self, idx):
        if idx < len(self.shp):
            return self.shp[idx]
        idx -= len(self.shp)
        return self.G[idx]

    def eval_result(self, outputs, batch, cfg = None):
        gt_coor = batch['coor']
        # print(outputs['pose3d'].size(), batch['index_bone_length'].size())
        pred_coor = outputs['pose3d'] * batch['index_bone_length'].view(-1,1,1).repeat(1,21,3)

        dis = torch.norm(gt_coor - pred_coor, dim = -1)

        dis = torch.mean(dis)        
        return {"dis": dis}

    def get_preds(self, outputs, batch):
        return outputs['pose3d'] * batch['index_bone_length'].view(-1,1,1).repeat(1,21,3)