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
import torch
import numpy as np
from easydict import EasyDict as edict

from src.dataset.BaseDataset import JointsDataset
from src.utils.imutils import im_to_torch, draw_heatmap, load_image, resize
from src.utils.misc import to_torch
from src.core.evaluate import get_preds_from_heatmap

class Tencent2D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(Tencent2D, self).__init__(cfg)

    def _get_db(self):
        return json.load(open(self.cfg.DATA_JSON_PATH))
        
    def transforms(self, cfg, img, coor):
        # resize
        if cfg.has_key('RESIZE'):
            coor[:, 0] = coor[:, 0] / img.size(1) * cfg.RESIZE
            coor[:, 1] = coor[:, 1] / img.size(2) * cfg.RESIZE
            img = resize(img, cfg.RESIZE, cfg.RESIZE)

        if self.is_train:
            # s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            # r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            
            # Flip
            if cfg.FLIP and random.random() <= 0.5:
                img = torch.flip(img, dims = [1])
                coor[:, 0] = img.size(1) - coor[:, 0]

            # Color 
            if cfg.COLOR_NORISE:
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        return img, coor
    def __getitem__(self, idx):
        w = self.db[idx]

        image_path   = os.path.join(self.cfg.ROOT, w[1], w[1] + w[2], 'image', w[0] + '.png')
        label_path = os.path.join(self.cfg.ROOT, w[1], w[1] + w[2], 'label', w[0] + '.json')

        img = load_image(image_path, mode = 'GBR') # C * H * W
        
        label = json.load(open(label_path))

        #calculate ground truth coordination
        coor = np.array(label['perspective'])
        coor[:, 0] = coor[:, 0] * img.size(1)
        coor[:, 1] = (1 - coor[:, 1]) * img.size(2)
        coor = coor[:, :2]

        #apply transforms into image and calculate cooresponding coor
        if self.cfg.TRANSFORMS:
            img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)

        #openpose require 22 channel, discard the last one
        heatmap = np.zeros((self.cfg.NUM_JOINTS, img.shape[1], img.shape[2]))
        for i in range(self.cfg.NUM_JOINTS - 1):
            heatmap[i, :, :] = draw_heatmap(heatmap[i], coor[i], self.cfg.HEATMAP.SIGMA, type = self.cfg.HEATMAP.TYPE) 


        meta = edict({'name': w[1] + ' ' + w[2] + ' ' + w[0]})

        assert coor.min() > 0, label_path
        
        return { 'input':  {'img':img},
                 'coor': to_torch(coor),
                 'heatmap': to_torch(heatmap),
                 'weight': 1,
                 'meta': meta}

    def eval_result(self, outputs, batch, cfg = None):
        if cfg is None:
            cfg = self.cfg
        preds = get_preds_from_heatmap(outputs['heatmap'][-1])
        # preds = get_preds_from_heatmap(batch['heatmap'])
        # print (preds[0][:5,:], batch['coor'][0][:5])
        diff = batch['coor'] - preds
        dis = torch.norm(diff, dim = -1)
        PcK_Acc = (dis < cfg.THR).float().mean()
        return {"dis": dis.mean(), "PcKAcc":PcK_Acc}

    def get_preds(self, outputs, batch):
        return get_preds_from_heatmap(outputs['heatmap'][-1])
    # def __len__(self):
    #     return 100