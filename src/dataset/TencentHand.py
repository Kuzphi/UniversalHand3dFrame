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
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image


class TencentHand(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(TencentHand, self).__init__(cfg)

    def _get_db(self):
        return json.load(open(self.cfg.DATA_JSON_PATH))
        
    def transforms(self, cfg, img, coor):
        if self.is_train:
            # s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            # r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            # if random.random() <= 0.5:
                # img = torch.from_numpy(fliplr(img.numpy())).float()
                # pts = shufflelr(pts, width=img.size(2), dataset='RHD')
                # c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        return img, coor
    def __getitem__(self, idx):
        w = self.db[idx]

        image_path   = os.path.join(self.cfg.ROOT, w[1], w[1] + w[2], 'image', w[0] + '.png')
        label_path = os.path.join(self.cfg.ROOT, w[1], w[1] + w[2], 'label', w[0] + '.json')

        img = load_image(image_path)

        
        label = json.load(open(label_path))

        #calculate ground truth coordination
        coor = np.array(label['perspective'])
        coor[:,0] = (coor[:,0]) * img.shape[0]
        coor[:,1] = (1 - coor[:,1]) * img.shape[1]
        coor = coor.astype(np.int)

        #conver to tensor
        coor = to_torch(coor)
        #apply transforms into image and calculate cooresponding coor
        if self.cfg.TRANSFORMS:
            img, label = self.transforms(self.cfg.TRANSFORMS, img , coor)
        # torch.with_no_gard():
        heat_map = np.zeros((self.cfg.NUM_JOINTS, img.shape[1], img.shape[2]))

        for i in range(self.cfg.NUM_JOINTS):
            heat_map[i, :, :] = draw_heatmap(heat_map[i], coor[i], self.cfg.HEATMAP.SIGMA, type = self.cfg.HEATMAP.TYPE) 


        meta = edict({
                'name': w[1] + ' ' + w[2] + ' ' + w[0]})

        return { 'input': img.cuda(),
                 'coor': to_torch(coor).cuda(),
                 'heat_map': to_torch(heat_map).cuda(),
                 'weight': 1,
                 'meta': meta}
        