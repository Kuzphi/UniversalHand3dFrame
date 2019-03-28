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
import scipy.io as sio
import  matplotlib.pyplot as plt
from easydict import EasyDict as edict

from src.dataset import BaseDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image, im_to_numpy


class STB(BaseDataset):
    """docstring for STB"""
    def __init__(self, cfg):
        super(STB, self).__init__(cfg)
    def _get_db(self):
        self.db = []
        self.name = []
        self.all = 0
        dic = pickle.load(open(self.cfg.DATA_JSON_PATH))
        for name in dic:
            if name in self.cfg.PICK:
                self.db.append(dic[name]['l'])
                self.all += 1500
                self.name.append(name + "_left")
                self.db.append(dic[name]['r'])
                self.all += 1500
                self.name.append(name + "_right")
        return self.db
    
    def __len__(self):
        return self.all

    # def transforms(self, cfg, img, coor):
    #     # resize
    #     if cfg.has_key('RESIZE'):
    #         coor[:, 0] = coor[:, 0] / img.size(1) * cfg.RESIZE
    #         coor[:, 1] = coor[:, 1] / img.size(2) * cfg.RESIZE
    #         img = resize(img, cfg.RESIZE, cfg.RESIZE)

    #     if self.is_train:
    #         # Flip
    #         if cfg.FLIP and random.random() <= 0.5:
    #             img = torch.flip(img, dims = [1])
    #             coor[:, 1] = img.size(1) - coor[:, 1]

    #         # Color 
    #         if cfg.COLOR_NORISE:
    #             img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
    #             img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
    #             img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

    #     return img, coor

    def __getitem__(self, idx):

        name = self.name[idx // 1500]
        coor = self.db[idx // 1500][idx % 1500,:,:]
        coor = to_torch(coor)

        name = name.split("_")
        image_path   = os.path.join(self.cfg.ROOT, 'color', name[0], name[1] + '_' + str(idx % 1500) + '.png')
        depth_path   = os.path.join(self.cfg.ROOT, 'depth', name[0], name[1] + '_' + str(idx % 1500) + '.pickle')
        img = load_image(image_path, mode = 'RGB')
        depthmap = pickle.load(open(depth_path)).unsqueeze(0)
        #apply transforms into image and calculate cooresponding coor
        # if self.cfg.TRANSFORMS:
        #     img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)
            
        # print (name, idx % 1500, coor)
        # fig = plt.figure(1)
        # ax = fig.add_subplot(111)
        # plot_hand(im_to_numpy(img), coor, ax)
        # plt.show()

        meta = edict({'name': name})
        # heatmap = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))
        # for i in range(self.cfg.NUM_JOINTS - 1):
        #     heatmap[i] = draw_heatmap(heatmap[i], coor[i], self.cfg.HEATMAP.SIGMA)

        return {'img':img,
                'depthmap': depthmap,
                'coor': coor,
                'meta': meta}

