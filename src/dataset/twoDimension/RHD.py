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
import scipy

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
from src.core.evaluate import get_preds_from_heatmap, AUC, calc_auc

class RHD2D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(RHD2D, self).__init__(cfg)

    def _get_db(self):
        self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        return sorted(self.anno.keys())
        
    def transforms(self, cfg, img, coor):
        if self.is_train:
            # s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            # r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            
            if cfg.FLIP and random.random() <= 0.5:
                img = torch.flip(img, dims = [0])
                coor[:, 0] = img.size(0) - coor[:, 0]

            # Color
            if cfg.COLOR_NORISE:
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        return img, coor
    def __getitem__(self, idx):
        name = self.db[idx]
        label = self.anno[name]

        image_path   = os.path.join(self.cfg.ROOT, name)
        img = load_image(image_path, mode = 'GBR') # already / 255

        coor = label['uv_coor']
        coor[1:,:] = coor[1:,:].reshape(5,4,-1)[:,::-1,:].reshape(20, -1)
        coor = np.array(coor)
        coor = to_torch(coor)
        #apply transforms into image and calculate cooresponding coor
        # if self.cfg.TRANSFORMS:
        #     img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)
        
        meta = edict({'name': name})
        heatmap = torch.zeros(22, img.size(1), img.size(2))
        for i in range(21):
            heatmap[i] = draw_heatmap(heatmap[i], coor[i], self.cfg.HEATMAP.SIGMA)
        
        return {'input': {'img':img},
                'heatmap': heatmap,
                'coor': to_torch(coor[:,:2]),
                'weight': 1,
                'meta': meta}

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
    # def __len__(self):
    #     return 100