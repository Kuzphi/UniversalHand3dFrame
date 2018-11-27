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
from src.core.evaluate import calc_auc, AUC, calc_auc

class RHD3D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(RHD3D, self).__init__(cfg)

    def _get_db(self):
        self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        return sorted(self.anno.keys())
        
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
            if cfg.COLOR_NORISE:
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        return img, coor
    def __getitem__(self, idx):
        name = self.db[idx]
        label = self.anno[name]

        image_path   = os.path.join(self.cfg.ROOT, name)
        img = load_image(image_path)

        coor = to_torch(label['xyz_coor'])
        index_bone_length = torch.norm(coor[12,:] - coor[11,:])
        coor[0, :] = (coor[0] + coor[12]) / 2.
        coor = coor - coor[:1,:].repeat(21,1)
        #apply transforms into image and calculate cooresponding coor
        # if self.cfg.TRANSFORMS:
        #     img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)
        # print(idx, coor.sum())
        meta = edict({'name': name})
        isleft = int(label['isleft'])

        return {'input': {'img':img, 
                          'hand_side': torch.tensor([isleft, 1 - isleft]).float(),                          
                          },
                'index_bone_length': index_bone_length,
                'coor': to_torch(coor),
                'weight': 1,
                'meta': meta}

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