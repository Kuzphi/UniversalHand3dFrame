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
from src.utils.imutils import load_image, resize
from src.core.evaluate import get_preds, get_preds_from_heatmap, AUC, calc_auc

class RHD2D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(RHD2D, self).__init__(cfg)

    def _get_db(self):
        self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        return sorted(self.anno.keys())
        
    def transforms(self, cfg, img, coor2d, matrix):
        # resize
        if cfg.has_key('RESIZE'):
            xscale, yscale = 1. * cfg.RESIZE / img.size(1), 1. * cfg.RESIZE / img.size(2) 
            coor2d[:, 0] *= xscale
            coor2d[:, 1] *= yscale
            scale =[[xscale,    0,  0],
                    [0,    yscale,  0],
                    [0,         0,  1]]
            matrix = np.matmul(scale, matrix)

            img = resize(img, cfg.RESIZE, cfg.RESIZE)

            
        if self.is_train:
            # Color 
            if cfg.COLOR_NORISE:
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 256
        return img, coor2d, matrix

    def __getitem__(self, idx):
        name = self.db[idx]
        label = self.anno[name]

        image_path   = os.path.join(self.cfg.ROOT, 'color', name + '.png')
        img = load_image(image_path)# already / 255 with C * W * H
        


        coor2d = label['project']
        assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 320
        matrix = label['K']
        #apply transforms into image and calculate cooresponding coor and camera instrict matrix
        if self.cfg.TRANSFORMS:
            img, coor2d, matrix = self.transforms(self.cfg.TRANSFORMS, img, coor2d, matrix)

        meta = edict({'name': name})
        isleft = name[-1] == 'L'

        coor2d[1:,:] = coor2d[1:,:].reshape(5,4,-1)[:,::-1,:].reshape(20, -1)
        coor2d = to_torch(np.array(coor2d))
        coor2d[:,:2] = coor2d[:,:2].long().float() 
        
        matrix = np.linalg.inv(matrix) #take the inversion of matrix

        #corresponding depth position in depth map
        index = torch.tensor([i * img.size(1) * img.size(2) + coor2d[i,0].long() * img.size(1) + coor2d[i,1].long() for i in range(21)])

        assert index.max() < img.size(1) * img.size(2) * 21, 'Wrong Position'
        heatmap = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))
        depth   = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))

        coor3d = coor2d.clone()
        coor3d[:,:2] *= coor3d[:, 2:]
        coor3d = torch.matmul(coor3d, to_torch(matrix).transpose(0, 1))

        root_depth = coor2d[0, 2].clone()
        index_bone_length = torch.norm(coor3d[9,:] - coor3d[10,:])

        for i in range(21):
            heatmap[i] = draw_heatmap(heatmap[i], coor2d[i], self.cfg.HEATMAP.SIGMA)
            depth[i]   = heatmap[i] * (coor2d[i, 2] - coor2d[0, 2]) / index_bone_length

        relative_depth = (coor2d[:,2] - coor2d[0, 2]) / index_bone_length
        # pred2d, pred3d = get_preds(heatmap.unsqueeze(0), depth.unsqueeze(0), to_torch(matrix).unsqueeze(0))

        # print(pred3d[0] - coor3d)

        return {'input': {'img':img,
                          'hand_side': torch.tensor([isleft, 1 - isleft]).float(),                          
                          },
                'index': index, 
                'matrix': to_torch(matrix),
                'heatmap': heatmap,
                'depth' :  depth,                
                'coor3d': to_torch(coor3d),
                'coor2d': to_torch(coor2d),                
                'root_depth': root_depth,
                'index_bone_length': index_bone_length,
                'relative_depth': relative_depth,
                'weight': 1,
                'meta': meta,
                }

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