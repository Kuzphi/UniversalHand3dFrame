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
from src.utils.imutils import load_image, resize
from src.core.evaluate import calc_auc, AUC, calc_auc

class RHD3D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(RHD3D, self).__init__(cfg)

    def _get_db(self):
        self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        return sorted(self.anno.keys())
        
    def transforms(self, cfg, img, coor, project_coor, matrix):
        # resize
        if cfg.has_key('RESIZE'):

            project_coor[:, 0] = project_coor[:, 0] * cfg.RESIZE / img.size(1) 
            project_coor[:, 1] = project_coor[:, 1] * cfg.RESIZE / img.size(2) 

            img = resize(img, cfg.RESIZE, cfg.RESIZE)
            scale =[[1. * cfg.RESIZE / img.shape[0], 0,  0],
                    [0,    1. * cfg.RESIZE / img.shape[1],  0],
                    [0,         0,  1]]
            matrix = np.matmul(scale, matrix)

            
        if self.is_train:
            # Color 
            if cfg.COLOR_NORISE:
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        return img, coor, project_coor, matrix
    def __getitem__(self, idx):
        name = self.db[idx]
        label = self.anno[name]

        image_path   = os.path.join(self.cfg.ROOT, 'color', name + '.png')
        img = load_image(image_path) #C * W * H
        coor = to_torch(label['xyz'])
        project_coor = label['project']

        # print(project_coor[:, 0].min(), project_coor[:, 1].max())
        assert project_coor[:, :2].min() >= 0 and project_coor[:, :2].max() < 320

        matrix = label['K']
        # norm the pose
        # index_bone_length = torch.norm(coor[12,:] - coor[11,:])
        # coor[0, :] = (coor[0] + coor[12]) / 2.
        # coor = coor - coor[:1,:].repeat(21,1)

        #apply transforms into image and calculate cooresponding coor and camera instrict matrix
        if self.cfg.TRANSFORMS:
            img, coor, project_coor, matrix = self.transforms(self.cfg.TRANSFORMS, img , coor, project_coor, matrix)

        # print(project_coor[:, 0].max())
        assert project_coor[:, :2].min() >= 0 and project_coor[:, :2].max() < 256
        
        matrix = np.linalg.inv(matrix) #take the inversion of matrix
        meta = edict({'name': name})
        isleft = name[-1] == 'L'
        #corresponding depth position in depth map
        project_coor = torch.tensor(project_coor).long()
        index = torch.tensor([i * img.size(1) * img.size(2) + project_coor[i,0] * img.size(1) + project_coor[i,1] for i in range(21)])

        assert index.max() < img.size(1) * img.size(2) * 21, 'Wrong Position'
        heatmap = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))

        for i in range(21):
            heatmap[i] = draw_heatmap(heatmap[i], project_coor[i], self.cfg.HEATMAP.SIGMA)

        return {'input': {'img':img,
                          'hand_side': torch.tensor([isleft, 1 - isleft]).float(),                          
                          },
                'index': index, 
                'matrix': to_torch(matrix),
                # 'index_bone_length': index_bone_length,
                'heatmap': heatmap,
                'coor': to_torch(coor),
                'project': to_torch(project_coor),
                'weight': 1,
                'meta': meta}

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
    #     return 10