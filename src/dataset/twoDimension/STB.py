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

from src.dataset.BaseDataset import JointsDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image, resize, im_to_numpy, plot_hand
from src.core.evaluate import get_preds_from_heatmap, AUC, calc_auc

class STB2D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(STB2D, self).__init__(cfg)
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

    def transforms(self, cfg, img, coor):
        # resize
        if cfg.has_key('RESIZE'):
            coor[:, 0] = coor[:, 0] / img.size(1) * cfg.RESIZE
            coor[:, 1] = coor[:, 1] / img.size(2) * cfg.RESIZE
            img = resize(img, cfg.RESIZE, cfg.RESIZE)

        if self.is_train:
            # Flip
            if cfg.FLIP and random.random() <= 0.5:
                img = torch.flip(img, dims = [1])
                coor[:, 1] = img.size(1) - coor[:, 1]

            # Color 
            if cfg.COLOR_NORISE:
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        return img, coor

    def __getitem__(self, idx):

        name = self.name[idx // 1500]
        coor = self.db[idx // 1500][idx % 1500,:,:]
        coor = to_torch(coor)

        name = name.split("_")
        image_path   = os.path.join(self.cfg.ROOT, name[0], name[1] + '_' + str(idx % 1500) + '.png')
        img = load_image(image_path, mode = 'RGB')

        #apply transforms into image and calculate cooresponding coor
        if self.cfg.TRANSFORMS:
            img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)
            
        # print (name, idx % 1500, coor)
        # fig = plt.figure(1)
        # ax = fig.add_subplot(111)
        # plot_hand(im_to_numpy(img), coor, ax)
        # plt.show()

        meta = edict({'name': name})
        heatmap = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))
        for i in range(self.cfg.NUM_JOINTS - 1):
            heatmap[i] = draw_heatmap(heatmap[i], coor[i], self.cfg.HEATMAP.SIGMA)

        return {'input': {'img':img},
                'heatmap': heatmap,
                'coor': coor, 
                'weight': 1,
                'meta': meta}

    def eval_result(self, outputs, batch, cfg = None):
        gt_coor = batch['coor']
        # print(outputs['pose3d'].size(), batch['index_bone_length'].size())
        pred_coor = outputs['pose3d'] * batch['index_bone_length'].view(-1,1,1).repeat(1,21,3)

        dis = torch.norm(gt_coor - pred_coor, dim = -1)

        dis = torch.mean(dis)        
        return {"dis": dis}

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