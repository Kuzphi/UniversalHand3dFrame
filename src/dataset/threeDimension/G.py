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

class G3D(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(G3D, self).__init__(cfg)

    def _get_db(self):
        self.names = ['Model1Tap','Model1Fist', 'Model1ArmRotate', 'Model1WristRotate']
        self.db = {}
        self.anno = {}
        for name in self.names:
            self.db[name] = []
            self.anno[name] = []
        
        for name in self.names:
            ipath = '/data/yufan/for_liangjian/%s/edge2hand_newD_newG_pix2pix_newfake_20k/test_latest/images/'%name
            lpath = '/data/liangjianData/TencentHand/Model1/%s/label/'%name
            for file in os.listdir(ipath):
                if file.endswith('fake_B.png'):
                    self.db[name].append(os.path.join(ipath, file))
                    idx = file[:7]
                    label = json.load(open(os.path.join(lpath, idx + '.json'),"r"))
                    self.anno[name].append(label['camera'])

        # self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        # return sorted(self.anno.keys())
        return self.db
        
    def transforms(self, cfg, img, coor):
        # resize
        if cfg.has_key('RESIZE'):
            img = resize(img, cfg.RESIZE, cfg.RESIZE)

        if self.is_train:
            # s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            # r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            
            # Flip
            if cfg.FLIP and random.random() <= 0.5:
                img = torch.flip(img, dims = [0])
                coor[0] = img.size(0) - coor[0]

            # Color 
            if cfg.COLOR_NORISE:
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        return img, coor

    def __getitem__(self, idx):
        name = self.names[idx // 5220]
        path = self.db[name][idx % 5220]
        coor = self.anno[name][idx % 5220]

        img = load_image(path)

        #calculate ground truth coordination
        coor = torch.tensor(coor).numpy()
        coor[1:,:] = coor[1:,:].reshape(5,4,-1)[:,::-1,:].reshape(20, -1)#iccv order !
        coor = np.array(coor)
        coor = to_torch(coor)
        coor = coor - coor[:1,:].repeat(21, 1)
        index_bone_length = torch.norm(coor[12,:] - coor[11,:])
        #apply transforms into image and calculate cooresponding coor
        if self.cfg.TRANSFORMS:
            img, label = self.transforms(self.cfg.TRANSFORMS, img , coor)

        # heat_map = np.zeros((self.cfg.NUM_JOINTS, img.shape[1], img.shape[2]))

        # for i in range(self.cfg.NUM_JOINTS):
        #     heat_map[i, :, :] = draw_heatmap(heat_map[i], coor[i], self.cfg.HEATMAP.SIGMA, type = self.cfg.HEATMAP.TYPE) 


        meta = edict({
                'name': path})

        return { 'input':  {'img':img,
                            'hand_side': torch.tensor([0, 1]).float()},
                 'coor': to_torch(coor),
                 # 'heat_map': to_torch(heat_map),
                 'index_bone_length': index_bone_length,
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
        median = np.median(dist)
        x, y = AUC(dist)
        auc = calc_auc(dist)
        auc00_30 = calc_auc(dist,  0, 50)
        auc30_50 = calc_auc(dist, 30, 50)
        print('AUC: ', auc)
        print('AUC  0 - 30: ', auc)
        print('AUC 30 - 50: ', auc)
        print('median:', median)
        import matplotlib.pyplot as plt
        fig = plt.figure('AUC')
        plt.plot(x, y)
        fig.savefig(os.path.join(cfg.CHECKPOINT,'AUC.png'))

    def __len__(self):
        return 5220 * 4