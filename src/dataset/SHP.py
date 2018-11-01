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
from easydict import EasyDict as edict

from src.dataset.BaseDataset import JointsDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image, resize
from src.core.evaluate import get_preds_from_heatmap

class SHP(JointsDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(SHP, self).__init__(cfg)
        self.upsampler = torch.nn.Upsample(scale_factor = 8, mode = 'bilinear', align_corners = True)
    def _get_db(self):
        self.db = []
        self.name = []
        self.all = 0
        for name in sorted(os.listdir(self.cfg.DATA_JSON_PATH)):
            if name[2:10] == 'Counting':
                if (name.startswith('B6Counting') and not self.is_train) or (not name.startswith('B6Counting') and self.is_train):
                    matPath = os.path.join(self.cfg.DATA_JSON_PATH, name)
                    self.db.append(sio.loadmat(matPath)['handPara'])
                    self.all += 1500
                    self.name.append(name[:-4])

        return self.db
    
    def __len__(self):
        return self.all

    def transforms(self, cfg, img, coor):

        if cfg.RESIZE:
            img = resize(img, cfg.RESIZE, cfg.RESIZE)

        if self.is_train:
            # s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            # r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            # if random.random() <= 0.5:
                # img = torch.from_numpy(fliplr(img.numpy())).float()
                # pts = shufflelr(pts, width=img.size(2), dataset='SHP')
                # c[0] = img.size(2) - c[0]

            # Color
            if cfg.COLOR_NORISE:
                img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
                img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

        return img, coor

    def __getitem__(self, idx):

        name = self.name[idx // 1500]
        coor = self.db[idx // 1500][:,:,idx % 1500]

        coor = coor.transpose(1,0)
        coor[1:, ] = coor[1:, :].reshape(5,4,-1)[::-1,::-1,:].reshape(20, -1)
        coor /= 100.
        coor = np.array(coor)

        name = name.split("_")

        if name[1] == 'BB':
            image_path   = os.path.join(self.cfg.ROOT, name[0]+"_cropped", "_".join([name[1], 'left', str(idx % 1500)]) + '.png')
        elif name[1] == 'SK':
            image_path   = os.path.join(self.cfg.ROOT, name[0]+"_cropped", "_".join([name[1], 'color', str(idx % 1500)]) + '.png')
        else:
            raise Exception("Unrecognized name {}".format(name))

        img = load_image(image_path)

        #apply transforms into image and calculate cooresponding coor
        if self.cfg.TRANSFORMS:
            img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)

        meta = edict({'name': name})
        isleft = 1
        heatmap = torch.zeros(self.cfg.NUM_JOINTS, img.size(0),img.size(1))

        # for i in xrange(self.cfg.NUM_JOINTS):
        #     heatmap[i,:,:] =  draw_heatmap(heatmap[i,:,:], coor, sigma = 1)

        return {'input': {'img':img, 
                          'hand_side': torch.tensor([isleft, 1 - isleft]).float(),
                          # 'heatmap': heatmap
                          },
                'coor': to_torch(coor),
                'weight': 1,
                'meta': meta}

    def eval_result(self, outputs, batch, cfg = None):
        gt_coor = batch['coor']
        pd_coor = outputs['pose3d']
        dis = torch.norm(gt_coor - pd_coor, dim = -1)
        dis = torch.mean(dis)        
        return {"dis": dis}

    def get_preds(self, outputs):
        return outputs['pose3d']

    def preds_demo(self, preds, fpath):
        for i in range(len(self)):
            img = self[i]['input']['img']
            img = im_to_numpy(img)
            canvas = (img + .5) * 255
            plt.figure(self[i]['meta']['name'])
            ax = plt.add_subplot(111)
            plot_hand_2d(canvas, preds[i], ax)
            plt.show()

    # def __len__(self):
    #     return 100