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
import matplotlib.pyplot as plt
import numpy as np

from easydict import EasyDict as edict
from mpl_toolkits.mplot3d import Axes3D

from src.dataset.BaseDataset import InferenceDataset
from src.utils.imutils import draw_joint2d, draw_joint3d
from src.utils.imutils import load_image, im_to_torch, im_to_numpy, draw_heatmap
from src.utils.misc import to_torch


from src.core.evaluate import get_preds_from_heatmap
from PIL import Image
import io
class Test(InferenceDataset):
    def __init__(self, cfg):
        super(Test, self).__init__(cfg)

    def __getitem__(self, idx):
        isleft = 1 if idx < len(self.db['left_hand']) else 0

        if isleft:
            img = self.db['left_hand'][idx]
        else:
            idx -= len(self.db['left_hand'])
            img = self.db['right_hand'][idx]
        img = Image.open(io.BytesIO(img))
        img = np.array(img)
        cv2.imwrite("origin_image/" +  str(idx) + '.jpg', img[:,:,::-1])
        img = cv2.resize(img,(256,256))
        # print (type(img), img.shape)
        img = img / 255. - .5
        img = im_to_torch(img)
        meta = {'isleft': isleft,
                'idx': idx}

        return { 'input': { "img": img,
                            "hand_side": torch.tensor([isleft, 1 - isleft]).float()},
                 'meta': meta}

    def __len__(self):
        return len(self.db['left_hand']) + len(self.db['right_hand'])
    
    def get_preds(self, outputs):
        pose3d = outputs['pose3d']
        heatmap = outputs['heatmap'][-1]
        pose2d = get_preds_from_heatmap(heatmap)
        return {'pose3d': pose3d,
                'pose2d': pose2d}
    def preds_demo(self, preds, fpath):
        print (len(self), len(preds['pose2d']), len(preds['pose3d']))
        for idx, (pose2d, pose3d) in enumerate(zip(preds['pose2d'], preds['pose3d'])):
            batch = self[idx]
            canvas = (batch['input']['img'] + 0.5) * 255
            canvas = im_to_numpy(canvas).astype(np.uint8).copy()
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection='3d')

            #draw 2d joint
            canvas = draw_joint2d(np.copy(canvas), pose2d)
            ax1.imshow(canvas)
            ax1.axis('off')

            #draw 3d joint
            draw_joint3d(pose3d, ax2)
            ax2.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            ax2.set_xlim([-3, 3])
            ax2.set_ylim([-3, 1])
            ax2.set_zlim([-3, 3])

            #show and save
            plt.savefig(fpath + str(idx))
            plt.show()
            