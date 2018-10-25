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

from src.dataset.BaseDataset import InferenceDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image

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
        