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

from src.dataset import JointsDataset, BaseDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image, resize, im_to_numpy
from src.model.utils.evaluation import get_preds_from_heatmap
from src.model.utils.evaluation import calc_auc, AUC, calc_auc
__all__ = ['Multiview']
class Multiview(BaseDataset):
	"""docstring for Multiview"""
	def __init__(self, cfg):
		super(Multiview, self).__init__(cfg)

	def _get_db(self):
		self.name = self.cfg.PICK
		self.db = pickle.load(open(self.cfg.DATA_JSON_PATH))
		self.all = 1500 * len(self.cfg.PICK)
		return self.db
	
	def __len__(self):
		# return 100
		return self.all
	def __getitem__(self, idx):
		name = self.name[idx // 1500]
		coor2d = self.db[name]['sk']['coor2d'][idx % 1500,:,:]
		matrix = self.db[name]['sk']['matrix'][idx % 1500]

		name = name.split('_')
		image_path   = os.path.join(self.cfg.ROOT, name[0], 'SK_' + str(idx % 1500) + '.png')
		img = load_image(image_path, mode = 'RGB')
		depth_path = os.path.join(self.cfg.ROOT, name[0], 'SK_depth_{}.pickle'.format(idx % 1500))
		depthmap = pickle.load(open(depth_path)).unsqueeze(0)

		# coor2d = label['project']
		# assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 320
		

		# coor2d = torch.matmul(coor3d, to_torch(matrix).transpose(0, 1))
		meta = edict({'name': name})
		return {'img':img,
				'depthmap': depthmap,
				# 'index': index, 
				'matrix': to_torch(matrix),
				'coor2d': to_torch(coor2d),
				'meta': meta,
				}

	