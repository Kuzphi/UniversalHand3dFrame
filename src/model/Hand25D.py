from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn
from src.model.networks.MSRA_ResNet import resnet
from src.model.BaseModel import BaseModel
from src.core.loss import CPMMSELoss
from src.core.evaluate import get_preds

class Hand25D(BaseModel):
	"""docstring for Hand25D"""
	def __init__(self, cfg):
		super(Hand25D, self).__init__(cfg)
		# if cfg.STAGE == 1: #train heatmap
		# 	self.set_requires_grad(self.network.module.depth)			
		# if cfg.STAGE == 2: #train only depth network
		# 	for name in self.network.module.__dict__['_modules']:
		# 		if 'depth' in name:
		# 			continue
		# 		network = self.network.module.__dict__['_modules'][name]
		# 		self.set_requires_grad(network)

		# if cfg.STAGE == 3: #train both networks
		# 	pass

	def set_batch(self, batch):	
		self.batch = batch
		
	def eval_result(self):
		self.get_preds()
		dis_3d = torch.norm(self.batch['coor3d'] - self.preds['pose3d'], dim = -1).mean()
		dis_2d = torch.norm(self.batch['coor2d'][:, :, :2] - self.preds['pose2d'], dim = -1).mean()
		return {"dis2d": dis_2d, "dis3d": dis_3d}

	def get_preds(self):
		preds_2d, preds_3d = get_preds(self.rgb_outputs['heatmap'][-1], self.depth_outputs['heatmap'][-1], self.batch['matrix'])
		self.preds = {'pose2d':preds_2d, 'pose3d': preds_3d}		
		return self.preds

	def criterion(self):
		criterion = nn.MSELoss()
		loss = torch.zeros(1).cuda()
		# if self.cfg.STAGE == 1 or self.cfg.STAGE == 3:
		target = self.batch['heatmap'].cuda()
		for pred in self.outputs['heatmap']:
			loss += criterion(pred, target)

		target = self.batch['depth'].cuda()
		for pred in self.outputs['depth']:
			loss += criterion(pred, target)

		# if self.cfg.STAGE == 2 or self.cfg.STAGE == 3:
			# bs    = self.outputs['depthmap'].size(0)
			# index = self.batch['index'].long().cuda()
			
			# depth = self.outputs['depthmap'].view(bs, -1).gather(1, index).view(bs, 21)
			# print (self.outputs.keys())
			# loss += nn.functional.smooth_l1_loss(self.outputs['depth'], self.batch['coor2d'][:,:, 2].cuda())
		return loss

