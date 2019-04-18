from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn
from src.model.BaseModel import BaseModel
from src.model.utils.loss import CPMMSELoss
from src.model.utils.evaluation import get_preds
from src.utils.misc import to_torch, to_numpy, to_cuda, to_cpu

__all__ = ['two_stream']
class two_stream(BaseModel):
	"""docstring for two_stream"""
	def __init__(self, cfg):
		super(two_stream, self).__init__(cfg)
		# if cfg.STAGE == 1: #train heatmap
		# 	self.set_requires_grad(self.network.module.depth)			
		# if cfg.STAGE == 2: #train only depth network
		# 	for name in self.network.module.__dict__['_modules']:
		# 		if name == 'depth':
		# 			continue
		# 		network = self.network.module.__dict__['_modules'][name]
		# 		self.set_requires_grad(network)
		# 	# for param in self.network.module.stage6.parameters():
		# 	# 		print(param.requires_grad)

		# if cfg.STAGE == 3: #train both networks
		# 	pass
	def forward(self):
		self.rgb_outputs = self.networks['RGB'](to_cuda(self.batch['input']))
		self.depth_outputs = self.networks['DEPTH'](to_cuda(self.batch['input']))

		self.loss 	 = self.criterion()
		self.rgb_outputs = to_cpu(self.rgb_outputs)
		self.depth_outputs = to_cpu(self.depth_outputs)

	def eval_result(self):
		self.get_preds()
		dis_3d = torch.norm(self.batch['coor3d'] - self.preds['pose3d'], dim = -1).mean()
		dis_2d = torch.norm(self.batch['coor2d'][:, :, :2] - self.preds['pose2d'], dim = -1).mean()
		return {"dis2d": dis_2d, "dis3d": dis_3d}

	def get_preds(self):
		# preds_2d, preds_3d = get_preds(self.batch['heatmap'], self.batch['depth'], self.batch)
		preds_2d, preds_3d = get_preds(self.batch['heatmap'], self.depth_outputs['heatmap'][-1], self.batch)
		# preds_2d, preds_3d = get_preds(self.rgb_outputs['heatmap'][-1], self.depth_outputs['heatmap'][-1], self.batch)
		self.preds = {'pose2d':preds_2d, 'pose3d': preds_3d}		
		return self.preds
		
	def criterion(self):
		criterion = nn.MSELoss()
		loss = torch.zeros(1).cuda()
		target = self.batch['depth'].cuda()
		for pred in self.depth_outputs['heatmap']:
			loss += criterion(pred, target)
		return loss

