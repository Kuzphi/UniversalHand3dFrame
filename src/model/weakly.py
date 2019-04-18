from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn
from src.model.BaseModel import BaseModel
from src.model.utils.loss import CPMMSELoss
from src.model.utils.evaluation import get_preds

__all__ = ['Weakly']

class Weakly(BaseModel):
	"""docstring for Hand25D"""
	def __init__(self, cfg):
		super(Weakly, self).__init__(cfg)
		if cfg.STAGE == 1: #train heatmap
			self.set_requires_grad(self.networks.module.depth)			

		# self.set_requires_grad(self.networks[].module.depth)			
		if cfg.STAGE == 2: #train only depth network
			for name, net in self.networks.iteritems():
				for net_name in net.module.__dict__['_modules']:
					if 'depth' in net_name:
						continue
					self.set_requires_grad(net.module.__dict__['_modules'][net_name])
			# for param in self.network.module.stage6.parameters():
			# 		print(param.requires_grad)

		if cfg.STAGE == 3: #train both networks
			pass

	def eval_result(self):
		self.get_preds()
		dis_3d = torch.norm(self.batch['coor3d'] - self.preds['pose3d'], dim = -1).mean()
		dis_2d = torch.norm(self.batch['coor2d'][:, :, :2] - self.preds['pose2d'], dim = -1).mean()
		return {"dis2d": dis_2d, "dis3d": dis_3d}

	def get_preds(self):
		preds_2d, preds_3d = get_preds(self.outputs['heatmap'][-1], self.outputs['depthmap'], self.batch)
		self.preds = {'pose2d':preds_2d, 'pose3d': preds_3d}		
		return self.preds

	def criterion(self):
		loss = torch.zeros(1).cuda()
		if self.cfg.STAGE == 1 or self.cfg.STAGE == 3:
			loss  += CPMMSELoss(self.outputs, self.batch)

		if self.cfg.STAGE == 2 or self.cfg.STAGE == 3:
			bs    = self.outputs['depthmap'].size(0)
			index = self.batch['index'].long().cuda()
			
			depth = self.outputs['depthmap'].view(bs, -1).gather(1, index).view(bs, 21)
			loss += nn.functional.smooth_l1_loss(depth, self.batch['coor2d'][:,:, 2].cuda())
		return loss

