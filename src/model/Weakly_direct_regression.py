from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn
from src.model.BaseModel import BaseModel
from src.core.loss import CPMMSELoss
from src.core.evaluate import get_preds_from_heatmap
__all__ = ['Weakly_direct_regression']
class Weakly_direct_regression(BaseModel):
	"""docstring for Weakly_direct_regression"""
	def __init__(self, cfg):
		super(Weakly_direct_regression, self).__init__(cfg)
		if cfg.STAGE == 1: #train heatmap
			pass
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
		dis3d = torch.norm(self.batch['coor3d'] - self.preds['pose3d'], dim = -1).mean()
		dis2d = torch.norm(self.batch['coor2d'][:, :, :2] - self.preds['pose2d'], dim = -1).mean()
		return {"dis2d": dis2d, "dis3d": dis3d}

	def get_preds(self):
		preds2d = get_preds_from_heatmap(self.outputs['heatmap'][-1])
		# preds2d = get_preds_from_heatmap(self.batch['heatmap'])
		preds3d = torch.zeros((preds2d.size(0), 21, 3))

		root_depth = self.batch['root_depth']
		index_bone_length = self.batch['index_bone_length']

		preds3d[:,:,:2] = preds2d.clone()
		preds3d[:,:,2]  = self.outputs['depth'] * index_bone_length.unsqueeze(1) + root_depth.unsqueeze(1)
		# preds3d[:,:,2]  = self.batch['relative_depth'] * index_bone_length.unsqueeze(1) + root_depth.unsqueeze(1)
		

		preds3d[:, :, :2] *= preds3d[:, :, 2:]

		for i in range(preds3d.size(0)):
			preds3d[i, :, :] = torch.matmul(preds3d[i, :, :], self.batch['matrix'][i].transpose(0,1))

		self.preds = {'pose2d':preds2d, 'pose3d': preds3d}
		return self.preds

	def criterion(self):
		loss = torch.zeros(1).cuda()
		if self.cfg.STAGE == 1 or self.cfg.STAGE == 3:
			loss  += CPMMSELoss(self.outputs, self.batch)

		if self.cfg.STAGE == 2 or self.cfg.STAGE == 3:
			# bs    = self.outputs['depthmap'].size(0)
			# index = self.batch['index'].long().cuda()
			
			# depth = self.outputs['depthmap'].view(bs, -1).gather(1, index).view(bs, 21)
			# print (self.outputs.keys())
			loss += nn.functional.smooth_l1_loss(self.outputs['depth'], self.batch['relative_depth'].cuda())
		return loss

