from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn
from src.model.networks.MSRA_ResNet import resnet
from src.model.BaseModel import BaseModel
from src.core.loss import CPMMSELoss
from src.core.evaluate import get_preds_from_heatmap

class direct_two_stream(BaseModel):
	"""docstring for direct_two_stream"""
	def __init__(self, cfg):
		super(direct_two_stream, self).__init__(cfg)
		weight = torch.load('pretrained_weight/cpmRHD.torch')
		x = {}
		for key in weight:
			x[key[7:]] = weight[key]
		# self.networks['direct_two_stream'].color_net.load_state_dict(weight)
		self.networks['direct_two_stream'].module.color_net.load_state_dict(x, strict = True)
		self.set_requires_grad(self.networks['direct_two_stream'].module.color_net)


	def eval_result(self):
		self.get_preds()
		dis3d = torch.norm(self.batch['coor3d'] - self.preds['pose3d'], dim = -1).mean()
		dis2d = torch.norm(self.batch['coor2d'][:, :, :2] - self.preds['pose2d'], dim = -1).mean()
		return {"dis2d": dis2d, "dis3d": dis3d}

	def get_preds(self):
		preds2d = get_preds_from_heatmap(self.outputs['color_hm'][-1])
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
		criterion = nn.MSELoss()
		# for hm in self.outputs['color_hm']:
		# 	loss += criterion(hm, self.batch['color_hm'].cuda())

		for hm in self.outputs['depth_hm']:
			loss += criterion(hm, self.batch['depth_hm'].cuda())
		
		loss += nn.functional.smooth_l1_loss(self.outputs['depth'], self.batch['relative_depth'].cuda())
		return loss

