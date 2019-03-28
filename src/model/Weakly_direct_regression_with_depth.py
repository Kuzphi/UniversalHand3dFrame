from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn
from src.model.BaseModel import BaseModel
from src.core.loss import CPMMSELoss
from src.utils.misc import to_cuda, to_cpu
from src.model.utils.evaluate import get_preds_from_heatmap

__all__ = ['Weakly_direct_regression_with_depth']
class Weakly_direct_regression_with_depth(BaseModel):
	"""docstring for Weakly_direct_regression_with_depth"""
	def __init__(self, cfg):
		super(Weakly_direct_regression_with_depth, self).__init__(cfg)
		
	def reprocess(self, cfg):

	def forward(self):
		self.outputs = self.networks['Regression'](to_cuda(self.batch['input']))

		root_depth = self.batch['root_depth']
		index_bone_length = self.batch['index_bone_length']	
		depthmap_max = self.batch['depthmap_max']
		depthmap_range = self.batch['depthmap_range']

		reg_input = torch.zeros(self.batch['coor3d'].shape).cuda()
		reg_input[:, :, :2] = self.batch['coor2d'][:, :, :2].cuda().clone()
		reg_input[:, :, 2] = self.outputs['depth'] * index_bone_length.unsqueeze(1).cuda() + root_depth.unsqueeze(1).cuda()
		reg_input[:, :, 2] = (depthmap_max.unsqueeze(1).cuda() - reg_input[:, :, 2]) / depthmap_range.unsqueeze(1).cuda()
		reg_input = reg_input.view(reg_input.size(0), -1, 1, 1)
		self.outputs['depthmap'] = self.networks['DepthRegularizer'](reg_input).squeeze()
		self.loss = self.criterion()

		self.outputs = to_cpu(self.outputs)

	def eval_result(self):
		self.get_preds()
		dis3d = torch.norm(self.batch['coor3d'] - self.preds['pose3d'], dim = -1).mean()
		dis2d = torch.norm(self.batch['coor2d'][:, :, :2] - self.preds['pose2d'], dim = -1).mean()
		return {"dis2d": dis2d, "dis3d": dis3d}

	def get_batch_preds(self):
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
		L1Loss = nn.L1Loss()
		loss = torch.zeros(1).cuda()
		loss += CPMMSELoss(self.outputs, self.batch)
		loss += nn.functional.smooth_l1_loss(self.outputs['depth'], self.batch['relative_depth'].cuda()) * 0.1 
		loss += L1Loss(self.outputs['depthmap'], self.batch['depthmap'].float().cuda())
		return loss

