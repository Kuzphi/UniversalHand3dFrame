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

class CPMWeakly(BaseModel):
	"""docstring for Hand25D"""
	def __init__(self, cfg):
		super(CPMWeakly, self).__init__(cfg)

	def eval_result(self):
		preds_2d, preds_3d = get_preds(self.outputs['heatmap'][-1], self.outputs['depthmap'])
		for i in range(preds_3d.shape[0]):
			preds_3d[i, :, :] = torch.matmul(preds_3d[i, :, :], self.batch['matrix'][i].transpose(0, 1))
		dis_3d = torch.norm(self.batch['coor'] - preds_3d, dim = -1).mean()
		dis_2d = torch.norm(self.batch['project'][:, :, :2] - preds_2d, dim = -1).mean()
		return {"dis2d": dis_2d, "dis3d": dis_3d}

	def get_preds(self):
		preds_2d, preds_3d = get_preds(self.outputs['heatmap'][-1], self.outputs['depthmap'])
		return {'pose2d':preds_2d, 'pose3d': preds_3d}

	def criterion(self,):
		loss  = CPMMSELoss(self.outputs, self.batch)
		# bs    = self.outputs['depthmap'].size(0)
		# index = self.batch['index'].long().cuda()
		
		# depth = self.outputs['depthmap'].view(bs, -1).gather(1, index).view(bs, 21)
		# loss += nn.functional.smooth_l1_loss(depth, self.batch['project'][:,:, 2].cuda())
		return loss

