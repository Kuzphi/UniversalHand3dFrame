from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn
from src.model.BaseModel import BaseModel
from src.core.loss import CPMMSELoss
from src.core.evaluate import get_preds_from_heatmap
__all__ = ['OpenPose']
class OpenPose(BaseModel):
	"""docstring for OpenPose"""
	def __init__(self, cfg):
		super(OpenPose, self).__init__(cfg)

	def eval_result(self):
		preds_2d = get_preds_from_heatmap(self.outputs['heatmap'][-1])
		dis_2d = torch.norm(self.batch['project'][:, :, :2] - preds_2d, dim = -1).mean()
		return {"dis2d": dis_2d}

	def get_preds(self):
		preds_2d = get_preds_from_heatmap(self.outputs['heatmap'][-1])
		return {'pose2d':preds_2d}

	def criterion(self,):
		loss  = CPMMSELoss(self.outputs, self.batch)
		return loss

