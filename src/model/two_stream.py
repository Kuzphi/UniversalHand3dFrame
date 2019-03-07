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
from src.model.networks import openpose_hand
from src.utils.misc import to_torch, to_numpy, to_cuda, to_cpu

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
	def train(self):
		self.rgb.train()
		self.depth.train()

	def eval(self):
		self.rgb.eval()
		self.depth.eval()

	def define_network(self):
		print("Setting up network")	
		self.rgb = eval(self.cfg.NETWORK.RGB.NAME)(**self.cfg.NETWORK.RGB)
		self.rgb = torch.nn.DataParallel(self.rgb, device_ids=self.cfg.GPUS).cuda()

		self.depth = eval(self.cfg.NETWORK.DEPTH.NAME)(**self.cfg.NETWORK.DEPTH)
		self.depth = torch.nn.DataParallel(self.depth, device_ids=self.cfg.GPUS).cuda()

		if self.cfg.NETWORK.RGB.PRETRAINED_WEIGHT_PATH:
			print("Loading Pretrained Weight")
			weight = torch.load(self.cfg.NETWORK.RGB.PRETRAINED_WEIGHT_PATH)
			self.rgb.load_state_dict(weight)

		if self.cfg.NETWORK.DEPTH.PRETRAINED_WEIGHT_PATH:
			print("Loading Pretrained Weight")
			weight = torch.load(self.cfg.NETWORK.DEPTH.PRETRAINED_WEIGHT_PATH)
			self.depth.load_state_dict(weight)

	def define_optimizer_and_scheduler(self):
		print("Setting up optimizer and optimizer scheduler")
		self.optimizer = eval('torch.optim.' + self.cfg.OPTIMIZER.NAME)(self.depth.parameters(),**self.cfg.OPTIMIZER.PARAMETERS)
		self.scheduler = eval('torch.optim.lr_scheduler.' + self.cfg.OPTIMIZER_SCHEDULE.NAME)(self.optimizer, **self.cfg.OPTIMIZER_SCHEDULE.PARAMETERS)
		self.scheduler.step()

	def forward(self):
		self.rgb_outputs = self.rgb(to_cuda(self.batch['input']))
		self.depth_outputs = self.depth(to_cuda(self.batch['input']))

		self.loss 	 = self.criterion()
		self.rgb_outputs = to_cpu(self.rgb_outputs)
		self.depth_outputs = to_cpu(self.depth_outputs)

	def eval_result(self):
		self.get_preds()
		dis_3d = torch.norm(self.batch['coor3d'] - self.preds['pose3d'], dim = -1).mean()
		dis_2d = torch.norm(self.batch['coor2d'][:, :, :2] - self.preds['pose2d'], dim = -1).mean()
		return {"dis2d": dis_2d, "dis3d": dis_3d}

	def get_preds(self):
		preds_2d, preds_3d = get_preds(self.batch['heatmap'], self.depth_outputs['heatmap'][-1], self.batch['matrix'])
		# preds_2d, preds_3d = get_preds(self.rgb_outputs['heatmap'][-1], self.depth_outputs['heatmap'][-1], self.batch['matrix'])
		self.preds = {'pose2d':preds_2d, 'pose3d': preds_3d}		
		return self.preds
		
	def criterion(self):
		criterion = nn.MSELoss()
		loss = torch.zeros(1).cuda()
		target = self.batch['depth'].cuda()
		for pred in self.depth_outputs['heatmap']:
			loss += criterion(pred, target)
		return loss

