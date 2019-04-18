import torch
import itertools
import numpy as np
import random

from torch import nn
from torch.nn.functional import interpolate

from src.model import BaseModel
from src.utils.imutils import resize, draw_heatmap
from src.utils.misc import to_cpu, to_torch, to_cuda
from src.utils.image_pool import ImagePool
from src.model.utils.loss import CPMMSELoss
from src.model.utils.evaluation import AUC, calc_auc, get_preds_from_heatmap
from src.model.networks.GAN import GANLoss
__all__ = ['Weakly_GAN']
class Weakly_GAN(BaseModel):

	def __init__(self, cfg):
		super(Weakly_GAN, self).__init__(cfg)
		self.netD = self.networks['Discriminator']
		self.netG = self.networks['Generator']
		self.optimizer_D = self.optimizers['D']
		self.optimizer_G = self.optimizers['G']
		self.fake_AB_pool = ImagePool(50)
		self.device = torch.device('cuda:0')
		self.criterionGAN = GANLoss(use_lsgan= False).to(self.device)

	def transforms(self, cfg, img, depth, coor2d, matrix):
		# resize
		if cfg.has_key('RESIZE'):
			xscale, yscale = 1. * cfg.RESIZE / img.size(1), 1. * cfg.RESIZE / img.size(2) 
			coor2d[:, 0] *= xscale
			coor2d[:, 1] *= yscale
			scale =[[xscale,    0,  0],
					[0,    yscale,  0],
					[0,         0,  1]]
			matrix = np.matmul(scale, matrix)

			img = resize(img, cfg.RESIZE, cfg.RESIZE)
			depth = depth.unsqueeze(0)
			depth = interpolate(depth, (128, 128), mode = 'bilinear', align_corners = True)[0,...]

		if cfg.COLOR_NORISE:
			img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
			img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
			img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

		assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 256
		return img, depth, coor2d, matrix

	def reprocess(self, input, data_cfg):
		img = input['img']
		depthmap = input['depthmap']

		coor2d = input['coor2d']
		assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 320
		matrix = input['matrix']
		meta = input['meta']
		#apply transforms into image and calculate cooresponding coor and camera instrict matrix
		if data_cfg.TRANSFORMS:
			img, depthmap, coor2d, matrix = self.transforms(data_cfg.TRANSFORMS, img, depthmap, coor2d, matrix)

		# if depthmap_max - depthmap_min < 1e-6:
		#     print(name, ": ", depthmap_max - depthmap_min)
		# depthmap = (depthmap.max() - depthmap) / (depthmap_max - depthmap_min)
		# print(depthmap)
		
		

		matrix = np.linalg.inv(matrix) #take the inversion of matrix

		coor3d = coor2d.clone()
		coor3d[:,:2] *= coor3d[:, 2:]
		coor3d = torch.matmul(coor3d, to_torch(matrix).transpose(0, 1))

		root_depth = coor2d[0, 2].clone()
		index_bone_length = torch.norm(coor3d[9,:] - coor3d[10,:])
		relative_depth = (coor2d[:,2] - root_depth) / index_bone_length

		depthmap *= float(2**16 - 1)
		depthmap = (depthmap - root_depth) / index_bone_length
		depthmap_max = depthmap.max()
		depthmap_min = depthmap.min()
		depthmap = (depthmap - depthmap_min) / (depthmap_max - depthmap_min)
		
		heatmap = torch.zeros(data_cfg.NUM_JOINTS, img.size(1), img.size(2))
		depth   = torch.zeros(data_cfg.NUM_JOINTS, img.size(1), img.size(2))

		for i in range(21):
			heatmap[i] = draw_heatmap(heatmap[i], coor2d[i], data_cfg.HEATMAP.SIGMA)
			depth[i]   = heatmap[i] * (coor2d[i, 2] - coor2d[0, 2]) / index_bone_length


		return {'input': {'img':img,
						  'depthmap': depthmap,
						  },
				'heatmap': heatmap,
				'matrix': to_torch(matrix),
				'color_hm': heatmap,
				'depth_hm' :  depth,
				'depthmap': depthmap,
				'depthmap_max': depthmap_max,
				'depthmap_range': depthmap_max - depthmap_min,
				'coor3d': to_torch(coor3d),
				'coor2d': to_torch(coor2d),
				'root_depth': root_depth,
				'index_bone_length': index_bone_length,
				'relative_depth': relative_depth,
				'weight': 1,
				'meta': meta,
				}

	def TaskLoss(self):
		L1Loss = nn.L1Loss()
		loss = torch.zeros(1).cuda()
		loss += CPMMSELoss(self.task_outputs, self.batch)
		loss += nn.functional.smooth_l1_loss(self.task_outputs['depth'], self.batch['relative_depth'].cuda()) * 0.1 
		loss += L1Loss(self.task_outputs['depthmap'], self.batch['depthmap'].float().cuda())
		return loss

	def set_batch(self, batch):
		# self.batch = batch
		self.batch = batch
		self.real_A = batch['input']['img'].cuda()
		self.real_B = batch['input']['depthmap'].cuda()		

	def forward(self):
		# print (self.real_A)
		self.fake_B = self.netG(self.real_A)
		task_input = {'img':self.real_A, 'depthmap': self.fake_B}
	
		self.task_outputs = self.networks['Regression'](to_cuda(task_input))
		root_depth = self.batch['root_depth']
		index_bone_length = self.batch['index_bone_length']	
		depthmap_max = self.batch['depthmap_max']
		depthmap_range = self.batch['depthmap_range']

		reg_input = torch.zeros(self.batch['coor3d'].shape).cuda()
		reg_input[:, :, :2] = self.batch['coor2d'][:, :, :2].cuda().clone()
		reg_input[:, :, 2] = self.task_outputs['depth'] * index_bone_length.unsqueeze(1).cuda() + root_depth.unsqueeze(1).cuda()
		reg_input[:, :, 2] = (depthmap_max.unsqueeze(1).cuda() - reg_input[:, :, 2]) / depthmap_range.unsqueeze(1).cuda()
		reg_input = reg_input.view(reg_input.size(0), -1, 1, 1)
		self.task_outputs['depthmap'] = self.networks['DepthRegularizer'](reg_input).squeeze()
		self.task_loss = self.TaskLoss()

		self.task_outputs = to_cpu(self.task_outputs)

	def backward_D(self):
		# Fake
		# stop backprop to the generator by detaching fake_B
		fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
		pred_fake = self.netD(fake_AB.detach())
		self.loss_D_fake = self.criterionGAN(pred_fake, False)
		
		# Real
		real_AB = torch.cat((self.real_A, self.real_B), 1)
		pred_real = self.netD(real_AB)
		self.loss_D_real = self.criterionGAN(pred_real, True)

		# Combined loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * .5

		self.loss_D.backward()

	def backward_G(self):
		# First, G(A) should fake the discriminator
		fake_AB = torch.cat((self.real_A, self.fake_B), 1)
		pred_fake = self.netD(fake_AB)
		self.loss_G_GAN = self.criterionGAN(pred_fake, True)

		# Second, G(A) = B
		# self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) #* self.opt.lambda_A
		self.loss_G_L1 = 0

		#Third Task Loss        
		self.loss_task = self.TaskLoss()

		self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_task * 10

		self.loss_G.backward()

	def step(self):
		self.forward()
		# update D
		self.set_requires_grad(self.netD, True) # Input of D has been detached
		self.optimizer_D.zero_grad()
		self.backward_D()
		self.optimizer_D.step()

		# update G
		self.set_requires_grad(self.netD, False)
		self.optimizer_G.zero_grad()
		self.optimizer_Task.zero_grad()
		self.backward_G()
		self.optimizer_G.step()
		self.optimizer_Task.step()

	def eval_batch_result(self, type):
		
		return {"dis2d": self.batch_result['dis2d'].mean(), 
				"dis3d": self.batch_result['dis3d'].mean(),
				"loss_task": self.loss_task,
				"loss_G_GAN": self.loss_G_GAN,
				"loss_G_L1": self.loss_G_L1,
				"loss_D_real": self.loss_D_real,
				"loss_D_fake": self.loss_D_fake}

	def get_batch_result(self, type):
		self.task_out = to_cpu(self.task_out)
		preds2d = get_preds_from_heatmap(self.task_out['color_hm'][-1])
		# preds2d = get_preds_from_heatmap(self.batch['heatmap'])
		preds3d = torch.zeros((preds2d.size(0), 21, 3))

		root_depth = self.batch['root_depth']
		index_bone_length = self.batch['index_bone_length']

		preds3d[:,:,:2] = preds2d.clone()
		preds3d[:,:,2]  = self.task_out['depth'] * index_bone_length.unsqueeze(1) + root_depth.unsqueeze(1)
		# preds3d[:,:,2]  = self.batch['relative_depth'] * index_bone_length.unsqueeze(1) + root_depth.unsqueeze(1)
		

		preds3d[:, :, :2] *= preds3d[:, :, 2:]

		for i in range(preds3d.size(0)):
			preds3d[i, :, :] = torch.matmul(preds3d[i, :, :], self.batch['matrix'][i].transpose(0,1))

		dis3d = torch.norm(self.batch['coor3d'] - self.batch_result['pose3d'], dim = -1)
		dis2d = torch.norm(self.batch['coor2d'][:, :, :2] - self.batch_result['pose2d'], dim = -1)

		self.batch_result = {'pose2d':preds2d, 'pose3d': preds3d, 'dis3d': dis3d, 'dis2d':dis2d}
		return self.batch_result

	def eval_epoch_result(self, type):
		dist = self.epoch_result['dis3d'].numpy() * 1000
		x = np.sort(dist)
		y = np.array([1. * (i + 1)/ len(x)  for i in range(len(x))])
		AUC20_50 = calc_auc(x, y, 20, 50)
		return {
			'AUC20_50': AUC20_50
		}