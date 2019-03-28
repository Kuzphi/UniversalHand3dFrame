import torch
import itertools
import numpy as np

from torch import nn

from src.model import BaseModel
from src.core.evaluate import get_preds_from_heatmap
from src.utils.misc import to_cpu
from src.utils.image_pool import ImagePool
from src.core.evaluate import AUC, calc_auc
class Weakly_GAN(BaseModel):

	def __init__(self, cfg):
		super(depth_regularizer, self).__init__(cfg)
        self.netD = self.networks['discriminator']
        self.netG = self.networks['generator']
        self.optimizer_D = self.optimizers['D']
        self.optimizer_G = self.optimizers['G']

    def TaskLoss(self):
        loss = torch.zeros(1).cuda()
        criterion = nn.MSELoss()
        for hm in self.task_out['color_hm']:
            loss += criterion(hm, self.batch['color_hm'].cuda())

        loss += nn.functional.smooth_l1_loss(self.task_out['depth'], self.batch['relative_depth'].cuda())
        return loss

    def set_input(self, batch):
        # self.batch = batch
        self.batch = batch
        self.real_A = batch['input']['img'].cuda()
        self.real_B = batch['input']['depthmap'].cuda()


    def forward(self):
        self.fake_B = self.netG(self.real_A)
        task_input = {'img':self.real_A, 'depthmap': self.fake_B}
        self.task_out = self.TaskNet(task_input)
        

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

    def batch_result(self, type):
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