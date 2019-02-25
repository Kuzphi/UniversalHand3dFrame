# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from src.utils.misc import to_torch



def DistanceLoss(outputs, batch):
	gt_coor = batch['coor']
	pred_coor = outputs['pose3d'].cpu() * batch['index_bone_length'].view(-1,1,1).repeat(1,21,3)

	dis = torch.norm(gt_coor - pred_coor, dim = -1)
	dis = torch.mean(dis) 
	return dis

def CPMMSELoss(outputs, batch):
	criterion = nn.MSELoss(size_average=True)
	loss = torch.zeros(1).cuda()
	target = batch['heatmap'].cuda()
	for pred in outputs['heatmap']:
		loss += criterion(pred, target)
	return loss

def get_preds_from_heatmap(scoremaps, softmax = False):
    """ Performs detection per scoremap for the hands keypoints. """
    s = scoremaps.shape
    assert len(s) == 4, "This function was only designed for 4D Scoremaps(B * C * H * W)."
    # assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    keypoint_coords = torch.zeros((s[0], s[1], 2))
    keypoint_coords = keypoint_coords.cuda()
    x = scoremaps.sum(dim = 3)
    weight = torch.arange(s[2]).view(1,1,-1).expand_as(x).float().cuda()
    keypoint_coords[:,:,1] = torch.sum(x * weight, dim = 2) / x.sum(dim = 2)

    y = scoremaps.sum(dim = 2)
    weight = torch.arange(s[3]).view(1,1,-1).expand_as(y).float().cuda()
    keypoint_coords[:,:,0] = torch.sum(y * weight, dim = 2) / y.sum(dim = 2)
    return keypoint_coords[:,:21,:]

def DistanceLoss2D(outputs, batch):
	preds = get_preds_from_heatmap(outputs['heatmap'][-1], softmax = True)
	diff = batch['coor'].cuda() - preds
	dis = torch.norm(diff, dim = -1).mean()
	return dis

# class JointsMSELoss(nn.Module):
#     def __init__(self, use_target_weight):
#         super(JointsMSELoss, self).__init__()
#         self.criterion = nn.MSELoss(size_average=True)
#         self.use_target_weight = use_target_weight

#     def forward(self, output, target, target_weight):
#         batch_size = output.size(0)
#         num_joints = output.size(1)
#         heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
#         heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
#         loss = 0

#         for idx in range(num_joints):
#             heatmap_pred = heatmaps_pred[idx].squeeze()
#             heatmap_gt = heatmaps_gt[idx].squeeze()
#             if self.use_target_weight:
#                 loss += 0.5 * self.criterion(
#                     heatmap_pred.mul(target_weight[:, idx]),
#                     heatmap_gt.mul(target_weight[:, idx])
#                 )
#             else:
#                 loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

#         return loss / num_joints
