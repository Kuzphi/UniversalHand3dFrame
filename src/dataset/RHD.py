# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import scipy

import os
import cv2
import json
import pickle
import torch
import numpy as np
from easydict import EasyDict as edict

from src.dataset.BaseDataset import BaseDataset

__all__  = ['RHD']
class RHD(BaseDataset):

    def __init__(self, cfg):
        super(RHD, self).__init__()

    def _get_db(self):
        self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        return sorted(self.anno.keys())
        
    # def transforms(self, cfg, img, depth, coor2d, matrix):
    #     # resize
    #     if cfg.has_key('RESIZE'):
    #         xscale, yscale = 1. * cfg.RESIZE / img.size(1), 1. * cfg.RESIZE / img.size(2) 
    #         coor2d[:, 0] *= xscale
    #         coor2d[:, 1] *= yscale
    #         scale =[[xscale,    0,  0],
    #                 [0,    yscale,  0],
    #                 [0,         0,  1]]
    #         matrix = np.matmul(scale, matrix)

    #         img = resize(img, cfg.RESIZE, cfg.RESIZE)
    #         depth = depth.unsqueeze(0)
    #         depth = interpolate(depth, (256, 256), mode = 'bilinear', align_corners = True)[0,...]
            
            
    #     if self.is_train:
    #         # Color 
    #         if cfg.COLOR_NORISE:
    #             img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
    #             img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
    #             img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

    #     assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 256
    #     return img, depth, coor2d, matrix

    # def __getitem__(self, idx):
    #     name = self.db[idx]
    #     label = self.anno[name]

    #     image_path   = os.path.join(self.cfg.ROOT, 'color', name + '.png')
    #     img = load_image(image_path)# already / 255 with C * W * H
        
    #     depth_path = os.path.join(self.cfg.ROOT, 'depth', name + '.pickle')
    #     depthmap = pickle.load(open(depth_path)).unsqueeze(0)

    #     coor2d = label['project']
    #     assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 320
    #     matrix = label['K']
    #     #apply transforms into image and calculate cooresponding coor and camera instrict matrix
    #     if self.cfg.TRANSFORMS:
    #         img, depthmap, coor2d, matrix = self.transforms(self.cfg.TRANSFORMS, img, depthmap, coor2d, matrix)

    #     # if depthmap_max - depthmap_min < 1e-6:
    #     #     print(name, ": ", depthmap_max - depthmap_min)
    #     # depthmap = (depthmap.max() - depthmap) / (depthmap_max - depthmap_min)
    #     # print(depthmap)
        
    #     meta = edict({'name': name})
    #     isleft = name[-1] == 'L'

    #     coor2d[1:,:] = coor2d[1:,:].reshape(5,4,-1)[:,::-1,:].reshape(20, -1)
    #     coor2d = to_torch(np.array(coor2d))
    #     coor2d[:,:2] = coor2d[:,:2].long().float() 
        
    #     matrix = np.linalg.inv(matrix) #take the inversion of matrix

    #     coor3d = coor2d.clone()
    #     coor3d[:,:2] *= coor3d[:, 2:]
    #     coor3d = torch.matmul(coor3d, to_torch(matrix).transpose(0, 1))

    #     root_depth = coor2d[0, 2].clone()
    #     index_bone_length = torch.norm(coor3d[9,:] - coor3d[10,:])
    #     relative_depth = (coor2d[:,2] - root_depth) / index_bone_length

    #     depthmap *= float(2**16 - 1)
    #     depthmap = (depthmap - root_depth) / index_bone_length
    #     depthmap_max = depthmap.max()
    #     depthmap_min = depthmap.min()
    #     depthmap = (depthmap - depthmap_min) / (depthmap_max - depthmap_min)
        
    #     #corresponding depth position in depth map
    #     index = torch.tensor([i * img.size(1) * img.size(2) + coor2d[i,0].long() * img.size(1) + coor2d[i,1].long() for i in range(21)])

    #     assert index.max() < img.size(1) * img.size(2) * 21, 'Wrong Position'
    #     heatmap = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))
    #     depth   = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))

    #     for i in range(21):
    #         heatmap[i] = draw_heatmap(heatmap[i], coor2d[i], self.cfg.HEATMAP.SIGMA)
    #         depth[i]   = heatmap[i] * (coor2d[i, 2] - coor2d[0, 2]) / index_bone_length


    #     return {'input': {'img':img,
    #                       'depthmap': depthmap,
    #                       # 'hand_side': torch.tensor([isleft, 1 - isleft]).float(),                          
    #                       },
    #             'index': index, 
    #             'matrix': to_torch(matrix),
    #             'color_hm': heatmap,
    #             'depth_hm' :  depth,
    #             'depthmap': depthmap,
    #             'depthmap_max': depthmap_max,
    #             'depthmap_range': depthmap_max - depthmap_min,
    #             'coor3d': to_torch(coor3d),
    #             'coor2d': to_torch(coor2d),
    #             'root_depth': root_depth,
    #             'index_bone_length': index_bone_length,
    #             'relative_depth': relative_depth,
    #             'weight': 1,
    #             'meta': meta,
    #             }
    def __getitem__(self, idx):
        name = self.db[idx]
        label = self.anno[name]

        image_path   = os.path.join(self.cfg.ROOT, 'color', name + '.png')
        img = load_image(image_path)# already / 255 with C * W * H
        
        depth_path = os.path.join(self.cfg.ROOT, 'depth', name + '.pickle')
        depthmap = pickle.load(open(depth_path)).unsqueeze(0)

        coor2d = label['project']
        matrix = label['K']
        assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 320
        meta = edict({'name': name})

        coor2d[1:,:] = coor2d[1:,:].reshape(5,4,-1)[:,::-1,:].reshape(20, -1)
        coor2d = to_torch(np.array(coor2d))
        coor2d[:,:2] = coor2d[:,:2].long().float() 
        return {"img": img,
                "matrix": matrix,
                "coord2d": coor2d,
                "depthmap": depthmap,
                "meta": meta
                }

