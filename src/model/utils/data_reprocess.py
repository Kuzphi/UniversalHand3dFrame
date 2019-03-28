# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        depth = interpolate(depth, (cfg.RESIZE, cfg.RESIZE), mode = 'bilinear', align_corners = True)[0,...]
        
        
    if self.is_train:
        # Color 
        if cfg.COLOR_NORISE:
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

    assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 256
    return img, depth, coor2d, matrix

def reprocess(input):
    #apply transforms into image and calculate cooresponding coor and camera instrict matrix
    if self.cfg.TRANSFORMS:
        img, depthmap, coor2d, matrix = self.transforms(self.cfg.TRANSFORMS, img, depthmap, coor2d, matrix)

    
    isleft = name[-1] == 'L'
    
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
    
    #corresponding depth position in depth map
    index = torch.tensor([i * img.size(1) * img.size(2) + coor2d[i,0].long() * img.size(1) + coor2d[i,1].long() for i in range(21)])

    assert index.max() < img.size(1) * img.size(2) * 21, 'Wrong Position'
    heatmap = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))
    depth   = torch.zeros(self.cfg.NUM_JOINTS, img.size(1), img.size(2))

    for i in range(21):
        heatmap[i] = draw_heatmap(heatmap[i], coor2d[i], self.cfg.HEATMAP.SIGMA)
        depth[i]   = heatmap[i] * (coor2d[i, 2] - coor2d[0, 2]) / index_bone_length
