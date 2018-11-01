
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils.imutils import batch_with_heatmap, plot_hand_3d, im_to_numpy

# def debug(outputs, batch):
#     gt_win, pred_win = None, None
#     print(outputs['pose3d'][:4])
#     print(batch['coor'][:4])
#     # gt_batch_img   = batch_with_heatmap(inputs = batch['input']['img'], outputs = batch['heat_map'])
#     pred_batch_img = batch_with_heatmap(inputs = batch['input']['img'], outputs = outputs['heatmap'][-1])
    
#     # ax1 = plt.subplot(121)
#     # ax1.title.set_text('Groundtruth')
#     # gt_win = plt.imshow(gt_batch_img)
#     ax2 = plt.subplot(111)
#     ax2.title.set_text('Prediction')
#     pred_win = plt.imshow(pred_batch_img)
#     plt.show()

def debug(outputs, batch):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, projection = '3d')
    ax3 = fig.add_subplot(133, projection = '3d')
    img = batch['input']['img'][0]
    img = im_to_numpy(img)
    img = ((img + 0.5) * 255).astype(np.int)
    print(img.shape, img.sum())
    ax1.imshow(img)
    plot_hand_3d(outputs['pose3d'][0] / 2., ax2)
    plot_hand_3d(batch['coor'][0], ax3)
    plt.show()
