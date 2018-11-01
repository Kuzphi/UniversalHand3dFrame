
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils.imutils import batch_with_heatmap

def debug(outputs, batch):
    gt_win, pred_win = None, None
    # gt_batch_img   = batch_with_heatmap(inputs = batch['inputs']['img'], outputs = batch['heat_map'])
    pred_batch_img = batch_with_heatmap(inputs = batch['input']['img'], outputs = outputs['heatmap'][-1])
    
    # ax1 = plt.subplot(121)
    # ax1.title.set_text('Groundtruth')
    # gt_win = plt.imshow(gt_batch_img)
    ax2 = plt.subplot(111)
    ax2.title.set_text('Prediction')
    pred_win = plt.imshow(pred_batch_img)
    plt.show()