import sys
sys.path.append('../')

from src.model import ICCV17
from src.utils.imutils import plot_hand_3d,load_image,im_to_numpy
import torch
import scipy
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json,os

names = ['Model1ArmRotate', 'Model1Fist', 'Model1Tap', 'Model1WristRotate']
idx = {
    names[0]: [381, 595,738],
    names[1]: [204, 1479, 2019],
    names[2]: [6,745,12428],
    names[3]: [42,47,554,736]
}
cnt= 0

for name in names:
    for id in idx[name]:
        # fig = plt.figure(1)
        # ax = fig.add_subplot(1,1,1, projection = '3d')
        # ax.axis('off')
        imgpath = '/data/liangjianData/TencentHand/Model1/%s/image/%s'%(name, str(id).zfill(7) + '.png')    
        os.system('cp %s %s'%(imgpath, './' + 'green_' + str(cnt).zfill(3)+'.png'))
        cnt += 1