import sys
sys.path.append('../')

from src.model import openpose_hand
from src.utils.imutils import plot_hand,load_image,im_to_numpy
import torch
import scipy
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.core.evaluate import get_preds_from_heatmap
import os

def get_model(weight_path):
	weight = torch.load(weight_path)
	if weight.has_key('state_dict'):
		weight = weight['state_dict']
	model = openpose_hand(num_joints = 22)	
	model = torch.nn.DataParallel(model).cuda().float()
	model.load_state_dict(weight)
	model.eval()
	return model

model = get_model('../pretrained_weight/openpose_hand.torch')

imgpath = '/data/liangjian_hand/train'
savepath = '/data/liangjian_hand/crop'
for name in os.listdir(imgpath):
	path = os.path.join(imgpath, name)
	OriImg = scipy.misc.imread(path)
	img = load_image(path, mode = 'GBR')
	img = img.view(-1, img.size(0), img.size(1), img.size(2))
	output = model({'img':img})
	coor = get_preds_from_heatmap(output['heatmap'][-1])
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	plot_hand(OriImg, coor, ax)
	plt.show()

