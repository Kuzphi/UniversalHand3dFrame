import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
sys.path.append("..")
from src.models.OpenPose import openpose_hand
#parser = argparse.ArgumentParser()
#parser.add_argument('--t7_file', required=True)
#parser.add_argument('--pth_file', required=True)
#args = parser.parse_args()

torch.set_num_threads(torch.get_num_threads())
# weight_name = '../model/OpenPose/Hand/hand_model.pth'
weight_name = '../model/OpenPose/Hand/openpose_hand.torch'

test_image = 'data/RHD/evaluation/cropped_bigger/00000_1.png'

# visualize
colors = [
		[100.,  100.,  100.], 
		[100.,    0.,    0.],
		[150.,    0.,    0.],
		[200.,    0.,    0.],
		[255.,    0.,    0.],
		[100.,  100.,    0.],
		[150.,  150.,    0.],
		[200.,  200.,    0.],
		[255.,  255.,    0.],
		[  0.,  100.,   50.],
		[  0.,  150.,   75.],
		[  0.,  200.,  100.],
		[  0.,  255.,  125.],
		[  0.,   50.,  100.],
		[  0.,   75.,  150.],
		[  0.,  100.,  200.],
		[  0.,  125.,  255.],
		[100.,    0.,  100.],
		[150.,    0.,  150.],
		[200.,    0.,  200.],
		[255.,    0.,  255.]]
def Hand_Inference(oriImg, Model = None, Name = ""):
	num_classes = 22
	if Model == None:
		model = openpose_hand(num_classes = num_classes)
		model.load_state_dict(torch.load(weight_name))
		model = torch.nn.DataParallel(model).cuda().float()
		model.eval()
	else:
		model = Model
	param_, model_ = config_reader('config_hand')

	#torch.nn.functional.pad(img pad, mode='constant', value=model_['padValue'])
	# tic = time.time()

	#test_image = 'a.jpg'
	with torch.no_grad():
		# imageToTest = Variable(T.unsqueeze(torch.from_numpy(oriImg.transpose(2,0,1)).float(),0)).cuda()
		# multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]
		heatmap_avg = torch.zeros(1,num_classes,oriImg.shape[0], oriImg.shape[1]).cuda()

	imageToTest_padded = oriImg[:,:,:,np.newaxis].transpose(3,2,0,1).astype(np.float32) / 255.0 - 0.5
	
	with torch.no_grad():		
		feed = T.from_numpy(imageToTest_padded).cuda()
		print("img:", feed.sum())
		output2 = model({'img':feed})['heatmap'][-1]
	print(output2.size(), output2.sum()) 
	heatmap = output2
	heatmap_avg[m] = heatmap[0].data

	heatmap_avg = T.mean(heatmap_avg, 0)
	heatmap_avg = T.transpose(T.transpose(T.squeeze(heatmap_avg),0,1),1,2).cuda() 
	heatmap_avg = heatmap_avg.cpu().numpy()

	# toc =time.time()
	# print 'time is %.5f'%(toc-tic) 
	# tic = time.time()

	all_peaks = []
	peak_counter = 0

	#maps = 
	keypoint_coords = np.zeros((21, 2))
	for part in range(21):
		map_ori = heatmap_avg[:,:,part]
		plt.imshow(oriImg[:,:,[2,1,0]])
		plt.imshow(heatmap_avg[:,:,part], alpha=.3)
		plt.savefig("demo_heat/test_part_"+ str(part) + ".png")
		# plt.savefig("demo_heat/test_part_"+ str(part) +"for" + '_'.join(Name.split('/')[-1].split(".")) + ".png")

		v, u = np.unravel_index(np.argmax(heatmap_avg[:, :, part]), (256, 256))
		keypoint_coords[part, 0] = u
		keypoint_coords[part, 1] = v

	return keypoint_coords


if __name__ == '__main__':
	import pickle
	# split = pickle.load(open('../data/RenderHand/split.pickle'))
	# split.Data_Path = '/home/liangjic/data/RenderHand/'
	plt.figure()
	# oriImg = cv2.imread(test_image) # B,G,R order
	# joint = Hand_Inference(oriImg)
	# canvas = cv2.imread(test_image) # B,G,R order
	model = openpose_hand(num_classes = 22)
	model = torch.nn.DataParallel(model).cuda().float()
	model.load_state_dict(torch.load(weight_name))
	# model = model.cuda()
	oriImg = cv2.imread(test_image)
	print (oriImg.shape, oriImg.sum())
	# oriImg = cv2.resize(oriImg,(256,256))

	joint = Hand_Inference(oriImg, Model = model)
	canvas = oriImg.copy()
	canvas = draw_hand(canvas, joint.astype(np.int), Edge = True)
	plt.imshow(canvas[:,:,::-1])
	plt.pause(10)
	# for i in split.valid:
	# 	print i
	# 	oriImg, label = split.get_sample(i)
	# 	canvas = oriImg.copy()
	# 	canvas = draw_hand(canvas, label)
	# 	plt.subplot(1,2,1)
	# 	plt.imshow(canvas[:,:,::-1])

	# 	joint = Hand_Inference(oriImg, Model = model, Name = str(i))
	# 	# print joint
	# 	joint = np.array(joint).astype(np.int)
	# 	canvas = oriImg.copy()
	# 	canvas = draw_hand(canvas, joint, Edge = True)
	# 	plt.subplot(1,2,2)
	# 	plt.imshow(canvas[:,:,::-1])
	# 	plt.pause(1)
	cv2.imwrite('result.png',canvas)