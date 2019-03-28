import sys
import torch
import pickle
sys.path.append('..')
from src.model.networks.Hand25D import Hand25D
# from src.model.OpenPose import openpose_hand
# model = openpose_hand(num_joints = 22)

# print model.state_dict().keys()
x = torch.load("openpose_hand.torch")

for stage in ['stage2', 'stage3', 'stage4', 'stage5', 'stage6']:
	new_weight = torch.zeros(128, 20, 7, 7)
	# torch.nn.init.xavier_uniform_(new_weight)

	weight = x['module.%s.conv1.weight' % stage]
	# print (weight.size(), new_weight.size())
	weight = torch.cat((weight, new_weight), 1)
	x['module.%s.conv1.weight' % stage] = weight
	########################################################################
	# x['module.%s.conv1.bias']
	new_weight = torch.zeros(20, 128, 1, 1)
	# torch.nn.init.xavier_uniform_(new_weight)

	weight = x['module.%s.conv7.weight' % stage]
	# print (weight.size(), new_weight.size())
	weight = torch.cat((weight, new_weight), 0)
	x['module.%s.conv7.weight' % stage] = weight
	########################################################################
	new_bias = torch.zeros(20)
	bias = x['module.%s.conv7.bias'%stage]
	bias = torch.cat((bias, new_bias))
	x['module.%s.conv7.bias'%stage] = bias


new_weight = torch.zeros(20, 512, 1, 1)
# torch.nn.init.xavier_uniform_(new_weight)

weight = x['module.conv6_2_CPM.weight']
weight = torch.cat((weight, new_weight), 0)
x['module.conv6_2_CPM.weight'] = weight

new_bias = torch.zeros(20)
bias = x['module.conv6_2_CPM.bias']
bias = torch.cat((bias, new_bias))
x['module.conv6_2_CPM.bias'] = bias


model = Hand25D(num_joints = 21)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
model.load_state_dict(x)
torch.save(x, "openpose_hand_with_depth.torch")