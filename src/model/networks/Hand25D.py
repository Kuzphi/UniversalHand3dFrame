import torch.nn as nn
import torch

# print net.params
__all__ = ['openpose_pose','openpose_hand', 'openpose_face']

class Repeat(nn.Module):
	def __init__(self, num_joints):
		super(Repeat, self).__init__()
		self.conv1 = nn.Conv2d(128 + num_joints, 128, kernel_size = 7, padding =3)
		self.conv2 = nn.Conv2d(128, 128, kernel_size = 7, padding =3)
		self.conv3 = nn.Conv2d(128, 128, kernel_size = 7, padding =3)
		self.conv4 = nn.Conv2d(128, 128, kernel_size = 7, padding =3)
		self.conv5 = nn.Conv2d(128, 128, kernel_size = 7, padding =3)
		self.conv6 = nn.Conv2d(128, 128, kernel_size = 1, padding =0)
		self.conv7 = nn.Conv2d(128, num_joints,  kernel_size = 1, padding =0)
		self.relu  = nn.ReLU(inplace = True)
	def forward(self, x):		
		out = self.relu(self.conv1(x))
		out = self.relu(self.conv2(out))
		out = self.relu(self.conv3(out))
		out = self.relu(self.conv4(out))
		out = self.relu(self.conv5(out))
		out = self.relu(self.conv6(out))
		out = self.conv7(out)
		return out

class softmax2d(nn.Module):
	"""docstring for softmax2d"""
	def __init__(self, c, w, h):
		super(softmax2d, self).__init__()
		self.beta = nn.Parameter(torch.zeros((c,w,h)).uniform_(0, 1))
		self.softmax2d = nn.Softmax2d()

	def forward(self, x):
		out = x * self.beta.expand_as(x)
		out = self.softmax2d(out)
		return out
		

class OpenPose_CPM(nn.Module):
	def __init__(self, num_joints):
		super(OpenPose_CPM, self).__init__()
		self.pool    = nn.MaxPool2d(2, padding = 0)
		self.relu    = nn.ReLU(inplace = True)
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size = 3, padding =1)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size = 3, padding =1)

		self.conv2_1 = nn.Conv2d(64, 128, kernel_size = 3, padding =1)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size = 3, padding =1)

		self.conv3_1 = nn.Conv2d(128, 256, kernel_size = 3, padding =1)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size = 3, padding =1)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size = 3, padding =1)
		self.conv3_4 = nn.Conv2d(256, 256, kernel_size = 3, padding =1)

		self.conv4_1 = nn.Conv2d(256, 512, kernel_size = 3, padding =1)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size = 3, padding =1)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size = 3, padding =1)
		self.conv4_4 = nn.Conv2d(512, 512, kernel_size = 3, padding =1)
		
		self.conv5_1 = nn.Conv2d(512, 512, kernel_size = 3, padding =1)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size = 3, padding =1)
		self.conv5_3_CPM = nn.Conv2d(512, 128, kernel_size = 3, padding =1)

		self.conv6_1_CPM = nn.Conv2d(128, 512, kernel_size = 1, padding =0)
		self.conv6_2_CPM = nn.Conv2d(512, num_joints, kernel_size = 1, padding =0)

		self.stage2 = Repeat(num_joints)
		self.stage3 = Repeat(num_joints)
		self.stage4 = Repeat(num_joints)
		self.stage5 = Repeat(num_joints)
		self.stage6 = Repeat(num_joints)
		self.depth  = Repeat(num_joints)
		self.depth_fc_1 = nn.Linear(32 * 32 * 22, 5000)
		self.depth_fc_2 = nn.Linear(5000, 3000)
		self.depth_fc_3 = nn.Linear(3000, 21)

		self.softmax2d = softmax2d(21, 256, 256)
		self.upsampler = nn.functional.interpolate
		# self.upsampler = nn.Upsample(scale_factor = 8, mode = 'bilinear', align_corners = True)

	def forward(self, x):
		out = self.relu(self.conv1_1(x['img']))
		out = self.relu(self.conv1_2(out))
		out = self.pool(out)

		out = self.relu(self.conv2_1(out))
		out = self.relu(self.conv2_2(out))
		out = self.pool(out)
		
		out = self.relu(self.conv3_1(out))
		out = self.relu(self.conv3_2(out))
		out = self.relu(self.conv3_3(out))
		out = self.relu(self.conv3_4(out))
		out = self.pool(out)

		out = self.relu(self.conv4_1(out))
		out = self.relu(self.conv4_2(out))
		out = self.relu(self.conv4_3(out))
		out = self.relu(self.conv4_4(out))

		out = self.relu(self.conv5_1(out))
		out = self.relu(self.conv5_2(out))
		out_0 = self.relu(self.conv5_3_CPM(out))

		out_1 = self.relu(self.conv6_1_CPM(out_0))
		out_1 = self.conv6_2_CPM(out_1)

		out_2 = torch.cat((out_1, out_0), 1)
		out_2 = self.stage2(out_2)

		out_3 = torch.cat((out_2, out_0), 1)
		out_3 = self.stage3(out_3)

		out_4 = torch.cat((out_3, out_0), 1)
		out_4 = self.stage4(out_4)

		out_5 = torch.cat((out_4, out_0), 1)
		out_5 = self.stage5(out_5)

		out_6 = torch.cat((out_5, out_0), 1)
		out_6 = self.stage6(out_6)


		outputs = [out_1, out_2, out_3, out_4, out_5, out_6]
		outputs = [self.upsampler(out, scale_factor = 8, mode = 'bilinear', align_corners = True) for out in outputs]

		heatmap = [out[:, 21:, ...] for out in outputs]
		depth   = [out[:, :21, ...] for out in outputs]
		return {'heatmap': heatmap, 'depth': depth}
		#stage 2
		# out = self.upsampler(out_6, scale_factor = 8, mode = 'bilinear', align_corners = True)

		# heatmap = self.softmax2d(out[:, :21, ...])
		# depth   = out[:, 21:, ...] * heatmap

		# s = heatmap.shape

		# x = heatmap.sum(dim = 3)
		# weight = torch.arange(s[2]).view(1,1,-1).expand_as(x).float().cuda()
		# coorX = (x * weight).sum(dim = 2, keepdim = True)

		# y = heatmap.sum(dim = 2)
		# weight = torch.arange(s[3]).view(1,1,-1).expand_as(y).float().cuda()
		# coorY = (y * weight).sum(dim = 2, keepdim = True)

		# coorZ   = depth.sum(dim = (2,3), keepdim = True)[...,0]

		# coor = torch.cat((coorX, coorY, coorZ), dim = -1)
		# return {'coor3d': coor, 'coor2d': coor[:,:,:2]}

def Hand25D(num_joints = 21, **kwargs):
	return OpenPose_CPM(num_joints * 2)