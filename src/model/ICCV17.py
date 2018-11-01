from __future__ import division

import torch.nn as nn
import torch
import torch.nn.functional as F
# print net.params
__all__ = ['ICCV17']

class Repeat(nn.Module):
	def __init__(self, channels, kernels, paddings = None, out = False, fg = False):
		super(Repeat, self).__init__()
		assert len(channels) == len(kernels) + 1, 'number of conv is not consistent'
		self.out = out
		self.fg = fg 
		if paddings == None:
			paddings = [ks // 2 for ks in kernels]

		self.convs = torch.nn.ModuleList()
		for i in range(len(kernels)):
			params = { 	'in_channels':channels[i],
						'out_channels':channels[i + 1], 
						'kernel_size':kernels[i],
						'padding':paddings[i]}

			self.convs.append(nn.Conv2d( **params))

		self.relu  = nn.functional.leaky_relu

	def forward(self, x):
		for idx in range(len(self.convs)):			
			x = self.convs[idx](x)
			if not self.out or idx < len(self.convs) - 1:
				x = self.relu(x)
			# if self.fg:
			# 	print("!!!!", x.sum(), x.size())
		return x

class PoseNet(nn.Module):
	"""docstring for PoseNet"""
	def __init__(self, num_class, **kwargs):
		super(PoseNet, self).__init__()
		self.pool    = nn.MaxPool2d(2, padding = 0)
		self.relu    = nn.functional.leaky_relu
		self.stage1_block_1 = Repeat([3,64,64], [3,3])
		self.stage1_block_2 = Repeat([64,128,128], [3,3])
		self.stage1_block_3 = Repeat([128,256,256,256,256], [3,3,3,3])
		self.stage1_block_4 = Repeat([256,512,512,256,256,256,256,128], [3,3,3,3,3,3,3])
		self.stage1_block_5 = Repeat([128,512,num_class],[1, 1], out = True)

		self.stage2 		= Repeat([149,128,128,128,128,128,128,num_class], [7,7,7,7,7,1,1], out = True, fg = True) # cat x and out1 128 + 21 = 149
		self.stage3			= Repeat([149,128,128,128,128,128,128,num_class], [7,7,7,7,7,1,1], out = True) # cat x and out2 128 + 21 = 149
	def forward(self, x):
		x = self.stage1_block_1(x)
		x = self.pool(x)
		x = self.stage1_block_2(x)
		x = self.pool(x)
		x = self.stage1_block_3(x)
		x = self.pool(x)
		x = self.stage1_block_4(x)
		out1 = x.clone() #need creating a new copy of x, otherwise x would be change in next line!!
		out1 = self.stage1_block_5(out1)
		out2 = torch.cat([out1, x], 1)
		out2 = self.stage2(out2)
		out3 = torch.cat([out2, x], 1)
		out3 = self.stage3(out3)

		return [out1, out2, out3]

class PosePior(nn.Module):
	"""docstring for PosePior"""
	def __init__(self, num_joints):
		super(PosePior, self).__init__()
		self.num_joints = num_joints
		self.relu  = nn.functional.leaky_relu
		self.conv_0_1 = nn.Conv2d(21,  32, 3, padding = 1)
		self.conv_0_2 = nn.Conv2d(32,  32, 3, stride = 2)
		self.conv_1_1 = nn.Conv2d(32,  64, 3, padding = 1)
		self.conv_1_2 = nn.Conv2d(64,  64, 3, stride = 2)
		self.conv_2_1 = nn.Conv2d(64, 128, 3, padding = 1)
		self.conv_2_2 = nn.Conv2d(128, 128, 3, stride = 2)

		self.fc_0 	  = nn.Linear(2050, 512) # 4*4*128 + 2 = 2050
		self.fc_1 	  = nn.Linear(512, 512)
		self.fc_out   = nn.Linear(512, 3 * num_joints)

	def _flip_right_hand(self, coords_xyz_canonical, hand_side):
		""" Flips the given canonical coordinates, when cond_right is true. Returns coords unchanged otherwise.
			The returned coordinates represent those of a left hand.

			Inputs:
				coords_xyz_canonical: Nx3 matrix, containing the coordinates for each of the N keypoints
		"""

		# flip hand according to hand side
		
		cond_right = torch.argmax(hand_side, 1) == 1
		cond_right = cond_right.view(-1, 1, 1).repeat(1, self.num_joints, 3)

		expanded = False
		s = coords_xyz_canonical.size()
		if len(s) == 2:
			coords_xyz_canonical.unsqueeze_(0)
			cond_right.unsqueeze_(0)
			expanded = True

		# mirror along y axis
		coords_xyz_canonical_mirrored = torch.stack([coords_xyz_canonical[:, :, 0], coords_xyz_canonical[:, :, 1], -coords_xyz_canonical[:, :, 2]], 2)

		# select mirrored in case it was a right hand
		coords_xyz_canonical_left = torch.where(cond_right, coords_xyz_canonical_mirrored, coords_xyz_canonical)

		if expanded:
			coords_xyz_canonical_left = torch.squeeze(coords_xyz_canonical_left, 0)

		return coords_xyz_canonical_left

	def forward(self, x, hand_side):
		out = self.relu(self.conv_0_1(x))
		out = F.pad(out, [0, 1, 0, 1])
		out = self.relu(self.conv_0_2(out))

		out = self.relu(self.conv_1_1(out))
		out = F.pad(out, [0, 1, 0, 1])
		out = self.relu(self.conv_1_2(out))

		out = self.relu(self.conv_2_1(out))
		out = F.pad(out, [0, 1, 0, 1])
		out = self.relu(self.conv_2_2(out))

		out = out.contiguous().permute(0,2,3,1)
		out = out.contiguous().view(out.size(0), -1)
		out = torch.cat([out, hand_side], 1)

		out = self.relu(self.fc_0(out))
		out = self.relu(self.fc_1(out))
		out = self.fc_out(out)
		out = out.view(out.size(0), self.num_joints, 3)
		out = self._flip_right_hand(out, hand_side)
		return out

class ViewPoint(nn.Module):
	"""docstring for ViewPoint"""
	def __init__(self):
		super(ViewPoint, self).__init__()
		self.relu  = nn.functional.leaky_relu
		self.conv_0_1 = nn.Conv2d(21,  64, 3, padding = 1)
		self.conv_0_2 = nn.Conv2d(64,  64, 3, stride = 2)
		self.conv_1_1 = nn.Conv2d(64,  128, 3, padding = 1)
		self.conv_1_2 = nn.Conv2d(128, 128, 3, stride = 2)
		self.conv_2_1 = nn.Conv2d(128, 256, 3, padding = 1)
		self.conv_2_2 = nn.Conv2d(256, 256, 3, stride = 2)
		self.fc_0 	  = nn.Linear(4098, 256)
		self.fc_1 	  = nn.Linear(256, 128)
		self.fc_ux    = nn.Linear(128, 1)
		self.fc_uy    = nn.Linear(128, 1)
		self.fc_uz    = nn.Linear(128, 1)

	def _get_rot_mat(self, ux_b, uy_b, uz_b):
		""" Returns a rotation matrix from axis and (encoded) angle."""

		u_norm = torch.sqrt(ux_b ** 2 + uy_b ** 2 + uz_b ** 2 + 1e-8)
		theta = u_norm

		# some tmp vars
		st_b = torch.sin(theta)
		ct_b = torch.cos(theta)
		one_ct_b = 1.0 - torch.cos(theta)

		st = st_b[:, 0]
		ct = ct_b[:, 0]
		one_ct = one_ct_b[:, 0]
		norm_fac = 1.0 / u_norm[:, 0]
		ux = ux_b[:, 0] * norm_fac
		uy = uy_b[:, 0] * norm_fac
		uz = uz_b[:, 0] * norm_fac
		l1 = torch.stack([ct+ux*ux*one_ct, ux*uy*one_ct-uz*st, ux*uz*one_ct+uy*st], dim = -1)
		l2 = torch.stack([uy*ux*one_ct+uz*st, ct+uy*uy*one_ct, uy*uz*one_ct-ux*st], dim = -1)
		l3 = torch.stack([uz*ux*one_ct-uy*st, uz*uy*one_ct+ux*st, ct+uz*uz*one_ct], dim = -1)
		mat = torch.stack([l1,l2,l3],dim = -1)
		mat = mat.permute(0,2,1)
		return mat

	def forward(self, x, hand_side):
		
		out = self.relu(self.conv_0_1(x))
		out = F.pad(out, [0, 1, 0, 1])
		out = self.relu(self.conv_0_2(out))

		out = self.relu(self.conv_1_1(out))
		out = F.pad(out, [0, 1, 0, 1])
		out = self.relu(self.conv_1_2(out))

		out = self.relu(self.conv_2_1(out))
		out = F.pad(out, [0, 1, 0, 1])
		out = self.relu(self.conv_2_2(out))

		out = out.contiguous().permute(0,2,3,1)
		out = out.contiguous().view(out.size(0), -1)
		out = torch.cat([out, hand_side], 1)

		out = self.relu(self.fc_0(out))
		out = self.relu(self.fc_1(out))
		ux = self.fc_ux(out)
		uy = self.fc_uy(out)
		uz = self.fc_uz(out)
		return self._get_rot_mat(ux, uy, uz)

class ICCV17(nn.Module):
	def __init__(self, num_joints, **kwargs):
		super(ICCV17, self).__init__()
		self.pose_net = PoseNet(num_joints)
		self.pose_pior = PosePior(num_joints)
		self.view_point = ViewPoint()

	def forward(self, x):
		img = x['img']
		hand_side = x['hand_side']
		heatmap = self.pose_net(img)
		# print ("heatmap", heatmap[-1].shape, heatmap[-1].sum())
		pose_can = self.pose_pior(heatmap[-1], hand_side)
		rotate_mat = self.view_point(heatmap[-1], hand_side)
		# print(pose_can)
		# print(rotate_mat)
		out = torch.matmul(pose_can, rotate_mat)
		# print (torch.matmul(pose_can[0], rotate_mat[0]))
		return {'pose3d' : out, 
				'heatmap': heatmap}