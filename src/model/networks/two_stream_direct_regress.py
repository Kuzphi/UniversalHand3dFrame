import torch.nn as nn
import torch

# print net.params
# __all__ = ['openpose_pose','openpose_hand', 'openpose_face']

class Repeat(nn.Module):
	def __init__(self, num_joints = 21, input_channel = None):
		super(Repeat, self).__init__()
		if input_channel is None:
			input_channel = 128 + num_joints
		self.conv1 = nn.Conv2d(input_channel, 128, kernel_size = 7, padding =3)
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

class OpenPose_CPM(nn.Module):
	def __init__(self, input_channel, num_joints):
		super(OpenPose_CPM, self).__init__()
		self.pool    = nn.MaxPool2d(2, padding = 0)
		self.relu    = nn.ReLU(inplace = True)
		self.conv1_1 = nn.Conv2d(input_channel, 64, kernel_size = 3, padding =1)
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

	def forward(self, x):
		out = self.relu(self.conv1_1(x))
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
		out_6 = self.stage5(out_6)

		outputs = [out_1, out_2, out_3, out_4, out_5, out_6]

		return outputs
class Two_stream_fusion(nn.Module):
	"""docstring for Two_stream_fusion"""
	def __init__(self, num_joints = 22):
		super(Two_stream_fusion, self).__init__()
		self.depth_net = OpenPose_CPM(input_channel = 1, num_joints = num_joints)
		self.color_net = OpenPose_CPM(input_channel = 3, num_joints = num_joints)
		self.depth_brach = Repeat(num_joints, input_channel = 44)
		self.color_brach = Repeat(num_joints, input_channel = 44)
		self.depth_fc_1 = nn.Linear(32 * 32 * 22, 512)
		self.depth_fc_2 = nn.Linear(512, 512)
		self.depth_fc_3 = nn.Linear(512, 21)
		self.upsampler = nn.functional.interpolate

	def forward(self, x):
		color = self.color_net(x['img'])
		depth = self.depth_net(x['depthmap'])

		whole = torch.cat((color[-1], depth[-1]), dim = 1)

		# color_hm = self.color_brach(whole)
		depth_hm = self.depth_brach(whole)

		out_depth = depth_hm.view(-1, 32 * 32 * 22)

		out_depth = self.depth_fc_1(out_depth)
		out_depth = self.depth_fc_2(out_depth)
		out_depth = self.depth_fc_3(out_depth)

		# color = color + [color_hm]
		depth = depth + [depth_hm]

		color = [self.upsampler(hm, scale_factor = 8, mode = 'bilinear', align_corners = True) for hm in color]
		depth = [self.upsampler(hm, scale_factor = 8, mode = 'bilinear', align_corners = True) for hm in depth]
		return {
				'color_hm': color,
				'depth_hm': depth,
				'depth':out_depth
		}
		
def CPMWeaklyDirectRegression(num_joints = 21, **kwargs):
	return OpenPose_CPM(num_joints)

def two_stream_fusion(num_joints = 22, ** kwargs):
	return Two_stream_fusion(num_joints = num_joints)