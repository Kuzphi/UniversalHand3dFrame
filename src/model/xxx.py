import torch.nn as nn
import torch

# print net.params
__all__ = ['renderhand']

class Repeat(nn.Module):
	def __init__(self, channel, num_class):
		super(Repeat, self).__init__()
		self.conv1 = nn.Conv2d(channel, 128, kernel_size = 7, padding =3)
		self.conv2 = nn.Conv2d(128, 128, kernel_size = 7, padding = 3)
		self.conv3 = nn.Conv2d(128, 128, kernel_size = 7, padding = 3)
		self.conv4 = nn.Conv2d(128, 128, kernel_size = 7, padding = 3)
		self.conv5 = nn.Conv2d(128, 128, kernel_size = 7, padding = 3)
		self.conv6 = nn.Conv2d(128, 128, kernel_size = 1, padding = 0)
		self.conv7 = nn.Conv2d(128, num_class, kernel_size = 1, padding = 0)
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

# class RefineNet(nn.Module):
# 	def __init__(self, num_class):
# 		super(RefineNet, self).__init__()
# 		self.conv1 = nn.Conv2d(  4, 128, kernel_size = 7, padding = 3)
# 		self.conv2 = nn.Conv2d(128, 128, kernel_size = 7, padding = 3)
# 		self.conv3 = nn.Conv2d(128, 128, kernel_size = 7, padding = 3)
# 		self.conv4 = nn.Conv2d(128, 128, kernel_size = 7, padding = 3)
# 		self.conv5 = nn.Conv2d(128, 128, kernel_size = 7, padding = 3)
# 		self.conv6 = nn.Conv2d(128,   4, kernel_size = 1, padding = 0)
# 		self.relu  = nn.ReLU(inplace = True)

# 	def forward(self, x):
# 		out = self.relu(self.conv1(  x))
# 		out = self.relu(self.conv2(out))
# 		out = self.relu(self.conv3(out))
# 		out = self.relu(self.conv4(out))
# 		out = self.relu(self.conv5(out))
# 		out = self.relu(self.conv6(out))
# 		return out




class OpenPose_CPM(nn.Module):
	def __init__(self, num_class):
		super(OpenPose_CPM, self).__init__()
		self.pool    = nn.MaxPool2d(2, padding = 0)
		self.relu    = nn.ReLU(inplace = True)
		self.conv1_1 = nn.Conv2d(  3,  64, kernel_size = 3, padding =1)
		self.conv1_2 = nn.Conv2d( 64,  64, kernel_size = 3, padding =1)

		self.conv2_1 = nn.Conv2d( 64, 128, kernel_size = 3, padding =1)
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
		self.conv6_2_CPM = nn.Conv2d(512, num_class, kernel_size = 1, padding =0)

		self.stage2 = Repeat(128 + num_class, num_class)
		self.stage3 = Repeat(128 + num_class, num_class)
		self.stage4 = Repeat(128 + num_class, num_class)
		self.stage5 = Repeat(128 + num_class, num_class)
		self.stage6 = Repeat(128 + num_class, num_class)

		
		self.thumb  = Repeat(5, 4)
		self.index  = Repeat(5, 4)
		self.middle = Repeat(5, 4)
		self.ring   = Repeat(5, 4)
		self.pinky  = Repeat(5, 4)


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
		out_6 = self.stage6(out_6)

		wrist  = out_6[:, 0: 1,:,:]
		thumb  = torch.cat((out_6[:, 1: 5,:,:], wrist), 1)
		index  = torch.cat((out_6[:, 5: 9,:,:], wrist), 1)
		middle = torch.cat((out_6[:, 9:13,:,:], wrist), 1)
		ring   = torch.cat((out_6[:,13:17,:,:], wrist), 1)
		pinky  = torch.cat((out_6[:,17:21,:,:], wrist), 1)

		thumb0  = self.thumb(thumb)
		index0  = self.index(index)
		middle0 = self.middle(middle)
		ring0   = self.ring(ring)
		pinky0  = self.pinky(pinky)

		out_7   = torch.cat((wrist,thumb0, index0, middle0, ring0, pinky0), 1)

		return [out_1, out_2, out_3, out_4, out_5, out_6, out_7]

def renderhand(num_classes = 22, **kwargs):
	return OpenPose_CPM(num_classes)
