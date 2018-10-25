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

		self.upsampler = nn.Upsample(scale_factor = 8, mode = 'bilinear', align_corners = True)

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

		outputs = [out_1, out_2, out_3, out_4, out_5, out_6]
		return [self.upsampler(out) for out in outputs]

class OpenPose_Pose(nn.Module):

	def make_layers(self, cfg_dict):
		layers = []
		for i in range(len(cfg_dict)-1):
			one_ = cfg_dict[i]
			for k,v in one_.iteritems():      
				if 'pool' in k:
					layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
				else:
					conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
					layers += [conv2d, nn.ReLU(inplace=True)]
		one_ = cfg_dict[-1].keys()
		k = one_[0]
		v = cfg_dict[-1][k]
		conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
		layers += [conv2d]
		return nn.Sequential(*layers)

	def get_model_dict(self):
		blocks = {}

		block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]

		blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

		blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

		for i in range(2,7):
			blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
		{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
			blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
		{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

		layers = []
		for i in range(len(block0)):
			one_ = block0[i]
			for k,v in one_.iteritems():      
				if 'pool' in k:
					layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
				else:
					conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
					layers += [conv2d, nn.ReLU(inplace=True)]  
			   
		models = {}           
		models['block0']=nn.Sequential(*layers)

		for k,v in blocks.iteritems():
			models[k] = self.make_layers(v)
		return models

	def __init__(self):
		super(OpenPose_Pose, self).__init__()

		model_dict = self.get_model_dict()
		self.model0   = model_dict['block0']
		self.model1_1 = model_dict['block1_1']        
		self.model2_1 = model_dict['block2_1']  
		self.model3_1 = model_dict['block3_1']  
		self.model4_1 = model_dict['block4_1']  
		self.model5_1 = model_dict['block5_1']  
		self.model6_1 = model_dict['block6_1']  
		
		self.model1_2 = model_dict['block1_2']        
		self.model2_2 = model_dict['block2_2']  
		self.model3_2 = model_dict['block3_2']  
		self.model4_2 = model_dict['block4_2']  
		self.model5_2 = model_dict['block5_2']  
		self.model6_2 = model_dict['block6_2']
		
		
	def forward(self, x):
		out1 = self.model0(x)
		
		out1_1 = self.model1_1(out1)
		out1_2 = self.model1_2(out1)
		out2  = torch.cat([out1_1,out1_2,out1],1)
		
		out2_1 = self.model2_1(out2)
		out2_2 = self.model2_2(out2)
		out3   = torch.cat([out2_1,out2_2,out1],1)
		
		out3_1 = self.model3_1(out3)
		out3_2 = self.model3_2(out3)
		out4   = torch.cat([out3_1,out3_2,out1],1)

		out4_1 = self.model4_1(out4)
		out4_2 = self.model4_2(out4)
		out5   = torch.cat([out4_1,out4_2,out1],1)
		
		out5_1 = self.model5_1(out5)
		out5_2 = self.model5_2(out5)
		out6   = torch.cat([out5_1,out5_2,out1],1)
			  
		out6_1 = self.model6_1(out6)
		out6_2 = self.model6_2(out6)
		
		return out6_1,out6_2

def openpose_pose(**kwargs):
	return OpenPose_Pose()

def openpose_face(num_joints = 71, **kwargs):
	return OpenPose_CPM(num_joints)

def openpose_hand(num_joints = 22, **kwargs):
	return OpenPose_CPM(num_joints)