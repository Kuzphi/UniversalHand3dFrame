import sys
sys.path.append('../')

import src.model.ICCV17
from src.utils.imutils import plothand_3d
import torch
weight1 = torch.load('pretrain_weight/iccv17-rhd-stb.torch')
weight2 = torch.load('pretrain_weight/Nov  1 21:27:42_train_ICCV17_RHD_Tencent/best/checkpoint.pth.tar')
model = ICCV17(num_joints = 21)
model.load_dict(weight1)
model.eval()

model.load_dict(weight2)

