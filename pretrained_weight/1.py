import sys
import torch
import pickle
sys.path.append('..')

from src.model import ICCV17
model = ICCV17(21)
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
# print (sorted(model.state_dict().keys()))
# print model.summary()
yy = {}
for key in model.state_dict():
	# print ("!!!!!!!")
	# print (w.name, w.data.size())
	# print (w)
	yy[key] =  model.state_dict()[key].size()
	# print(yy, yy[key])

now = sorted(model.state_dict().keys())
xx = {}
for fpath in ['posenet3d-rhd-stb.pickle']:
	w = pickle.load(open(fpath,'r'))
	for key in w:
		xx[key] = w[key]

ori = xx.keys()
ori = sorted(ori)
result = {}
for a, b in zip(ori, now):
	# print a, xx[a].size
	cnt = 1
	for j in  yy[b]:
		cnt *= j
	print(a, b)
	print(xx[a].size, yy[b])
	assert xx[a].size == cnt
	
	sz = yy[b]

	if len(sz) == 4:
		w = xx[a].reshape(sz[3],sz[2],sz[1],sz[0])
		w = w.transpose(3,2,0,1)
	elif len(sz) == 2:
		w = xx[a].reshape(sz[1],sz[0])
		w = w.transpose(1,0)
	else:
		w = xx[a]
	result[b] = torch.from_numpy(w)

# import torch
# import tensorflow
# import numpy
# import scipy
# img = scipy.misc.imread("00000.jpg", mode='RGB') 
# img = scipy.misc.imresize(img, (256, 256))
# img = (img.astype('float') / 255.0) - 0.5
# y = numpy.random.rand(1,256,256,149)
# inp = torch.load("../out2.torch")
# y = inp.cpu().numpy().transpose(0,2,3,1)

# filterx = xx['PoseNet2D/conv6_1/weights'].reshape(7,7,149,128)
# bias = xx['PoseNet2D/conv6_1/biases']

# a = tensorflow.nn.conv2d(y, filterx, [1,1,1,1], 'SAME')
# b = tensorflow.nn.bias_add(a, bias)
# with tensorflow.Session() as sess:
# 	t = (sess.run(b))

# x = torch.nn.Conv2d(3,64,3, padding = 3)


# filter = numpy.transpose(filterx, (3,2,0,1))
# x.weight = torch.nn.Parameter(torch.from_numpy(filter))
# x.bias = torch.nn.Parameter(torch.from_numpy(bias))
# z = numpy.transpose(y, (0,3,1,2))
# l = x(torch.from_numpy(z).float())
# l = l.detach().numpy()
# l = numpy.transpose(l,(0,2,3,1))

# print(t.sum(), t.shape)
# print(l.sum(), l.shape)
# print(numpy.abs(t-l).max())
torch.save(result, 'iccv17_pretrained-rhd-stb.torch')
