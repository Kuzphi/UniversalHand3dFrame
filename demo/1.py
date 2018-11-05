import sys
sys.path.append('../')

from src.model import ICCV17
from src.utils.imutils import plot_hand_3d,load_image,im_to_numpy
import torch
import scipy
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def get_model(weight_path):
	weight = torch.load(weight_path)
	if weight.has_key('state_dict'):
		weight = weight['state_dict']
	model = ICCV17(num_joints = 21)	
	model = torch.nn.DataParallel(model).cuda().float()
	model.load_state_dict(weight)
	model.eval()
	return model
x = pickle.load(open('cropped_anno_bigger.pickle'))
gt = np.array(x['00000_1.png']['xyz_coor'])

OriImg = scipy.misc.imread("test_img/00000_1.png")

model1 = get_model('../pretrained_weight/iccv17-rhd-stb.torch')
model2 = get_model('../output/Nov  1 21:27:42_train_ICCV17_RHD_Tencent/best/checkpoint.pth.tar')

imgpath = 'test_img/00000_1.png'
img  = load_image(imgpath).view(-1,3,256,256)

pose1 = model1({'img':img, 'hand_side':torch.tensor([[1, 0]]).float() })
pose2 = model2({'img':img, 'hand_side':torch.tensor([[1, 0]]).float() })

pose1 = pose1['pose3d'][0].detach().cpu().numpy()
pose2 = pose2['pose3d'][0].detach().cpu().numpy()
gt = np.array(gt)

fig = plt.figure(1)
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222, projection = '3d')
ax2 = fig.add_subplot(223, projection = '3d')
ax3 = fig.add_subplot(224, projection = '3d')
# img = im_to_numpy(img[0])
# img = (img + 0.5) * 255
ax0.imshow(OriImg)
# ax1.imshow(OriImg)
ax1.set_title('Ground Turth')
ax2.set_title('baseline')
ax3.set_title('new model')

ax0.axis('off')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
gt = torch.tensor(gt)
index_bone_length = torch.norm(gt[12,:] - gt[11,:])
gt[0, :] = (gt[0] + gt[12]) / 2.
gt = gt - gt[:1,:].repeat(21,1)
gt = gt.numpy()

pose1 *= index_bone_length
pose2 *= index_bone_length

plot_hand_3d(gt, ax1, linewidth='3')
plot_hand_3d(pose1, ax2, linewidth='3')
plot_hand_3d(gt, ax2, color_fixed = np.array([0,0,0]), linewidth='3')

plot_hand_3d(pose2, ax3, linewidth='3')
plot_hand_3d(gt, ax3, color_fixed = np.array([0,0,0]), linewidth='3')
plt.savefig('1.png')
plt.show()

# from matplotlib.transforms import Bbox
# def full_extent(ax, pad=0.0):
#     """Get the full extent of an axes, including axes labels, tick labels, and
#     titles."""
#     # For text objects, we need to draw the figure first, otherwise the extents
#     # are undefined.
#     ax.figure.canvas.draw()
#     items = ax.get_xticklabels() + ax.get_yticklabels() 
# #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
#     items += [ax, ax.title]
#     bbox = Bbox.union([item.get_window_extent() for item in items])

#     return bbox.expanded(1.0 + pad, 1.0 + pad)
# extent = full_extent(ax2).transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('ax2_figure.png', bbox_inches=extent)

# extent = full_extent(ax3).transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('ax3_figure.png', bbox_inches=extent)
