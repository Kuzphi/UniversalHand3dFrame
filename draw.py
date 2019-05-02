import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import torch 

dpi = 200
dpi_draw = 200
SHP_PSO = [[20,25,30,35,40,45,50],[0.31,0.54,.68,.75,.81,.83,.84],0.709]
SHP_ICPPSO = [[20,25,30,35,40,45,50],[0.51,0.65,0.72,0.77,.81,.82,.825],0.748]
SHP_Cai = [[20,25,30,35,40,45,50],[0.952,0.979,0.99,0.995,0.997,0.999,1.0],0.990]
SHP_Iqbal = [[20,25,30,35,40,45,50],[0.96,0.984,0.992,0.996,0.9985,1.0,1.0],0.994]
SHP_CHPR = [[20,25,30,35,40,45,50],[0.58,0.71,0.81,0.86,0.9,0.93,0.95],0.839]
#SHP_arxiv = [np.arange(20,51,2.5).tolist()[1:],[0.49,0.57,0.64,0.71,0.76,0.79,0.83,0.86,0.892,0.915,0.93,0.94],0.770] #1609.09058
SHP_ZB = [np.arange(20,51,2.5).tolist()[1:],[0.86,0.89,0.91,0.922,0.931,0.94,0.95,0.96,0.964,0.968,0.971,0.973],0.948] #1609.09058

def read_pck(filename,arange=np.arange(15,51,5)):
    result_dic = pickle.load(open(filename,'rb'),encoding='latin1')
    y = result_dic["y"]
    x = result_dic["x"]
    x_axis = arange.tolist()[1:]
    y_axis = []
    for val in x_axis:
        yval = (x < val).sum()
        y_axis.append(1. * yval/len(x))
    print(y_axis)
    return x_axis,y_axis

def load_result(filename,arange=np.arange(15,51,5)):
    result_dic = pickle.load(open(filename,'rb'))
    x = result_dic['dis3d'].view(-1)
    x = np.array(x)
    x_axis = arange.tolist()[1:]
    y_axis = []
    for val in x_axis:
        yval = (x < val).sum()
        y_axis.append(1. * yval/len(x))
    print(y_axis)
    return x_axis,y_axis


SHP_Caid = load_result('infer_result/STB_direct_with_depth/preds.pickle')
SHP_Cai  = load_result('infer_result/STB_direct/preds.pickle')
Ours     = load_result('infer_result/STB_GAN/preds.pickle')

Ours, = plt.plot(Ours[0], Ours[1], 'ro--',label="Ours(AUC = 0.990)")
SHP_PSO_plot, = plt.plot(SHP_PSO[0], SHP_PSO[1], 'mo--',label="PSO(AUC=0.709)")
SHP_ICPPSO_plot, = plt.plot(SHP_ICPPSO[0], SHP_ICPPSO[1], 'go--',label="ICPPSO(AUC = 0.748)")
CHPR_plot, = plt.plot(SHP_CHPR[0], SHP_CHPR[1], 'ys--',label="CHPR(AUC = 0.839)")
Cai_plot, = plt.plot(SHP_Cai[0], SHP_Cai[1], 'bv--',label="Cai w/o depth et al.(AUC = 0.976)")
Caid_plot, = plt.plot(SHP_Caid[0], SHP_Caid[1], 'cv--',label="Cai w/ depthet al.(AUC = 0.984)")
# Iqbal_plot, = plt.plot(SHP_Iqbal[0], SHP_Iqbal[1], 'yv--',label="Iqbal et al.")
zb_plot, = plt.plot(SHP_ZB[0], SHP_ZB[1], 'ks--',label="Z&B(AUC = 0.948)")
plt.grid(True,linestyle='dotted')
plt.title("PCK Curve of STB dataset")
plt.ylabel('PCK')
plt.xlabel('Threshold in mm')
plt.ylim((0.3,1))
plt.xlim((19,51))
plt.legend(handles=[SHP_PSO_plot,SHP_ICPPSO_plot,CHPR_plot,Cai_plot,Caid_plot,zb_plot, Ours])
plt.savefig("./STB.png", dpi=dpi_draw)
plt.clf()


RHD_ZB = [[21,25,30,35,40,45,50],[.46,.54,.62,.695,.75,.8,.82],0.675]
RHD_Caid = load_result('infer_result/RHD_direct_with_depth/preds.pickle')
RHD_Cai  = load_result('infer_result/RHD_direct/preds.pickle')
RHD_Ours = load_result('infer_result/RHD_GAN/preds.pickle')

# plt.figure(figsize=(1000/dpi, 1400/dpi), dpi=dpi)
plt.title("PCK Curve of RHD dataset")
plt.grid(True,linestyle='dotted')

Ours_plot, = plt.plot(RHD_Ours[0], RHD_Ours[1], 'rs--',label="Ours(AUC = 0.859)")
# Caid_plot, = plt.plot(RHD_Caid[0], RHD_Caid[1], 'ks--',label="Cai et al. w/ depth")
Cai_plot, = plt.plot(RHD_Cai[0], RHD_Cai[1], 'cs--',label="Cai et al. w/o depth(AUC = 0.839)")
ZB_plot, = plt.plot(RHD_ZB[0], RHD_ZB[1], 'ys--',label="Z&B (AUC = 0.675)")

plt.ylabel('PCK')
plt.xlabel('Threshold in mm')
plt.ylim((0.3, 1.))
plt.xlim((19,51))
plt.legend(handles=[Ours_plot, Cai_plot, ZB_plot])
plt.savefig("./RHD.png", dpi=dpi_draw)
plt.clf()


MHD_Cai  = load_result('infer_result/MHP_direct/preds.pickle')
MHD_Ours = load_result('infer_result/MHP_GAN/preds.pickle')

# plt.figure(figsize=(1000/dpi, 1400/dpi), dpi=dpi)
plt.title("PCK Curve of MHD dataset")
plt.grid(True,linestyle='dotted')

Ours_plot, = plt.plot(MHD_Ours[0], MHD_Ours[1], 'rs--',label="Ours(AUC = 0.939)")
# Caid_plot, = plt.plot(RHD_Caid[0], RHD_Caid[1], 'ks--',label="Cai et al. w/ depth")
Cai_plot, = plt.plot(MHD_Cai[0], MHD_Cai[1], 'cs--',label="Cai et al. w/o depth(AUC = 0.928)")
# ZB_plot, = plt.plot(RHD_ZB[0], RHD_ZB[1], 'ys--',label="Z&B (AUC = 0.675)")

plt.ylabel('PCK')
plt.xlabel('Threshold in mm')
plt.ylim((0.3, 1.))
plt.xlim((19,51))
plt.legend(handles=[Ours_plot, Cai_plot])
plt.savefig("./MHD.png", dpi=dpi_draw)

