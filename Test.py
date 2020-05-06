############################# Import Section #################################
# Imports related to PyTorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision

# Generic imports
import os
import time
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from model.DyanOF import OFModel
from model.Dyan_Pix2Pix import Dyan_Pix2Pix
from utils import getListOfFolders

from skimage import measure
from scipy.misc import imread, imresize
from scipy import io
from utils import gridRing

############################# Import Section #################################


# Hyper Parameters
FRA = 10
PRE = 0
N_FRAME = FRA + PRE
N_FRAME = 20
T = FRA
numOfPixels = 64 * 64

gpu_id = 3
opticalflow_ckpt_file = '/data/Huaiyu/DYAN/DYAN_kth/weight/KTH_Model_44.pth'

def creatRealDictionary(T, Drr, Dtheta, gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones = Variable(Wones, requires_grad=False)
    for i in range(0, T):
        W1 = torch.mul(torch.pow(Drr, i), torch.cos(i * Dtheta))
        W2 = torch.mul(torch.pow(-Drr, i), torch.cos(i * Dtheta))
        W3 = torch.mul(torch.pow(Drr, i), torch.sin(i * Dtheta))
        W4 = torch.mul(torch.pow(-Drr, i), torch.sin(i * Dtheta))
        W = torch.cat((Wones, W1, W2, W3, W4), 0)
        WVar.append(W.view(1, -1))
    dic = torch.cat((WVar), 0)
    G = torch.norm(dic, p=2, dim=0)
    idx = (G == 0).nonzero()
    nG = G.clone()
    nG[idx] = np.sqrt(T)
    G = nG
    dic = dic / G
    return dic

def warp(input, tensorFlow):
    torchHorizontal = torch.linspace(-1.0, 1.0, input.size(3))
    torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(input.size(0), 1, input.size(2),
                                                                          input.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
    torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))

    tensorGrid = torch.cat([torchHorizontal, torchVertical], 1).cuda(gpu_id)
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=input, grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='border')


##################### Only for Kitti dataset need to define: ##############

############################################################################

# Load the model
kth_model = torch.load('/data/Huaiyu/DYAN/DYAN_kth/weight/161/161_Model_6.pth').cuda(gpu_id)
Sample = torch.FloatTensor(1, N_FRAME, numOfPixels)
kth_model.eval()

train_list = False

rootDir = '/data/Huaiyu/DYAN/data/kth/processed/'
saveDir = '/data/Huaiyu/DYAN/data/kth/dyan_161'
classesList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
classesList.sort()
folderList = []
for i in range(len(classesList)):
    classesList[i] = os.path.join(rootDir, classesList[i])
    classes_videoList = [name for name in os.listdir(classesList[i]) if os.path.isdir(os.path.join(classesList[i]))]
    classes_videoList.sort()
    for j in range(len(classes_videoList)):
        classes_videoList[j] = os.path.join(classesList[i], classes_videoList[j])
    folderList.extend(classes_videoList)
folderList.sort()
print(folderList)

##################### Testing script ONLY for Kitti dataset: ##############
count_num = 0
print('Sample', Sample.shape)
for folder in folderList:
    print("Started testing for - " + folder)
    for k in range(N_FRAME):
        imgname = os.path.join(folder, 'image-%03d_64x64' % (k + 10) + '.png')
        # print(imgname)
        img = cv2.imread(imgname, 0)
        pix = np.array(img)
        pix = pix[:, :, np.newaxis]
        pix = pix / 225
        pix = np.transpose(pix, (2, 0, 1))
        Sample[:, k, :] = torch.from_numpy(pix.reshape(1, numOfPixels)).type(torch.FloatTensor).unsqueeze(0)
        s1 = Sample[:, 0:FRA, :].cuda(gpu_id)
        s2 = Sample[:, FRA:N_FRAME, :].cuda(gpu_id)
    with torch.no_grad():
        kth_model.eval()
        s1e = kth_model.forward2(Variable(s1)).data.cpu().numpy()
        s2e = kth_model.forward2(Variable(s2)).data.cpu().numpy()
    np.save(os.path.join(saveDir, 'A', 'train', folder.split('/')[-1]), s1e)
    np.save(os.path.join(saveDir, 'B', 'train', folder.split('/')[-1]), s2e)

############################################################################
