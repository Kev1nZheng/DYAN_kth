############################# Import Section #################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Imports related to PyTorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
# Generic imports
import time
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# from model.DyanOF import OFModel
from model.DyanOF_Compact import OFModel_C
from model.Dyan_Pix2Pix import Dyan_Pix2Pix
from utils import getListOfFolders

from skimage import measure
from scipy.misc import imread, imresize
from scipy import io
from utils import gridRing
from dataset.kth import KTH, KTH_LIST
from torch.utils.data import Dataset, DataLoader

############################# Import Section #################################
# Hyper Parameters
FRA = 10
PRE = 0
N_FRAME = FRA + PRE
N_FRAME = 20
T = FRA
numOfPixels = 128 * 128


def creatRealDictionary(T, Drr, Dtheta):
    WVar = []
    Wones = torch.ones(1).cuda()
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
    torchHorizontal = torchHorizontal.view(1, 1, 1, input.size(3)).expand(
        input.size(0), 1, input.size(2), input.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, input.size(2))
    torchVertical = torchVertical.view(1, 1, input.size(2), 1).expand(input.size(0), 1, input.size(2), input.size(3))

    tensorGrid = torch.cat([torchHorizontal, torchVertical], 1).cuda()
    tensorFlow = torch.cat([
        tensorFlow[:, 0:1, :, :] / ((input.size(3) - 1.0) / 2.0),
        tensorFlow[:, 1:2, :, :] / ((input.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(
        input=input,
        grid=(tensorGrid + tensorFlow).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='border')


##################### Only for Kitti dataset need to define: ##############

############################################################################

# Load the model
kth_model = torch.load('/data/huaiyu/DYAN/DYAN_kth/weight/e3d_action_full_128/128_Model_8.pth').cuda()
kth_model.eval()
saveDir = '/data/huaiyu/data/kth_action_full_exp/e3d_action_full_128'
rootDir = '/data/huaiyu/data/kth_action/'
trainingData = KTH_LIST(rootDir, 10)
dataloader = DataLoader(trainingData, batch_size=1, shuffle=False, num_workers=1)

##################### Testing script ONLY for Kitti dataset: ##############
count_num = 0
kth_model.eval()
for i_batch, sample in enumerate(dataloader):
    frames_A = sample['frames_A'].squeeze(0).cuda()
    frames_B = sample['frames_B'].squeeze(0).cuda()
    clip_name = sample['Name'][0].split('/')[-2] + '_%05d' % int(sample['Name'][0].split('/')[-1][-9:-5])
    print(i_batch, clip_name)
    with torch.no_grad():
        s1e = kth_model.forward2(frames_A).data.cpu().numpy()
        s2e = kth_model.forward2(frames_B).data.cpu().numpy()

        np.save(os.path.join(saveDir, 'A', 'test', clip_name), s1e)
        np.save(os.path.join(saveDir, 'B', 'test', clip_name), s2e)

############################################################################
