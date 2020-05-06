############################# Import Section #################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Imports related to PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# Generic imports

import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# import hickle as hkl

# Dependencies classes and functions
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import videoDataset
from utils import save_checkpoint
from utils import getIndex
from dataset.kth import KTH, KTH_LIST
# Import Model
# from model.DyanOF import OFModel

from model.DyanOF_Compact import OFModel_C

############################# Import Section #################################

# HyperParameters for the Network
NumOfPoles = 40
EPOCH = 120
BATCH_SIZE = 1
LR = 0.0001

FRA = 10  # input number of frame
PRE = 0  # output number of frame
N_FRAME = FRA + PRE
N = NumOfPoles * 4
T = FRA  # number of row in dictionary(same as input number of frame)
saveEvery = 1

# Load saved model

checkptname = "./weight/e3d_action_full_128/128_Model_"

# Load input data
rootDir = '/data/huaiyu/data/kth_action/'
trainingData = KTH_LIST(rootDir, 10)
dataloader = DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

# Initializing r, theta
P, Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

# Create the model
# model = OFModel(Drr, Dtheta, T, PRE)
model = OFModel_C(Drr, Dtheta, T, PRE)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120],
                                     gamma=0.1)  # if Kitti: milestones=[100,150]
loss_mse = nn.MSELoss()
start_epoch = 1

print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()
t = time.time()
# Start the Training
for epoch in range(start_epoch, EPOCH + 1):
    loss_value = []
    scheduler.step()

    for i_batch, sample in enumerate(dataloader):
        data = sample['frames_A'].squeeze(0).cuda()
        expectedOut = Variable(data)
        inputData = Variable(data)
        optimizer.zero_grad()
        output = model(inputData)
        # if Kitti: loss = loss_mse(output, expectedOut)
        loss = loss_mse(output, expectedOut)
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()
        loss_value.append(loss.data.item())

    loss_val = np.mean(np.array(loss_value))
    s = 'Epoch:{} | train loss: {:.6f}  time: {:.1f}s'.format(epoch, loss_val, time.time() - t)
    t = time.time()
    print(s)
    with open('./weight/e3d_action_full_64/64_results.txt', 'a') as file:
        file.write(s + '\n')

    if epoch % saveEvery == 0:
        # save_checkpoint({'epoch': epoch + 1,
        #                  'state_dict': model.state_dict(),
        #                  'optimizer': optimizer.state_dict(),
        #                  }, checkptname + str(epoch) + '.pth')
        # print(model.l1.weight.data)
        torch.save(model, checkptname + str(epoch) + '.pth')
