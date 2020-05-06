import numpy as np
import cv2
import os
import torch

rootDir = '/data/Huaiyu/DYAN/caltech/train/'
saveDir = '/data/Huaiyu/DYAN/DYAN_test_5_15/saveDir_5_23_cn50/img/'
folderList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
folderList.sort()

s1e = np.load('/data/Huaiyu/DYAN/DYAN_test_5_15/saveDir_5_23_cn50/npy/test_5_23_cn50_5.npy')
s1e1 = np.load('/data/Huaiyu/DYAN/DYAN_test_5_15/saveDir_5_23_cn50/npy/test_5_23_cn50_1_5.npy')
s1e2 = np.load('/data/Huaiyu/DYAN/DYAN_test_5_15/saveDir_5_23_cn50/npy/test_5_23_cn50_1_5.npy')
# s1e = s1e[:, :, 0:1]
print(s1e.shape)
# s1e = s1e.reshape(161, 20480)
# s1e = s1e.squeeze(0)
s1e = np.transpose(s1e, (1, 2, 0))
s1e = torch.from_numpy(s1e)
print('s1e shape', s1e.shape)
s1e = torch.reshape(s1e, (10, 160, 128, 1))
s1e = s1e.data.numpy()
for i in range(10):
    filename = "test_5_23_cn50_5_{:0>3d}.jpg".format(i)
    filename = os.path.join(saveDir, filename)
    print(filename)
    # cv2.imwrite(filename, s1e[:, :, i])
    img = s1e[i, :, :, :]
    print('img shape', img.shape)
    amin, amax = img.min(), img.max()
    norimg = (img - amin) / (amax - amin)
    norimg = norimg * 255
    norimg = cv2.resize(norimg, (160, 128))
    print(filename)
    cv2.imwrite(filename, norimg)

s1e1 = s1e1.reshape(161, 20480)

print(s1e.shape)
# s1e = np.reshape(128, 160, 161)
