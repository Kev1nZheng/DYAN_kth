import torch
import torch.nn as nn
import cv2
import numpy as np

original = cv2.imread('/data/Huaiyu/DYAN/caltech/test/set06V002/00000.jpg')
original = cv2.resize(original, (160, 128))
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
filename = 'view_image_o.jpg'
cv2.imwrite(filename, original)
# original = torch.from_numpy(original)
print('original', original.shape)

img_A = cv2.imread('/data/Huaiyu/DYAN/DYAN_test_5_15/saveDir_cn10/img/test_5_15_cn10_2_000.jpg')
img_A = cv2.resize(img_A, (160, 128))
img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
filename = 'view_image_a.jpg'
cv2.imwrite(filename, img_A)
# img_A = torch.from_numpy(img_A)
print('img_A', img_A.shape)
# loss_mse = nn.MSELoss()
different = img_A - original
filename = 'view_image_d.jpg'
cv2.imwrite(filename, different)
loss = np.sum(different)

print(loss)
