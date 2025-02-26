############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable

from math import sqrt
import numpy as np


############################# Import Section #################################

# Create Dictionary
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
    # print(idx)
    nG[idx] = np.sqrt(T)
    G = nG

    dic = dic / G

    return dic


def fista(D, Y, lambd, maxIter, gpu_id):
    # print('shapes',D.shape,Y.shape)
    DtD = torch.matmul(torch.t(D), D)
    L = torch.norm(DtD, 2)
    linv = 1 / L
    DtY = torch.matmul(torch.t(D), Y)
    x_old = Variable(torch.zeros(DtD.shape[1], DtY.shape[2]).cuda(gpu_id), requires_grad=True)
    t = 1
    y_old = x_old
    lambd = lambd * (linv.data.cpu().numpy())
    A = Variable(torch.eye(DtD.shape[1]).cuda(gpu_id), requires_grad=True) - torch.mul(DtD, linv)

    DtY = torch.mul(DtY, linv)

    Softshrink = nn.Softshrink(lambd)
    with torch.no_grad():
        for ii in range(maxIter):
            Ay = torch.matmul(A, y_old)
            del y_old
            with torch.enable_grad():
                x_new = Softshrink((Ay + DtY))
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
            tt = (t - 1) / t_new
            y_old = torch.mul(x_new, (1 + tt))
            y_old -= torch.mul(x_old, tt)
            if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-4:
                x_old = x_new
                break
            t = t_new
            x_old = x_new
            del x_new
    return x_old


class Encoder(nn.Module):
    def __init__(self, Drr, Dtheta, T, gpu_id):
        super(Encoder, self).__init__()

        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.T = T
        self.gid = gpu_id

    def forward(self, x):
        dic = creatRealDictionary(self.T, self.rr, self.theta, self.gid)
        ## for UCF Dataset:
        sparsecode = fista(dic, x, 0.1, 100, self.gid)
        ## for Kitti Dataset: sparsecode = fista(dic,x,0.01,80,self.gid)
        return Variable(sparsecode)


class Decoder(nn.Module):
    def __init__(self, rr, theta, T, PRE, gpu_id):
        super(Decoder, self).__init__()

        self.rr = rr
        self.theta = theta
        self.T = T
        self.PRE = PRE
        self.gid = gpu_id

    def forward(self, x):
        dic = creatRealDictionary(self.T + self.PRE, self.rr, self.theta, self.gid)
        result = torch.matmul(dic, x)
        return result


class Dyan_Pix2Pix(nn.Module):
    def __init__(self, Drr, Dtheta, T, PRE, gpu_id):
        super(Dyan_Pix2Pix, self).__init__()
        self.l1 = Encoder(Drr, Dtheta, T, gpu_id)
        self.l2 = Decoder(self.l1.rr, self.l1.theta, T, PRE, gpu_id)
        self.R_channel = 50
        self.conv1 = nn.Conv2d(161, self.R_channel, bias=False, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(self.R_channel, 161, bias=False, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.l1(x)  # [3,10,20480] to [3,161,20480]
        # x = x[0, :, :]  # [3,161,20480] to [1,161,20480]

        x = torch.reshape(x, (1, 161, 128, 160))
        x = self.conv1(x)  # [1,20,128,160]
        weight_conv1 = torch.reshape(self.conv1.weight.data, (self.R_channel, 161))

        file = open('conv1_weight.csv', 'w')
        np.savetxt(file, weight_conv1.cpu().numpy())
        file.close()

        weight_conv1_T = weight_conv1.transpose(0, 1)  # [161,20]

        # file = open('conv1_T_weight.csv', 'w')
        # np.savetxt(file, weight_conv1_T.cpu().numpy())
        # file.close()

        weight_conv2 = torch.mm(weight_conv1_T, weight_conv1)  # [161,161]

        # file = open('conv1_mul_weight.csv', 'w')
        # np.savetxt(file, weight_conv2.cpu().numpy())
        # file.close()

        weight_conv2 = torch.pinverse(weight_conv2)

        # file = open('conv1_inv_weight.csv', 'w')
        # np.savetxt(file, weight_conv2.cpu().numpy())
        # file.close()

        weight_conv2 = torch.mm(weight_conv2, weight_conv1_T)  # [161,161]*[161,20]--[161,20]

        # file = open('conv2_weight1.csv', 'w')
        # np.savetxt(file, torch.reshape(self.conv2.weight.data, (161, self.R_channel)).cpu().numpy())
        # file.close()

        weight_conv2 = torch.reshape(weight_conv2, (161, self.R_channel, 1, 1))
        self.conv2.weight.data = weight_conv2

        file = open('conv2_weight2.csv', 'w')
        np.savetxt(file, torch.reshape(self.conv2.weight.data, (161, self.R_channel)).cpu().numpy())
        file.close()
        # self.conv2.weight.register_hook(print)
        # print('\n')
        x = self.conv2(x)
        # print('x',x.shape)
        # x = torch.cat((x, x, x), 0)
        x = torch.reshape(x, (1, 161, 20480))

        x = self.l2(x)
        # x.register_hook(print)
        # print('x after l2', x.shape)
        return x

    def forward2(self, x):
        return self.l1(x)

    def forward3(self, x):
        x = self.l1(x)  # [3,10,20480] to [3,161,20480]
        # x = x[0, :, :]  # [3,161,20480] to [1,161,20480]
        x = torch.reshape(x, (1, 161, 128, 160))
        x = self.conv1(x)  # [1,20,128,160]
        weight_conv1 = torch.reshape(self.conv1.weight.data, (self.R_channel, 161))
        file = open('conv1_weight.csv', 'w')
        np.savetxt(file, weight_conv1.cpu().numpy())
        file.close()
        weight_conv1_T = weight_conv1.transpose(0, 1)  # [161,20]
        weight_conv2 = torch.mm(weight_conv1_T, weight_conv1)  # [161,161]
        weight_conv2 = torch.pinverse(weight_conv2)
        weight_conv2 = torch.mm(weight_conv2, weight_conv1_T)  # [161,161]*[161,20]--[161,20]
        weight_conv2 = torch.reshape(weight_conv2, (161, self.R_channel, 1, 1))
        self.conv2.weight.data = weight_conv2
        x = self.conv2(x)
        x = torch.reshape(x, (1, 161, 20480))
        return x
