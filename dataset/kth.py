import torch
import torch.utils.data as data
import numpy as np
import os, random
import cv2

img_channel = 1


class KTH(data.Dataset):
    def __init__(self, rootDir, N_FRAME):
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 64 * 64
        classesList = [
            name for name in os.listdir(rootDir)
            if os.path.isdir(os.path.join(rootDir))
        ]
        classesList.sort()
        self.listOfFolders = []
        for i in range(len(classesList)):
            classesList[i] = os.path.join(rootDir, classesList[i])
            classes_videoList = [
                name for name in os.listdir(classesList[i])
                if os.path.isdir(os.path.join(classesList[i]))
            ]
            classes_videoList.sort()
            for j in range(len(classes_videoList)):
                classes_videoList[j] = os.path.join(classesList[i],
                                                    classes_videoList[j])
            self.listOfFolders.extend(classes_videoList)
        self.listOfFolders.sort()

    def __len__(self):
        return len(self.listOfFolders)

    def readClip(self, folderName):
        path = os.path.join(self.rootDir, folderName)
        sample_A = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        sample_B = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        frames = [each for each in os.listdir(path) if each.endswith('.png')]
        nFrames = len(frames)
        # startid = random.randint(0, nFrames - 40)
        startid = 10
        for framenum in range(self.nfra):
            imgname_A = os.path.join(
                path, 'image-%03d_64x64' % (framenum + startid) + '.png')
            imgname_B = os.path.join(
                path, 'image-%03d_64x64' % (framenum + startid + 10) + '.png')
            img_A = cv2.imread(imgname_A)
            img_B = cv2.imread(imgname_B)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
            pix_A = np.array(img_A)
            pix_B = np.array(img_B)
            pix_A = pix_A / 225
            pix_B = pix_B / 225
            pix_A = pix_A[:, :, np.newaxis]
            pix_B = pix_B[:, :, np.newaxis]
            pix_A = np.transpose(pix_A, (2, 0, 1))
            pix_B = np.transpose(pix_B, (2, 0, 1))
            sample_A[:, framenum] = torch.from_numpy(
                pix_A.reshape(img_channel,
                              self.numpixels)).type(torch.FloatTensor)
            sample_B[:, framenum] = torch.from_numpy(
                pix_B.reshape(img_channel,
                              self.numpixels)).type(torch.FloatTensor)
        return sample_A, sample_B

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame_A, Frame_B = self.readClip(folderName)
        sample = {'frames_A': Frame_A, 'frames_B': Frame_B, 'Name': folderName}
        return sample


class KTH_LIST(data.Dataset):
    def __init__(self, rootDir, N_FRAME):
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.imgSize = 128
        self.numpixels = self.imgSize * self.imgSize

        with open('/data/huaiyu/DYAN/DYAN_kth/dataset/e3d_kth_test_list.txt',
                  'r') as f:
            self.list = f.readlines()

    def __len__(self):
        return len(self.list)

    def readClip(self, folderName, startid):
        path = folderName
        sample_A = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        sample_B = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        frames = [each for each in os.listdir(path) if each.endswith('.jpg')]
        nFrames = len(frames)

        for framenum in range(self.nfra):
            imgname_A = os.path.join(
                path, 'image_%04d' % (framenum + startid) + '.jpg')
            imgname_B = os.path.join(
                path, 'image_%04d' % (framenum + startid + 10) + '.jpg')
            img_A = cv2.imread(imgname_A)
            img_B = cv2.imread(imgname_B)
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

            img_A = cv2.resize(img_A, (self.imgSize, self.imgSize))
            img_B = cv2.resize(img_B, (self.imgSize, self.imgSize))

            pix_A = np.array(img_A)
            pix_B = np.array(img_B)
            pix_A = pix_A / 225
            pix_B = pix_B / 225
            pix_A = pix_A[:, :, np.newaxis]
            pix_B = pix_B[:, :, np.newaxis]
            pix_A = np.transpose(pix_A, (2, 0, 1))
            pix_B = np.transpose(pix_B, (2, 0, 1))
            sample_A[:, framenum] = torch.from_numpy(
                pix_A.reshape(img_channel,
                              self.numpixels)).type(torch.FloatTensor)
            sample_B[:, framenum] = torch.from_numpy(
                pix_B.reshape(img_channel,
                              self.numpixels)).type(torch.FloatTensor)
        return sample_A, sample_B

    def __getitem__(self, idx):
        folderName = os.path.dirname(self.list[idx])
        startid = int(self.list[idx].split('/')[-1][-9:-5])
        Frame_A, Frame_B = self.readClip(folderName, startid)
        sample = {
            'frames_A': Frame_A,
            'frames_B': Frame_B,
            'Name': self.list[idx]
        }
        return sample


if __name__ == '__main__':
    rootDir = '/data/huaiyu/data/kth_action/'
    trainingData = KTH_LIST(rootDir=rootDir, N_FRAME=10)
    for i, input in enumerate(trainingData):
        print('Name', input['Name'])
        print('input frames_A', input['frames_A'].size())
        print(i)
