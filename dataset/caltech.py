import torch
import torch.utils.data as data
import numpy as np
import os, random
import cv2

img_channel = 1


class caltech_data(data.Dataset):
    def __init__(self, folderList, rootDir, N_FRAME):
        self.listOfFolders = folderList
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 128 * 160

    def __len__(self):
        return len(self.listOfFolders)

    def readClip(self, folderName):
        path = os.path.join(self.rootDir, folderName)
        sample_A = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        sample_B = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        frames = [each for each in os.listdir(path) if each.endswith('.jpg')]
        nFrames = len(frames)
        startid = random.randint(0, nFrames - 40)
        for framenum in range(self.nfra):
            imgname_A = os.path.join(path, '%05d' % (framenum + startid + 1) + '.jpg')
            imgname_B = os.path.join(path, '%05d' % (framenum + startid + 11) + '.jpg')
            img_A = cv2.imread(imgname_A)
            img_B = cv2.imread(imgname_B)
            img_A = cv2.resize(img_A, (128, 160))
            img_B = cv2.resize(img_B, (128, 160))
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

            sample_A[:, framenum] = torch.from_numpy(pix_A.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
            sample_B[:, framenum] = torch.from_numpy(pix_B.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
        return sample_A, sample_B

    def __getitem__(self, idx):
        folderName = self.listOfFolders[idx]
        Frame_A, Frame_B = self.readClip(folderName)
        sample = {'frames_A': Frame_A, 'frames_B': Frame_B}
        return sample


class caltech_dataloader(data.Dataset):
    def __init__(self, rootDir, N_FRAME):
        self.rootDir = rootDir
        self.nfra = N_FRAME
        self.numpixels = 128 * 160
        dirpath = os.listdir(self.rootDir)
        dirpath.sort()
        self.list = []
        sum = 0
        for i in range(len(dirpath)):
            path = os.path.join(self.rootDir, dirpath[i])
            frames = [each for each in os.listdir(path) if each.endswith('.jpg')]
            nFrames = len(frames)
            nClips = (nFrames // 20) - 1
            sum = sum + nClips
            for j in range(0, nClips):
                startid = j * 20
                startid_path = os.path.join(path, '%05d' % startid)
                self.list.append(startid_path)
        with open('your_file.txt', 'w') as f:
            for item in self.list:
                f.write("%s\n" % item)

    def __len__(self):
        return len(self.list)

    def readClip(self, folderName, startid):
        path = folderName
        sample_A = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        sample_B = torch.FloatTensor(img_channel, self.nfra, self.numpixels)
        frames = [each for each in os.listdir(path) if each.endswith('.jpg')]
        nFrames = len(frames)
        startid = random.randint(startid, startid + 20)
        for framenum in range(self.nfra):
            imgname_A = os.path.join(path, '%05d' % (framenum + startid) + '.jpg')
            imgname_B = os.path.join(path, '%05d' % (framenum + startid + 10) + '.jpg')
            img_A = cv2.imread(imgname_A)
            img_B = cv2.imread(imgname_B)
            img_A = cv2.resize(img_A, (128, 160))
            img_B = cv2.resize(img_B, (128, 160))
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

            sample_A[:, framenum] = torch.from_numpy(pix_A.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
            sample_B[:, framenum] = torch.from_numpy(pix_B.reshape(img_channel, self.numpixels)).type(torch.FloatTensor)
        return sample_A, sample_B

    def __getitem__(self, idx):
        folderName = os.path.dirname(self.list[idx])
        startid = int(self.list[idx][-5:])
        Frame_A, Frame_B = self.readClip(folderName, startid)
        sample = {'frames_A': Frame_A, 'frames_B': Frame_B}
        return sample


if __name__ == '__main__':
    rootDir = '/data/Huaiyu/DYAN/caltech/train/'
    folderList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
    folderList.sort()

    trainingData = caltech_dataloader(rootDir=rootDir,
                                      N_FRAME=10)
    for i, input in enumerate(trainingData):
        print('input frames_A', input['frames_A'].size())
        print(i)
