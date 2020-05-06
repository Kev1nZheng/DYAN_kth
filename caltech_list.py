import os

root_dir = '/data/Huaiyu/DYAN/caltech/train/'
dirpath = []
dirpath = os.listdir(root_dir)
dirpath.sort()
list = []
print(dirpath)
sum = 0
for i in range(len(dirpath)):
    path = os.path.join(root_dir, dirpath[i])
    print(path)
    frames = [each for each in os.listdir(path) if each.endswith('.jpg')]
    nFrames = len(frames)
    print(nFrames)
    nClips = (nFrames // 20)
    sum = sum + nClips
    for j in range(0, nClips):
        startid = j * 20
        startid_path = os.path.join(path, '%05d' % startid)
        list.append(startid_path)
        print(j, startid_path)
print(len(list))
print(sum)
