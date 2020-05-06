import os

category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
category_2 = ['jogging', 'running']
rootDir = '/data/Huaiyu/DYAN/data/kth_action'
listOfFolders = []
classesList = [name for name in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir))]
classesList.sort()

for i in range(len(classesList)):
    classes_videoList = []
    classesList[i] = os.path.join(rootDir, classesList[i])
    classes_videoList = [name for name in os.listdir(classesList[i]) if
                         os.path.isdir(os.path.join(classesList[i]))]
    classes_videoList.sort()

    for j in range(len(classes_videoList)):
        classes_videoList[j] = os.path.join(classesList[i], classes_videoList[j])
        # print('classes_videoList', classes_videoList[j])
    listOfFolders.extend(classes_videoList)
listOfFolders.sort()
sum = 0
list = []
for i in range(len(listOfFolders)):
    frames = [each for each in os.listdir(listOfFolders[i]) if each.endswith('.jpg')]
    frames.sort()
    first_frame = int(frames[0][6:10])
    last_frame = int(frames[-1][6:10])
    # print('\nfirst_frame,last_frame', first_frame, last_frame)
    nFrames = len(frames)
    # print('frames:', frames)
    print('listOfFolders:', listOfFolders[i])
    # print(listOfFolders[i].split('_')[-3])
    if listOfFolders[i].split('_')[-3] in category_1:
        current_id = first_frame
        nClips = 0
        while current_id <= last_frame - 20:
            startid = current_id
            endid = current_id + 20
            # print('startid', startid)
            # print('endid', endid)
            # startid_name = 'image_{:0>4d}.jpg'.format(startid)
            # endid_name = 'image_{:>04d}.jpg'.format(endid)
            check_flag = 1
            for j in range(startid, endid + 1):
                # print('image_{:0>4d}.jpg'.format(j))
                if 'image_{:0>4d}.jpg'.format(j) not in frames:
                    check_flag = 0
            if check_flag == 1:
                startid_path = os.path.join(listOfFolders[i], '%05d' % startid)
                list.append(startid_path)
                nClips = nClips + 1
                current_id = current_id + 20
            else:
                current_id = current_id + 1
        print('nClips', nClips)
    elif listOfFolders[i].split('_')[-3] in category_2:
        current_id = first_frame
        nClips = 0
        while current_id <= last_frame - 20:
            startid = current_id
            endid = current_id + 20
            # startid_name = 'image_{:0>4d}.jpg'.format(startid)
            # endid_name = 'image_{:>04d}.jpg'.format(endid)
            check_flag = 1
            for j in range(startid, endid + 1):
                if 'image_{:0>4d}.jpg'.format(j) not in frames:
                    check_flag = 0
            if check_flag == 1:
                startid_path = os.path.join(listOfFolders[i], '%05d' % startid)
                list.append(startid_path)
                nClips = nClips + 1
                current_id = current_id + 3
            else:
                current_id = current_id + 1
        print('nClips', nClips)
    else:
        print('category error!!!')
        break
    sum = sum + nClips
print('sum', sum)

with open('./your_file_2.txt', 'w') as f:
    for item in list:
        f.write("%s\n" % item)
