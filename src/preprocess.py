#-*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np


def load_data(data_dir='../dataset/TRAIN/digits'):
    '''
    read pic from dir
    :param data_dir
    :return: (X,y) X: np.array(dataset_size, 32*32, dtype = float64);
                    y: np.array(dataset_size, 32*32, dtype = int64);
    '''
    data = []
    labels = []
    f = open('analyse.txt', 'wb')
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if '.jpg' in file:
                if '-' in file:
                    label = file.split('-')[1].split('.jpg')[0][:1]
                else:
                    label = file.split('.jpg')[0][:1]
            elif '[' in file:
                # TEST set
                label = file.split('[')[1][:1]
            else:
                s = file.split('.png')[0]
                if '_' in s:
                    label = s.split('_')[0][:1]
                else:
                    label = s.split('.')[1][:1]

            labels.append(int(label))

            pic = np.array(Image.open(os.path.join(root, file)), dtype='float64')
            if pic.shape == (32, 32, 3):
                # /digits, Li Wanjin, number/
                # print file
                # if not (check_all_True(pic[:, :, 0] == pic[:, :, 1]) and check_all_True(pic[:, :, 2] == pic[:, :, 1])):
                #     print 'error!'
                pic = (pic[:, :, 0]+pic[:, :, 1])/2
            elif pic.shape == (32, 32, 4):
                # /hjk_picture/
                # print file
                # if not (check_all_True(pic[:, :, 3] == 255*np.ones((32,32,1)))):
                #     print 'error!'
                pic = (pic[:, :, 0]+pic[:, :, 1]+pic[:, :, 2])/3
            elif pic.shape == (32, 35, 4):
                pic = (pic[:, 1:33, 0]+pic[:, 1:33, 1]+pic[:, 1:33, 2])/3
            elif pic.shape == (32, 32):
                pass
            else:
                print 'error shape!'
                print file, pic.shape
            pic = pic.reshape(32 * 32, )
            data.append(pic)
            f.write(file+':'+label+'\n'+str(pic)+'\n')
    f.close()
    return np.array(data), np.array(labels)


def check_all_True(nparray):
    flag = True
    for ele in nparray.flat:
        if ele == False:
            flag = False
            break
    return flag

if __name__ == '__main__':
    np.set_printoptions(threshold=np.NaN)
    # npd = load_data()
    # np.savez("data.npz", X=npd[0], y=npd[1])
    X, y = load_data('../dataset/TRAIN/digits')
    np.savez("data0.npz", X=X, y=y)
    X, y = load_data('../dataset/TRAIN/hjk_picture')
    np.savez("data1.npz", X=X, y=y)
    X, y = load_data('../dataset/TRAIN/Li Wanjin')
    np.savez("data2.npz", X=X, y=y)
    X, y = load_data('../dataset/TRAIN/number')
    np.savez("data3.npz", X=X, y=y)
    # X,y = load_data('../dataset/TEST')
    # np.savez("test.npz", X=X, y=y)
