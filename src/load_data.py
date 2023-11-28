import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class YaleDataset(object):
    def __init__(self, data_path):
        super(YaleDataset, self).__init__()
        self.data_path = data_path
        self.n = 165
        self.X_train = np.zeros((105,77760))
        self.y_train = np.zeros((105), dtype=np.int32)
        self.X_test = np.zeros((60,77760))
        self.y_test = np.zeros((60), dtype=np.int32)
        self.X_glasses = np.zeros((30,77760))
        self.y_glasses = np.zeros((30), dtype=np.int32)
        
    def load_data(self):
        train_indices = [None]*15
        img_read_per_person = [0]*15
        num_train = 0
        num_test = 0
        glass_pids = []
        num_glasses = 0
        for i in range(15):     # 11 images split into 7 for train and 4 for test
            train_indices[i] = list(np.random.choice(11, 7, replace=False))
        for f in os.listdir(self.data_path):
            if not 'txt' in f and not 'DS_Store' in f and not 'zip' in f:
                img_path = os.path.join(self.data_path, f)
                image = plt.imread(img_path)
                pid = int(f.split('.')[0].split('subject')[1]) - 1
                if img_read_per_person[pid] in train_indices[pid]:
                    self.X_train[num_train,:] = image.flatten()
                    self.y_train[num_train] = pid
                    num_train += 1
                else:
                    self.X_test[num_test,:] = image.flatten()
                    self.y_test[num_test] = pid
                    num_test += 1
                img_read_per_person[pid] += 1
                if 'noglasses' in f:
                    self.X_glasses[num_glasses,:] = image.flatten()
                    self.y_glasses[num_glasses] = 0
                    num_glasses += 1
                    glass_pids.append(pid)
                elif 'glasses' in f:
                    self.X_glasses[num_glasses,:] = image.flatten()
                    self.y_glasses[num_glasses] = 1
                    num_glasses += 1
                    glass_pids.append(pid)
        # print(img_read_per_person, num_train, num_test) #[11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11] 105 60
        # here they did argsort, Idk why
        #self.X_test = (self.X_test).T
        #self.X_train = (self.X_train).T

        #idxs = np.argsort(self.y_train)
        #self.y_train = self.y_train[idxs]
        #self.X_train = self.X_train[idxs,:]
#
        #idxs = np.argsort(self.y_test)
        #self.y_test = self.y_test[idxs]
        #self.X_test = self.X_test[idxs,:]
#
        idxs = np.argsort(self.y_glasses)
        self.y_glasses = self.y_glasses[idxs]
        self.X_glasses = self.X_glasses[idxs,:]

class YaleB(object):
    def __init__(self, data_path):
        super(YaleB, self).__init__()
        self.data_path = data_path
        self.n = 2470
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
    def load_data(self):
        train_indices = [None]*38
        img_read_per_person = [0]*38
        num_train = 0
        num_test = 0
        for i in range(38):     # 64 images split into 39 for train and 26 for test
            train_indices[i] = list(np.random.choice(65, 39, replace=False))
        for p_dir in os.listdir(self.data_path):
            p_path = os.path.join(self.data_path, p_dir)
            for f in os.listdir(p_path):
                img_path = os.path.join(p_path, f)
                image = plt.imread(img_path)
                pid = int(p_dir[5:]) - 1
                if pid > 13:
                    pid -= 1
                if img_read_per_person[pid] in train_indices[pid]:
                    #self.X_train[num_train,:] = image.reshape(-1)
                    #self.y_train[num_train] = pid
                    self.X_train.append(image.reshape(-1))
                    self.y_train.append(pid)
                    num_train += 1
                else:
                    #self.X_test[num_test,:] = image.reshape(-1)
                    #self.y_test[num_test] = pid
                    self.X_test.append(image.reshape(-1))
                    self.y_test.append(pid)
                    num_test += 1
                img_read_per_person[pid] += 1
        #idx = np.argsort(self.y_train)
        #self.X_train = self.X_train[idx,:]
        #self.y_train = self.y_train[idx]
        #idx = np.argsort(self.y_test)
        #self.X_test = self.X_test[idx,:]
        #self.y_test = self.y_test[idx]
        self.X_test = np.array(self.X_test)
        self.X_train = np.array(self.X_train)
        self.y_test = np.array(self.y_test)
        self.y_train = np.array(self.y_train)
        idxs = np.argsort(self.y_train)
        self.y_train = self.y_train[idxs]
        self.X_train = self.X_train[idxs,:]
        idxs = np.argsort(self.y_test)
        self.y_test = self.y_test[idxs]
        self.X_test = self.X_test[idxs,:]

class CMU_Dataset(object):
    def __init__(self, data_path):
        super(CMU_Dataset, self).__init__()
        self.data_path = data_path
        self.X_train = np.zeros((1, 15360))
        self.y_train = []
        self.X_test = np.zeros((1, 15360))
        self.y_test = []
        self.c = 0


    def readpgm(self, name):
        with open(name) as f:
            lines = f.readlines()
        # here,it makes sure it is ASCII format (P2)
        assert lines[0].strip() == 'P2' 
        # Converts data to a list of integers
        data = []
        for line in lines[1:]:
            data.extend([int(c) for c in line.split()])
        return (np.array(data[3:]),(data[1],data[0]),data[2])


    def load_data(self):
        np.random.seed(0)

        for dir1 in os.listdir(self.data_path):
            choice = np.random.permutation(list(range(0, 32)))
            i=0
            if not os.path.isdir(os.path.join(self.data_path, dir1)):
                continue
            for file in os.listdir(os.path.join(self.data_path, dir1)):
                image_path = os.path.join(self.data_path, dir1,  file)
                image, img_size, img_max = self.readpgm(image_path)
                image = np.resize(image,(1, 15360))
                image = image.astype('float32')
                if choice[i]%3==0:
                    self.X_test = np.concatenate((self.X_test, image), axis = 0)
                    self.y_test.append(self.c)
                else:
                    self.X_train = np.concatenate((self.X_train, image), axis = 0)
                    self.y_train.append(self.c)
                i=i+1
            self.c=self.c+1
        self.X_train = self.X_train[1:]
        self.X_test = self.X_test[1:]
        self.y_test = np.array(self.y_test)
        self.y_train = np.array(self.y_train)
        #return self.X_train[1:] , np.array(self.y_train), self.X_test[1:] , np.array(self.y_test)

