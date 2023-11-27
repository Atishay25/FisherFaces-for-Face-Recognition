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
        self.y_train = np.zeros((105,1), dtype=np.int32)
        self.X_test = np.zeros((60,77760))
        self.y_test = np.zeros((60,1), dtype=np.int32)
        self.X_glasses = np.zeros((30,77760))
        self.y_glasses = np.zeros((30,1), dtype=np.int32)
        
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
                if 'glasses' in f:
                    self.X_glasses[pid,:] = image.flatten()
                    self.y_glasses[pid] = 1
                    num_glasses += 1
                    glass_pids.append(pid)
                elif 'noglasses' in f:
                    self.X_glasses[pid,:] = image.flatten()
                    self.y_glasses[pid] = 0
                    num_glasses += 1
                    glass_pids.append(pid)
        # print(img_read_per_person, num_train, num_test) #[11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11] 105 60
        # here they did argsort, Idk why

class YaleB(object):
    def __init__(self, data_path):
        super(YaleB, self).__init__()
        self.data_path = data_path
        self.n = 2470
        self.X_train = np.zeros((1482,32256))
        self.y_train = np.zeros((1482), dtype=np.int32)
        self.X_test = np.zeros((988,32256))
        self.y_test = np.zeros((988), dtype=np.int32)
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
                    self.X_train[num_train,:] = image.flatten()
                    self.y_train[num_train] = pid
                    num_train += 1
                else:
                    self.X_test[num_test,:] = image.flatten()
                    self.y_test[num_test] = pid
                    num_test += 1
                img_read_per_person[pid] += 1
        #print(img_read_per_person, num_train, num_test)
        #print(img_read_per_person, self.X_train.shape[0], self.X_test.shape[0])
        # sort and take nonzero indices
        #idx = np.argsort(self.y_train)
        #self.X_train = self.X_train[idx,:]
        #self.y_train = self.y_train[idx]
#
        #idx = np.argsort(self.y_test)
        #self.X_test = self.X_test[idx,:]
        #self.y_test = self.y_test[idx]
        #print(img_read_per_person, self.X_train.shape[0], self.X_test.shape[0])

def ld(data_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for f in os.listdir(data_path):
        if not 'txt' in f and not 'DS_Store' in f and not 'zip' in f:
            image = Image.open(data_path + '/' + f)
            subject_num = f.split('.')[0]
            img_path = save_path + '/' + subject_num + '/' + f + '.jpg'
            if not os.path.isdir(save_path + '/' + subject_num):
                os.mkdir(save_path + '/' + subject_num)
            if not os.path.isfile(img_path):
                image.save(img_path)


if __name__ == '__main__':
    #yale = YaleDataset('./../data/yale')
    #yale.load_data()

    yale_B = YaleB('./../data/yaleB')
    yale_B.load_data()
