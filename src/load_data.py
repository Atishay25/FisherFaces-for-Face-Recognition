import os
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.utils.data import Dataset

class YaleDataset(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
        self.n = 165

def load_data(data_path, save_path):
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

def preprocessing():

    train_dataset, test_dataset  = random_split(data, [130,35])

    batch_size = 16

    yaledata = {'train_load' : DataLoader(train_dataset, batch_size = batch_size, shuffle=False),
                'test_load' : DataLoader(test_dataset, batch_size = batch_size)}

    return yaledata