import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from load_data import YaleDataset, YaleB
from algorithms import PCA, Fisherfaces
import random


class FaceRecognitionEigen(object):
    def __init__(self, k):
        self.k = k

    def train(self, x, y_labels, light):
        self.eigen_model = PCA(n_components=self.k)
        self.eigen_model.fit(x,light)
        self.alphas = self.eigen_model.transform(x)
        self.y_labels = y_labels

    def predict(self, x):
        eigen_coeff = np.expand_dims(self.alphas, 1)
        test_alpha = np.expand_dims(self.eigen_model.transform(x), 0)
        diff = np.sum(((test_alpha - eigen_coeff)**2),axis=2)
        pred_idx = np.argmin(diff, 0)
        print(diff.shape, "diff")
        return self.y_labels[pred_idx]
        
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--dataset", type=str, default="yale", help="yale or yaleB or harvard or cmu")
    args = parser.parse_args()
    data_path = ""
    dataset = None
    print("Dataset:",args.dataset)
    if args.dataset == "yale":
        data_path = "./../data/yale"
        dataset = YaleDataset(data_path)
    elif args.dataset == "yaleB":
        data_path = "./../data/yaleB"
        dataset = YaleB(data_path)
    elif args.dataset == "cmu":
        data_path = "./../data/cmu"
    else:
        print("Invalid dataset")
        exit()
    dataset.load_data()
    model = FaceRecognitionEigen(30)
    model.train(dataset.X_train, dataset.y_train, light=False)
    preds = model.predict(dataset.X_test)
    print(preds.shape)
    print(dataset.y_test.shape)
    eigen_acc = (np.sum(preds == dataset.y_test))/dataset.y_test.shape[0]
    print(1-eigen_acc)
    
    
    
