import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from load_data import YaleDataset, YaleB, CMU_Dataset
from algorithms import PCA, FaceRecognitionFisher
import random

random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)

class FaceRecognitionEigen(object):
    def __init__(self, k):
        self.k = k

    def train(self, x, y_labels, light=False):
        self.eigen_model = PCA(n_components=self.k)
        self.eigen_model.fit(x,light)
        self.alphas = self.eigen_model.transform(x)
        self.y_labels = y_labels

    def predict(self, x):
        #eigen_coeff = np.expand_dims(self.alphas, 1)
        #test_alpha = np.expand_dims(self.eigen_model.transform(x), 0)
        #diff = np.sum(((test_alpha - eigen_coeff)**2),axis=2)
        #pred_idx = np.argmin(diff, 0)
        #return self.y_labels[pred_idx]
        alpha_p = self.eigen_model.transform(x)
        preds = []
        for p in range(x.shape[0]):
            dist = np.sum((self.alphas - alpha_p[p,:])**2,axis=1)
            min_ind = np.argmin(dist)
            preds.append(self.y_labels[min_ind])
        return np.array(preds)
    
def error_rate(y_pred, y_true):
    return (np.sum(y_pred != y_true))/y_true.shape[0]
    
def eval_all(x_train, y_train, x_test, y_test, params):
    eigen_model = FaceRecognitionEigen(params['eigen'])
    eigen_model.train(x_train, y_train)
    y_pred = eigen_model.predict(x_test)
    error_eigen = error_rate(y_pred, y_test)
    eigen_model_light = FaceRecognitionEigen(params['eigen_light'])
    eigen_model_light.train(x_train, y_train, light=True)
    y_pred = eigen_model_light.predict(x_test)
    error_eigen_light = error_rate(y_pred, y_test)
    fisher_model = FaceRecognitionFisher(params['fisher'])
    fisher_model.fit(x_train, y_train)
    y_pred = fisher_model.predict(x_test)
    error_fisher = error_rate(y_pred, y_test)
    return error_eigen, error_eigen_light, error_fisher
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--dataset", type=str, default="yale", help="yale or yaleB or harvard or cmu")
    args = parser.parse_args()
    data_path = ""
    dataset = None
    print("Dataset:",args.dataset)
    params = {}
    if args.dataset == "yale":
        data_path = "./../data/yale"
        dataset = YaleDataset(data_path)
        params = {'eigen': 60, 'eigen_light': 60, 'fisher': 15}
    elif args.dataset == "yaleB":
        data_path = "./../data/yaleB"
        dataset = YaleB(data_path)
        params = {'eigen': 60, 'eigen_light': 60, 'fisher': 38}
    elif args.dataset == "cmu":
        data_path = "./../data/cmu"
        dataset = CMU_Dataset(data_path)
        params = {'eigen': 60, 'eigen_light': 60, 'fisher': 25}
    else:
        print("Invalid dataset")
        exit()
    dataset.load_data()
    eigen_error, eigen_light_error, fisher_error = eval_all(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test, params)
    print("ERROR RATES -")
    print("-------------------------------------")
    print("EigenFaces : \t\t\t","{:.6f}".format(100*eigen_error))
    print("Eigenfaces (Leaving Top 3) : \t", "{:.6f}".format(100*eigen_light_error))
    print("FisherFaces : \t\t\t", "{:.6f}".format(100*fisher_error))
    if args.dataset == 'yale':
        glass_fe = 0
        glass_ee = 0
        glass_ele = 0
        params = {'eigen': 10, 'eigen_light': 10, 'fisher': 10}
        x_g = dataset.X_glasses
        y_g = dataset.y_glasses
        n_g = y_g.shape[0]
        for i in range(n_g//2):
            leave1_x = np.delete(x_g, [2*i, 2*i + 1], 0)
            leave1_y = np.delete(y_g, [2*i, 2*i + 1], 0)
            ee, ele, fe = eval_all(leave1_x, leave1_y, x_g[2*i:(2*i + 1),:], y_g[2*i:(2*i + 1)],params=params)
            glass_ee += 2*ee
            glass_ele += 2*ele
            glass_fe += 2*fe

        glass_fe /= n_g
        glass_ee /= n_g
        glass_ele /= n_g
        print("")
        print("Glass Recognition Error rates (calculated using Leaving one out) -")
        print("-------------------------------------")
        print("EigenFaces : \t\t\t","{:.6f}".format(100*glass_ee))
        print("Eigenfaces (Leaving Top 3) : \t", "{:.6f}".format(100*glass_ele))
        print("FisherFaces : \t\t\t", "{:.6f}".format(100*glass_fe))
    ##model = FaceRecognitionEigen(60)
    ##model.train(dataset.X_train, dataset.y_train, light=True)
    #model = FaceRecognitionFisher(38)
    #model.fit(dataset.X_train, dataset.y_train)
    #preds = model.predict(dataset.X_test)
    ##for i in range(len(preds)):
    ##    print(preds[i], dataset.y_test[i])
    #eigen_acc = (np.sum(preds == dataset.y_test))/dataset.y_test.shape[0]
    #print((1-eigen_acc))
    #
    
    
