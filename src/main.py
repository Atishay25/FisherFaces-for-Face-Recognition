import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from load_data import preprocessing, load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--dataset", type=str, default="yale", help="yale or yaleB or harvard or cmu")
    args = parser.parse_args()
    data_path = None
    save_path = None
    print(args.dataset)
    if args.dataset == "yale":
        data_path = "./../data/yale"
        save_path = "./preprocessing"
        image_size = (150,150)
        num_subjects = 15
        num_images = 11
    load_data(data_path, save_path)
    pre_data = preprocessing()

    
    
