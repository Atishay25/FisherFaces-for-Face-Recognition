import numpy as np
import argparse
from load_data import YaleDataset, YaleB, CMU_Dataset
from algorithms import FaceRecognitionFisher, FaceRecognitionEigen

np.random.seed(0)
    
def error_rate(y_pred, y_true):             # error rate = (1 - recognition rate)
    return (np.sum(1*(y_pred != y_true)))/y_true.shape[0]
    
def eval_all(x_train, y_train, x_test, y_test, params,dataset):         # evaluate a given dataset on all algorithms
    eigen_model = FaceRecognitionEigen(params['eigen'])
    eigen_model.train(x_train, y_train)
    y_pred = eigen_model.predict(x_test)
    error_eigen = error_rate(y_pred, y_test)

    eigen_model_light = FaceRecognitionEigen(params['eigen_light'])
    eigen_model_light.train(x_train, y_train, light=True)
    y_pred = eigen_model_light.predict(x_test)
    error_eigen_light = error_rate(y_pred, y_test)

    fisher_model = FaceRecognitionFisher(dataset=dataset,out_dim=params['fisher'])
    fisher_model.fit(x_train, y_train)
    y_pred = fisher_model.predict(x_test)
    error_fisher = error_rate(y_pred, y_test)

    return error_eigen, error_eigen_light, error_fisher
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("--dataset", type=str, default="yale", help="Provide dataset: yale or yaleB or cmu",choices=['yale', 'yaleB', 'cmu'])
    args = parser.parse_args()
    data_path = ""
    dataset = None
    print("Dataset:",args.dataset)
    params = {}
    if args.dataset == "yale":
        data_path = "./../data/yale"
        dataset = YaleDataset(data_path)
        params = {'eigen': 50, 'eigen_light': 50, 'fisher': 15}
    elif args.dataset == "yaleB":
        data_path = "./../data/yaleB"
        dataset = YaleB(data_path)
        params = {'eigen': 50, 'eigen_light': 50, 'fisher': 38}
    elif args.dataset == "cmu":
        data_path = "./../data/cmu"
        dataset = CMU_Dataset(data_path)
        params = {'eigen': 50, 'eigen_light': 50, 'fisher': 15}
    else:
        print("Invalid dataset")
        exit()
    dataset.load_data()
    eigen_error, eigen_light_error, fisher_error = eval_all(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test, params,args.dataset)
    print("ERROR RATES -")
    print("-------------------------------------")
    print("EigenFaces : \t\t\t","{:.3f}".format(100*eigen_error))
    print("Eigenfaces (Leaving Top 3) : \t", "{:.3f}".format(100*eigen_light_error))
    print("FisherFaces : \t\t\t", "{:.3f}".format(100*fisher_error))
    if args.dataset == 'yale':          # Perform Glass Recognition for Yale
        glass_fe = 0
        glass_ee = 0
        glass_ele = 0
        params = {'eigen': 10, 'eigen_light': 10, 'fisher': 1}
        x_g = dataset.X_glasses
        y_g = dataset.y_glasses
        n_g = y_g.shape[0]
        for i in range(n_g//2):         # Using "Leave One Out" Method to evaluate for Glass Recognition
            leave1_x = np.delete(x_g, [2*i, 2*i + 1], 0)
            leave1_y = np.delete(y_g, [2*i, 2*i + 1], 0)
            ee, ele, fe = eval_all(leave1_x, leave1_y, x_g[2*i:(2*i + 1),:], y_g[2*i:(2*i + 1)],params,args.dataset)
            glass_ee += ee
            glass_ele += ele
            glass_fe += fe
        glass_fe /= n_g
        glass_ee /= n_g
        glass_ele /= n_g
        print("")
        print("Glass Recognition Error rates (calculated using Leaving one out) -")
        print("-------------------------------------")
        print("EigenFaces : \t\t\t","{:.3f}".format(100*glass_ee))
        print("Eigenfaces (Leaving Top 3) : \t", "{:.3f}".format(100*glass_ele))
        print("FisherFaces : \t\t\t", "{:.3f}".format(100*glass_fe))

    
