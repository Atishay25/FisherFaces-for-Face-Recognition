import numpy as np
import scipy

class PCA(object):
    def __init__(self, n_components):
        self.k = n_components

    # fit the train data into PCA
    def fit(self,x, light=False):         # x := n x d
        n = x.shape[0]
        self.x_mean = (np.mean(x, axis=0)).reshape(1,-1)
        x_centered = x - self.x_mean
        L = np.dot(x_centered,x_centered.T)/(n-1)
        eig_vals, eig_vecs = np.linalg.eig(L)
        sorted_index = np.argsort(eig_vals)[::-1]
        eigenvec = eig_vecs[:, sorted_index]
        W = eigenvec[:, :x_centered.shape[1]]
        V = x_centered.T @ W
        for i in range(n):
            V[:,i] = V[:,i]/np.linalg.norm(V[:,i])
        self.V_k = None
        if light:                           # Removing top 3 eigenvectors for lighting
            self.V_k = V[:,3:self.k+3]
        else:
            self.V_k = V[:,:self.k]

    # transform into the reduced dimension k
    def transform(self,x):
        x_centered = x - self.x_mean
        output = -x_centered @ self.V_k
        return output
        
class Linear_discriminant_Analysis:
    def __init__(self, dataset, out_dim=1):
        super(Linear_discriminant_Analysis, self).__init__()
        self.out_dim = out_dim
        self.dataset = dataset

    # fit training data (after PCA) into LDA for Fisherface algorithm
    def fit(self, X, labels):       # X: nxd
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.c = np.max(labels) + 1

        unique_labels = list(np.unique(labels))
        class_mean_faces = []       
        num_i = []      
        x_classed = []
        labels_ = np.reshape(labels, (labels.shape[0]))
        for label in unique_labels:             # Compute mean face for each class
            idxs = (labels_ == label)
            if self.dataset == 'cmu':
                idxs = 1*idxs
            x_classed.append(X[idxs,:])
            num_i.append(np.sum(idxs))
            class_mean_faces.append(np.mean(X[idxs,:],axis=0).reshape(-1,1))
        self.mean_face = np.mean(X, axis=0).reshape(-1,1)               # mean of images of all classes
        
        Sb = np.zeros((self.d, self.d))         # Compute scatter matrices
        Sw = np.zeros((self.d, self.d))

        for i in range(self.c):
            v = class_mean_faces[i] - self.mean_face
            norm_xclass = x_classed[i] - class_mean_faces[i].reshape(1,-1)
            Sb = Sb + (num_i[i]*(v @ v.T))
            Sw = Sw + (norm_xclass.T @ norm_xclass)
        
        eigenvalues, self.W = scipy.linalg.eig(Sb,Sw)
        indices = np.argsort(eigenvalues)[::-1]
        self.all_eigenvalues = eigenvalues[indices]
        indices = indices[:self.out_dim]
        eigenvalues = eigenvalues[indices]
        self.W = self.W[:, indices]

    # Project faces onto the subspace spanned by the top eigenfaces
    def transform(self, X):
        output = np.dot(X-self.mean_face.reshape(1,-1),self.W)
        return output

# Fisherface algorithm
class Fisherfaces:
    def __init__(self,dataset,num_components=1):
        super(Fisherfaces, self).__init__()
        self.num_components = num_components
        self.dataset= dataset
    def fit(self, X, labels):
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.c = labels.max()+1         # number of classes
        self.pca = PCA(n_components=self.n - self.c)
        self.fisher_ld = Linear_discriminant_Analysis(self.dataset,out_dim=self.num_components)
        self.pca.fit(X)
        modified_X = self.pca.transform(X)
        self.fisher_ld.fit(modified_X, labels)
    def transform(self,X):
        return self.fisher_ld.transform(self.pca.transform(X))

# Helper Class to use FisherFace for Face Recognition
class FaceRecognitionFisher(object):
    def __init__(self, out_dim, dataset):
        super(FaceRecognitionFisher, self).__init__()
        self.out_dim = out_dim
        self.dataset = dataset
    def fit(self, X, labels):
        self.fishermodel = Fisherfaces(self.dataset,self.out_dim)
        self.fishermodel.fit(X,labels)
        self.train_embeds = self.fishermodel.transform(X)
        self.labels = labels
    def predict(self, X):
        alpha_p = self.fishermodel.transform(X)     # project input images onto the subspace
        preds = []
        for p in range(X.shape[0]):                 # find nearest neighbor in the subspace, to find similar face
            dist = np.sum((self.train_embeds - alpha_p[p,:])**2,axis=1)
            min_ind = np.argmin(dist)
            preds.append(self.labels[min_ind])
        return np.array(preds)
    
# Helper Class to use PCA for Face Recognition
class FaceRecognitionEigen(object):
    def __init__(self, k):
        self.k = k
    def train(self, x, y_labels, light=False):
        self.eigen_model = PCA(n_components=self.k)
        self.eigen_model.fit(x,light)
        self.alphas = self.eigen_model.transform(x)
        self.y_labels = y_labels
    def predict(self, x):
        alpha_p = self.eigen_model.transform(x)         # project input images onto the eigenspace
        preds = []
        for p in range(x.shape[0]):                     # find nearest neighbor in the subspace, to find similar face for each test image
            dist = np.sum((self.alphas - alpha_p[p,:])**2,axis=1)
            min_ind = np.argmin(dist)
            preds.append(self.y_labels[min_ind])
        return np.array(preds)
    

    
