import os
import numpy as np
import scipy

class PCA:
    def __init__(self, n_components):
        self.k = n_components
    def fit(self,x, light=False):         # x := n x d
        n = x.shape[0]
        self.x_mean = (np.mean(x, axis=0)).reshape(1,-1)
        x_centered = x - self.x_mean
        L = np.dot(x_centered,x_centered.T)/(n-1)
        eig_vals, eig_vecs = np.linalg.eig(L)
        '''
        sorted_index = np.argsort(eig_vals)[::-1]
        if light:
            sorted_index = sorted_index[3:self.k+3]
        else:
            sorted_index = sorted_index[:self.k]
        V = x.T @ eig_vecs
        for i in range(n):
            V[i, :] = V[i, :]/(np.linalg.norm(V[i, :])+1e-8)
        self.V_k = V[:,sorted_index]
        '''
        sorted_index = np.argsort(eig_vals)[::-1]
        eigenvec = eig_vecs[:, sorted_index]
        W = eigenvec[:, :x_centered.shape[1]]
        V = x_centered.T @ W
        #print(x.shape, self.x_mean.shape, x_centered.shape, L.shape, W.shape, V.shape)
        for i in range(n):
            V[:,i] = V[:,i]/np.linalg.norm(V[:,i])
        self.V_k = None
        if light:
            self.V_k = V[:,3:self.k+3]
        else:
            self.V_k = V[:,:self.k]

    def transform(self,x):
        x_centered = x - self.x_mean
        output = -x_centered @ self.V_k
        return output
        

class Linear_discriminant_Analysis:
    def __init__(self, out_dim=1):
        super(Linear_discriminant_Analysis, self).__init__()
        self.out_dim = out_dim

    def fit(self, X, labels):       # X: nxd
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.c = np.max(labels) + 1

        unique_labels = list(np.unique(labels))
        num_classes = len(unique_labels)    # number of unique lables(classes)
        # Compute mean face for each class
        class_mean_faces = []       # Mui
        num_i = []      # Ni
        x_classed = []
        labels_ = np.reshape(labels, (labels.shape[0]))
        for label in unique_labels:
            idxs = (labels_ == label)
            x_classed.append(X[idxs,:])
            num_i.append(np.sum(idxs))
            class_mean_faces.append(np.mean(X[idxs,:],axis=0).reshape(-1,1))
        self.mean_face = np.mean(X, axis=0).reshape(-1,1)  # mean of images of all classes
        # Compute scatter matrices
        Sb = np.zeros((self.d, self.d))
        Sw = np.zeros((self.d, self.d))

        #for label in unique_labels:
        #    idxs = (labels_ == label)
        #    class_images = X[idxs,:]
        #    num_imgs = np.sum(idxs)
        #    mean_cimg = np.mean(class_images, axis=0).reshape(-1,1)
        #    v =
#
        for i in range(self.c):
            v = class_mean_faces[i] - self.mean_face
            norm_xclass = x_classed[i] - class_mean_faces[i].reshape(1,-1)
            Sb += (num_i[i]*(v @ v.T))
            Sw += (norm_xclass.T @ norm_xclass)

        #for label, class_mean_face in zip(unique_labels, class_mean_faces):
        #    class_images = X[labels_ == label,:]
        #    diff_within = class_images - class_mean_face.reshape(1,-1)
        #    Sw = Sw + (diff_within.T@diff_within)
        #    diff_between = class_mean_face - self.mean_face
        #    Sb = Sb +  (np.sum(labels_ == label) * (diff_between @ diff_between.T))

        # Solve the generalized eigenvalue problem
        #eigenvalues, self.W = np.linalg.eigh(np.linalg.inv(Sw).dot(Sb))     # W is eigenvector matrix
        
        eigenvalues, self.W = scipy.linalg.eig(Sb,Sw)
        # Sort eigenvalues and corresponding eigenvectors in descending order
        indices = np.argsort(eigenvalues)[::-1]
        self.all_eigenvalues = eigenvalues[indices]
        #self.all_eigenvectors = self.W[:, indices]

        # Select the top eigenfaces
        indices = indices[:self.out_dim]
        eigenvalues = eigenvalues[indices]
        self.W = self.W[:, indices]

        # Project faces onto the subspace spanned by the top eigenfaces
        #self.projected_faces = np.dot(X - self.mean_face.reshape(1,-1), self.W)
    def transform(self, X):
        output = np.dot(X-self.mean_face.reshape(1,-1),self.W)
        return output


class Fisherfaces:
    def __init__(self, num_components=1):
        super(Fisherfaces, self).__init__()
        self.num_components = num_components

    def fit(self, X, labels):
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.c = labels.max()+1
        ##############  here add code for self.pca = PCA() ###############
        self.pca = PCA(n_components=self.n - self.c)
        self.fisher_ld = Linear_discriminant_Analysis(out_dim=self.num_components)
        ##############  here add code to change X according to PCA algorithm ###############
        self.pca.fit(X)
        modified_X = self.pca.transform(X)
        self.fisher_ld.fit(modified_X, labels)
    def transform(self,X):
        return self.fisher_ld.transform(self.pca.transform(X))

class FaceRecognitionFisher(object):
    def __init__(self, out_dim):
        super(FaceRecognitionFisher, self).__init__()
        self.out_dim = out_dim
    def fit(self, X, labels):
        self.fishermodel = Fisherfaces(self.out_dim)
        self.fishermodel.fit(X,labels)
        self.train_embeds = self.fishermodel.transform(X)
        self.labels = labels
    def predict(self, X):
        # Project the input image onto the subspace
        ##############  here add code to change "image" to be transformed according to PCA algorithm ###############
        #image = self.pca.transform(X)
        #projected_image = np.dot(image - self.fisher_ld.mean_face, self.fisher_ld.W)
#
        ## Find the nearest neighbor in the subspace
        #distances = [np.linalg.norm(projected_image - face) for face in self.fisher_ld.projected_faces]
        #min_distance_index = np.argmin(distances)
        #test_alpha = np.expand_dims(self.fishermodel.transform(X), 0)
        #train_alpha = np.expand_dims(self.train_embeds, 1)
        #dists = np.sum((train_alpha-test_alpha)**2, axis=2)
        #min_distance_index = np.argmin(dists,axis=0)
        #return self.labels[min_distance_index]
        alpha_p = self.fishermodel.transform(X)
        preds = []
        for p in range(X.shape[0]):
            dist = np.sum((self.train_embeds - alpha_p[p,:])**2,axis=1)
            min_ind = np.argmin(dist)
            preds.append(self.labels[min_ind])
        return np.array(preds)
    

    
