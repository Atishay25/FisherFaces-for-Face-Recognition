import os
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.k = n_components

    def fit(self,x, light):
        n = x.shape[0]
        self.x_mean = np.mean(x, axis=0)
        x_centered = x - self.x_mean
        L = np.dot(x_centered, x_centered.T)/(n-1)
        eig_vals, eig_vecs = np.linalg.eig(L)
        sorted_index = np.argsort(eig_vals)[::-1]
        eigenvec = eig_vecs[:, sorted_index]
        W = eigenvec[:, :x_centered.shape[1]]
        V = np.dot(W, x_centered)
        #print(x.shape, self.x_mean.shape, x_centered.shape, L.shape, W.shape, V.shape)
        for i in range(n):
            V[i, :] = V[i, :]/np.linalg.norm(V[i, :])
        self.V_k = None
        if light:
            self.V_k = V[3:self.k+3, :]
        else:
            self.V_k = V[:self.k, :]

    def transform(self,x):
        x_centered = x - self.x_mean
        output = x_centered @ self.V_k.T
        return output
        

class Linear_discriminant_Analysis:
    def __init__(self, out_dim=1):
        super(Linear_discriminant_Analysis, self).__init__()
        self.out_dim = out_dim
        self.projected_faces = None     # projection of images on eigenspace
        self.mean_face = None           # mean of images of all classes
        self.W = None                   # eigenspace

        def fit(self, X, labels):       # X: nxd
            self.n = X.shape[0]
            self.d = X.shape[1]

            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)    # number of unique lables(classes)

            # Compute mean face for each class
            class_mean_faces = []
            for label in unique_labels:
                class_images = X[labels == label]
                class_mean_face = np.mean(class_images, axis=0)
                class_mean_faces.append(class_mean_face.reshape(-1,1))

            self.mean_face = np.mean(class_mean_faces, axis=0)  # mean of images of all classes

            # Compute scatter matrices
            Sw = np.zeros((self.d, self.d))
            Sb = np.zeros((self.d, self.d))

            for label, class_mean_face in zip(unique_labels, class_mean_faces):
                class_images = X[labels == label]

                diff_within = class_images - class_mean_face
                Sw += (diff_within.T@diff_within)

                diff_between = class_mean_face - self.mean_face
                Sb += len(class_images) * (diff_between@diff_between.T)

            # Solve the generalized eigenvalue problem
            eigenvalues, self.W = np.linalg.eigh(np.linalg.inv(Sw).dot(Sb))     # W is eigenvector matrix

            # Sort eigenvalues and corresponding eigenvectors in descending order
            indices = np.argsort(eigenvalues)[::-1]
            self.all_eigenvalues = eigenvalues[indices]
            self.all_eigenvectors = self.W[:, indices]

            # Select the top eigenfaces
            indices = indices[:self.out_dim]
            eigenvalues = eigenvalues[indices]
            self.W = self.W[:, indices]

            # Project faces onto the subspace spanned by the top eigenfaces
            self.projected_faces = np.dot(X - self.mean_face, self.W)



class Fisherfaces:
    def __init__(self, num_components=1):
        super(Fisherfaces, self).__init__()
        self.num_components = num_components
        self.fisher_ld = None
        self.labels = []
        self.projected_image = None

    def fit(self, X, labels):
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.c = labels.max()+1
        
        self.fisher_ld = Linear_discriminant_Analysis(out_dim=self.num_components)
        ##############  here add code for self.pca = PCA() ###############
        self.pca = PCA(k = self.n - self.c)
        ##############  here add code to change X according to PCA algorithm ###############
        modified_X = self.pca.fit_transform(X)
        self.fisher_ld.fit(modified_X, labels)
        self.labels = labels

    def predict(self, X):
        # Project the input image onto the subspace
        ##############  here add code to change "image" to be transformed according to PCA algorithm ###############
        image = self.pca.transform(X)
        projected_image = np.dot(image - self.fisher_ld.mean_face, self.fisher_ld.W)

        # Find the nearest neighbor in the subspace
        distances = [np.linalg.norm(projected_image - face) for face in self.fisher_ld.projected_faces]
        min_distance_index = np.argmin(distances)

        return self.labels[min_distance_index]
    

    
