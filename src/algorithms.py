import numpy as np

class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self,x):
        n = x.shape[0]
        L = np.dot(x.T, x)/n
        eig_vals, eig_vecs = np.linalg.eig(L)
        sorted_index = np.argsort(eig_vals)[::-1]
        eigenvec = eig_vecs[:, sorted_index]
        W = eigenvec[:, :x.shape[1]]
        V = np.dot(x, W)
        for i in range(V.shape[1]):
            V[:, i] = V[:, i]/np.linalg.norm(V[:, i])
        V_k = V[:, :self.k]
    
