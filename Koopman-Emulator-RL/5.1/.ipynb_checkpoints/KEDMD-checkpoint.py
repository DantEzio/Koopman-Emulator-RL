# -*- coding: utf-8 -*-
"""
@author: chong
"""
import numpy as np
import scipy.linalg as linalg


class KEDMD:
    def __init__(self, x, y, lam, kernel_function):
        self.x, self.y, self.kernel_function = x, y, kernel_function
        
        dataset_size = x.shape[0]
        lam_i = lam * np.mat(np.identity(dataset_size))
        Gxx = np.mat(np.zeros((dataset_size, dataset_size)))
        for i in range(0, dataset_size):
            for j in range(0, dataset_size):
                Gxx[i, j] = kernel_function(x[i,:], x[j,:])
        
        #k=np.exp(-cdist(x, x))
        Gxy=y
        self.K = np.linalg.inv(lam_i + Gxx) * Gxy
        print(self.K.shape)

    def decom(self):
        self.u,self.s,self.v=linalg.svd(self.K,full_matrices=False)
        #y=X_test.dot(u).dot(np.diag(s)).dot(v)
    
    def predict(self, x):
        prediction_y = np.zeros((x.shape[0],self.y.shape[1]))
        for i in range(0, x.shape[0]):
            kappa = np.mat(np.zeros(self.x.shape[0])).T
            for j in range(0, self.x.shape[0]):
                kappa[j,0] = self.kernel_function(self.x[j], x[i])
            #print(kappa.shape,self.coe.shape)
            prediction_y[i] = self.K.T.dot(kappa).reshape((1,-1))
        return prediction_y