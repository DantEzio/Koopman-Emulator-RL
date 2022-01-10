# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:53:19 2020

@author: chong
"""

import numpy as np
import scipy.linalg as linalg
from scipy.spatial.distance import pdist, cdist

import matplotlib.pyplot as plt

import pandas as pd

#matrix decomposition tools
# return U,s,V   A=U*diag(s)*V'
def truncated_svd(A,m=np.inf):
    m=min((m,)+A.shape)
    U,s,Vh=linalg.svd(A)
    tol=max(A.shape)*np.spacing(s[0])
    m=min(m,np.count_nonzero(s>tol))
    return U[:,:m],s[:m],Vh[:m].T

# return U,s   A=U*diag(s)*U'
def truncated_svd_sym(A,m=np.inf):
    m=min(m,A.shape[0])
    S,U=linalg.schur(A)
    s=np.diag(S)
    abs_s=np.absolute(s)
    tol=A.shape[0]*np.spacing(abs_s.max())
    m=min(m,np.count_nonzero(abs_s>tol))
    idx=(-abs_s).argsort()[:m]
    return U[:,idx],s[idx]

# return U,s   A=U*diag(s)*U'
def truncated_svd_psd(A,m=np.inf,eps=0):
    m=min(m,A.shape[0])
    S,U=linalg.schur(A)
    s=np.diag(S)
    tol=A.shape[0]*np.spacing(np.absolute(s).max())
    mi=s.min()
    if mi<0 and -mi>tol:
        tol=-mi
    tol = np.maximum(tol, np.absolute(s).max()*eps)
    m=min(m,np.count_nonzero(s>tol))
    idx=(-s).argsort()[:m]
    return U[:,idx],s[idx]

# return R   A=R*R'
def cholcov(A,m=np.inf):
    U,s=truncated_svd_psd(A,m)
    return U*np.sqrt(s)

# return pinv(R)   A=R*R'
def pinv_cholcov(A,m=np.inf,eps=0):
    U,s=truncated_svd_psd(A,m)
    tol=np.absolute(s).max()*eps
    mm=min(m,np.count_nonzero(s>tol))
    return (U[:,:mm]*(1.0/np.sqrt(s[:mm]))).T

# The new observable is U'*f
def truncated_eig(K, m=np.inf, return_reconstruct_K=False):
    m = min(m, K.shape[0])
    w, Q = linalg.eig(K) # K = Q.dot(w).dot(linalg.inv(Q))
    idx = np.argsort(np.abs(w))[::-1]
    w = w[idx]
    Q = Q[:, idx]
    iQ = linalg.inv(Q)
    K_hat = (Q[:, :m] * w[:m]).dot(iQ[:m, :])
    if not np.allclose(np.real(K_hat), K_hat):
        for n in range(m + 1, K.shape[0] + 1):
            K_hat = (Q[:, :n] * w[:n]).dot(iQ[:n])
            if np.allclose(np.real(K_hat), K_hat):
                break 
    K_hat = np.real(K_hat)
    U, s, V = truncated_svd(K_hat)
    #print(linalg.norm(K_hat - (U*s).dot(V.T)))
    K_new = (U.T.dot(V * s)).T
    if return_reconstruct_K:
        return K_new, U, K_hat
    return K_new, U


class kernel_edmd():
    def __init__(self, input_dim):
        '''
        :input_dim: dimension of data
        :gramian_function: gramian_function(X) returns Gramian matrix of (X, X), gramian_function(X, Y) returns Gramian matrix of (X, Y),
        '''
        self.input_dim = input_dim
        #self.gramian_function = gramian_function
        
    #kernel
    def gramian_function(self, X, Y=None):
        if Y is None:
            return np.exp(-(cdist(X, X) / 1.5)**2)
        else:
            return np.exp(-(cdist(X, Y) / 1.5)**2)    
    
    def feature_mapping(self, X):
        return self.gramian_function(X, self.X_train)
    
    def whiten(self, X, eps=0, return_whitened_data=False):
        #whitening transformation: (Chi-self.whiten_mean).dot(self.whiten_T.T)
        Chi = self.feature_mapping(X)
        m = np.mean(Chi, axis=0)
        C = (Chi - m).T.dot(Chi - m)/X.shape[0]
        T = pinv_cholcov(C, eps=eps)
        self.whiten_mean = m
        self.whiten_T = T
        self.feature_dim = T.shape[1]
        if return_whitened_data:
            return np.hstack([np.ones([Chi.shape[0], 1]), (Chi - m).dot(T.T)])
    
    def whitened_feature_mapping(self, X):
        return np.hstack([np.ones([X.shape[0], 1]), (self.feature_mapping(X)-self.whiten_mean).dot(self.whiten_T.T)])
    
    def train(self, X, Y, eps=0.):
        N = X.shape[0]
        self.X_train = X.copy()
        self.Y_train = Y.copy()        
        Chi_0 = self.whiten(X, eps=eps, return_whitened_data=True)
        Chi_1 = self.whitened_feature_mapping(Y)
        self.K = Chi_0.T.dot(Chi_1) / N
    
    def prediction(self, x0, m, T, obs_mapping=None, nontrivial_mode_num=np.inf, return_std=False):
        """
        Predict the mean value of the trajectory of the observable from time 1 to time T with x(0)=x0
        :x0: start point
        :T: trajectory length
        :obs_mapping: The observable function from (N * input_dim) array to (N * ...) array. If None, it is identity mapping
        :nontrivial_mode_num: Number of non-trivial singular functions
        :return_std: If return estimated std
        :return: Trajectory of the observable, or (traj, traj_std)
        """
        N = self.Y_train.shape[0]
        m = min(m + 1, self.K.shape[0])
        K, U = truncated_eig(self.K, m)
        m = K.shape[0]
        Chi_0 = self.whitened_feature_mapping(self.X_train).dot(U)
        
        if obs_mapping is None:
            O = self.Y_train
        else:
            O = obs_mapping(self.Y_train)
        if return_std:
            O = np.hstack([O, O ** 2])
        
        G = Chi_0.T.dot(O) / N
        
        f0 = self.whitened_feature_mapping(x0.reshape(1, -1)).dot(U).T

        traj = np.empty([T, O.shape[1]])
        for t in range(T):
            traj[t] = (G.T.dot(np.linalg.matrix_power(K.T, t))).dot(f0).reshape(-1)
        if return_std:
            traj_std = np.sqrt(np.maximum(traj[:, (int)(traj.shape[1]/2):] - traj[:, :(int)(traj.shape[1]/2)] ** 2, 0))
            return traj[:, :(int)(traj.shape[1]/2)], traj_std,m
        return traj,m

class dmd():
    def __init__(self, input_dim):
        '''
        :input_dim: dimension of data
        :gramian_function: gramian_function(X) returns Gramian matrix of (X, X), gramian_function(X, Y) returns Gramian matrix of (X, Y),
        '''
        self.input_dim = input_dim
        #self.gramian_function = gramian_function
        
    #kernel
    def gramian_function(self, X, Y=None):
        if Y is None:
            return np.exp(-(cdist(X, X) / 1.5)**2)
        else:
            return np.exp(-(cdist(X, Y) / 1.5)**2)    
    
    def feature_mapping(self, X):
        return X#self.gramian_function(X, self.X_train)
    
    def whiten(self, X, eps=0, return_whitened_data=False):
        #whitening transformation: (Chi-self.whiten_mean).dot(self.whiten_T.T)
        Chi = self.feature_mapping(X)
        m = np.mean(Chi, axis=0)
        C = (Chi - m).T.dot(Chi - m)/X.shape[0]
        T = pinv_cholcov(C, eps=eps)
        self.whiten_mean = m
        self.whiten_T = T
        self.feature_dim = T.shape[1]
        if return_whitened_data:
            return np.hstack([np.ones([Chi.shape[0], 1]), (Chi - m).dot(T.T)])
    
    def whitened_feature_mapping(self, X):
        return np.hstack([np.ones([X.shape[0], 1]), (self.feature_mapping(X)-self.whiten_mean).dot(self.whiten_T.T)])
    
    def train(self, X, Y, eps=0.):
        N = X.shape[0]
        self.X_train = X.copy()
        self.Y_train = Y.copy()        
        Chi_0 = self.whiten(X, eps=eps, return_whitened_data=True)
        Chi_1 = self.whitened_feature_mapping(Y)
        self.K = Chi_0.T.dot(Chi_1) / N
    
    def prediction(self, x0, m, T, obs_mapping=None, nontrivial_mode_num=np.inf, return_std=False):
        """
        Predict the mean value of the trajectory of the observable from time 1 to time T with x(0)=x0
        :x0: start point
        :T: trajectory length
        :obs_mapping: The observable function from (N * input_dim) array to (N * ...) array. If None, it is identity mapping
        :nontrivial_mode_num: Number of non-trivial singular functions
        :return_std: If return estimated std
        :return: Trajectory of the observable, or (traj, traj_std)
        """
        N = self.Y_train.shape[0]
        m = min(m + 1, self.K.shape[0])
        K, U = truncated_eig(self.K, m)
        m = K.shape[0]
        Chi_0 = self.whitened_feature_mapping(self.X_train).dot(U)
        
        if obs_mapping is None:
            O = self.Y_train
        else:
            O = obs_mapping(self.Y_train)
        if return_std:
            O = np.hstack([O, O ** 2])
        
        G = Chi_0.T.dot(O) / N
        
        f0 = self.whitened_feature_mapping(x0.reshape(1, -1)).dot(U).T

        traj = np.empty([T, O.shape[1]])
        for t in range(T):
            traj[t] = (G.T.dot(np.linalg.matrix_power(K.T, t))).dot(f0).reshape(-1)
        if return_std:
            traj_std = np.sqrt(np.maximum(traj[:, (int)(traj.shape[1]/2):] - traj[:, :(int)(traj.shape[1]/2)] ** 2, 0))
            return traj[:, :(int)(traj.shape[1]/2)], traj_std,m
        return traj,m


class linear_mmd_dmd():
    
    def __init__(self, input_dim, feature_mapping):
        '''
        :input_dim: dimension of data
        :feature mapping: A function from (N * input_dim) array to (N * feature_dim) array
        :gramian_function: gramian_function(X) returns Gramian matrix of (X, X), gramian_function(X, Y) returns Gramian matrix of (X, Y),
        '''
        self.input_dim = input_dim
        self.feature_mapping = feature_mapping
        #self.gramian_function = gramian_function
    
    #kernel
    def gramian_function(self, X, Y=None):
        if Y is None:
            return np.exp(-(cdist(X, X) / 1.5)**2)
        else:
            return np.exp(-(cdist(X, Y) / 1.5)**2)
    
    #estimation of bandwidth for kernel
    #just for reference
    def estimate_kernel_width(self,Y):
        N = Y.shape[0]
        M = 10000
        return np.sqrt(np.median(np.sum((Y[np.random.randint(N, size=M)] - Y[np.random.randint(N, size=M)])**2, axis=1)))
    
    def whiten(self, X, eps=0, return_whitened_data=False):
        #whitening transformation: (Chi-self.whiten_mean).dot(self.whiten_T.T)
        Chi = self.feature_mapping(X)
        m = np.mean(Chi, axis=0)
        C = (Chi - m).T.dot(Chi - m)/X.shape[0]
        T = pinv_cholcov(C, eps=eps)
        self.whiten_mean = m
        self.whiten_T = T
        self.feature_dim = T.shape[1]
        if return_whitened_data:
            return (Chi - m).dot(T.T)
    
    def whitened_feature_mapping(self, X):
        return (self.feature_mapping(X)-self.whiten_mean).dot(self.whiten_T.T)
    
    def calculate_singular_function(self, X):
        '''
        Left singular function from (N * input_dim) array to (N * ...) array
        '''
        return self.whitened_feature_mapping(X).dot(self.U)
    
    def train(self, X, Y, eps=0.):
        N = X.shape[0]
        self.X_train = X.copy()
        self.Y_train = Y.copy()
        
        Chi_0 = self.whiten(X, eps=eps, return_whitened_data=True)
        Chi_1 = self.whitened_feature_mapping(Y)
        G = self.gramian_function(Y)
        self.U, self.s = truncated_svd_psd(Chi_0.T.dot(G).dot(Chi_0) / (N * N))
        self.s = np.sqrt(self.s)
        self.F = Chi_0.dot(self.U)
        self.K = np.zeros([self.s.shape[0] + 1, self.s.shape[0] + 1])
        self.K[0, 0] = 1
        self.K[1:, 0] = self.U.T.dot(Chi_1.mean(0))
        self.K[1:, 1:] = self.U.T.dot(Chi_1.T).dot(Chi_0).dot(self.U) / N
        self.K = self.K.T
    
    def prediction(self, x0, m, T, obs_mapping=None, nontrivial_mode_num=np.inf, return_std=False):
        """
        Predict the mean value of the trajectory of the observable from time 1 to time T with x(0)=x0
        :x0: start point
        :T: trajectory length
        :obs_mapping: The observable function from (N * input_dim) array to (N * ...) array. If None, it is identity mapping
        :nontrivial_mode_num: Number of non-trivial singular functions
        :return_std: If return estimated std
        :return: Trajectory of the observable, or (traj, traj_std)
        """
        N = self.Y_train.shape[0]
        m = min(m,self.s.shape[0])
        if obs_mapping is None:
            O = self.Y_train
        else:
            O = obs_mapping(self.Y_train)
        if return_std:
            O = np.hstack([O, O ** 2])

        G = np.empty([m + 1, O.shape[1]])
        G[0] = O.mean(0)
        G[1:, :] = self.F[:,:m].T.dot(O) / N
        
        K = self.K[:m+1, :m+1]
        
        f0 = np.empty(m+1)
        f0[0] = 1
        f0[1:] = self.calculate_singular_function(x0.reshape(1, -1))[0, :m]

        traj = np.empty([T, O.shape[1]])
        for t in range(T):
            traj[t] = (G.T.dot(np.linalg.matrix_power(K.T, t))).dot(f0)
        if return_std:
            traj_std = np.sqrt(np.maximum(traj[:, (int)(traj.shape[1]/2):] - traj[:, :(int)(traj.shape[1]/2)] ** 2, 0))
            return traj[:, :(int)(traj.shape[1]/2)], traj_std
        return traj

def test2():
    #df1 = pd.read_excel(r'./save_data/excelfile/rain0_set0_inflow_data.xlsx')
    
    df_s0 = pd.read_excel(r'./save_data/excelfile/rain0_set0_state_data.xlsx')
    df_a0= pd.read_excel(r'./save_data/excelfile/rain0_set0_action_data.xlsx')
    dt_s0=df_s0.values[:,1:]
    dt_a0=df_a0.values[:,1:]
    data=np.concatenate((dt_s0,dt_a0),axis=1)
    
    for it in range(3):
        for jt in range(3):   
            df_s0 = pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_state_data.xlsx')
            df_a0= pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_action_data.xlsx')
            dt_s0=df_s0.values[:,1:]
            dt_a0=df_a0.values[:,1:]
            tem=np.concatenate((dt_s0,dt_a0),axis=1)
            data=np.concatenate((data,tem),axis=0)
    
    data=np.array(data)
    print(data.shape)
    
    X_train, Y_train=data[:359,:],data[1:360,:]
    X_test, Y_test=data[:359,:],data[1:360,:]

    feature_dim = 1000
    feature_W = np.random.rand(X_train.shape[1], feature_dim) * 2 -1
    feature_b = np.random.rand(feature_dim)
    def feature_mapping(X):
        return np.exp(-(X.dot(feature_W) + feature_b) ** 2 / 2)
    
    model_mmd_dmd = linear_mmd_dmd(input_dim=X_train.shape[1], feature_mapping=feature_mapping)
    model_mmd_dmd.train(X_train, Y_train)
    
    model_kernel_edmd = kernel_edmd(input_dim=X_train.shape[1])
    model_kernel_edmd.train(X_train, Y_train)
    
    Y_out1=model_mmd_dmd.prediction(X_test[0,:],Y_test.shape[0])
    Y_out2,_=model_kernel_edmd.prediction(X_test[0,:],Y_test.shape[0])
    
    print(np.sum(np.sum(Y_out1-Y_test))/(data.shape[0]*data.shape[1]))
    print(np.sum(np.sum(Y_out2-Y_test))/(data.shape[0]*data.shape[1]))
    
    df_out1=pd.DataFrame(Y_out1)
    df_out2=pd.DataFrame(Y_out2)
    df_sout=pd.DataFrame(Y_test)
    
    writer = pd.ExcelWriter('./results/result1.xlsx')
    df_out1.to_excel(writer,sheet_name='out_KVAD')
    df_out2.to_excel(writer,sheet_name='out_KEDMD')
    df_sout.to_excel(writer,sheet_name='sout')
    writer.save()
    
    print(Y_out1.shape)
    j=0
    for i in range(5):
        plt.figure()
        plt.plot(Y_out2[j:j+18,i],label='KEDMD')
        plt.plot(Y_out1[j:j+18,i],label='KVAD')
        plt.plot(Y_test[j:j+18,i],label='truth')    
        plt.legend()




if __name__=='__main__':
    test2()
    