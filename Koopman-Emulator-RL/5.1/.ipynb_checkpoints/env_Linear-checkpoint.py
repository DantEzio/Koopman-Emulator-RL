# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 09:12:48 2021

@author: chong
"""


#import gym

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#linear
from sklearn import linear_model
    

def Pre_data():
    #Prepare training data (set0)
    df_s0 = pd.read_excel(r'./save_data/excelfile/rain0_set0_state_data.xlsx')
    df_a0= pd.read_excel(r'./save_data/excelfile/rain0_set0_action_data.xlsx')
    df_r0= pd.read_excel(r'./save_data/excelfile/rain0_set0_rain_data.xlsx')
    dt_s0=df_s0.values[:,1:]
    dt_a0=df_a0.values[:,1:]
    dt_r0=df_r0.values[:,1:]
    
    data=np.concatenate((dt_s0,dt_a0,dt_r0),axis=1)
    for it in [0,1,2,3,4,5,6,7,8,9]:
        for jt in [0,2]:   
            df_s0 = pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_state_data.xlsx')
            df_a0= pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_action_data.xlsx')
            df_r0= pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_rain_data.xlsx')
            dt_s0=df_s0.values[:,1:]
            dt_a0=df_a0.values[:,1:]
            dt_r0=df_r0.values[:,1:]
            tem=np.concatenate((dt_s0,dt_a0,dt_r0),axis=1)
            data=np.concatenate((data,tem),axis=0)
            
    #Prepare test1 data
    df_s0 = pd.read_excel(r'./save_data/excelfile/rain0_set1_state_data.xlsx')
    df_a0= pd.read_excel(r'./save_data/excelfile/rain0_set1_action_data.xlsx')
    df_r0= pd.read_excel(r'./save_data/excelfile/rain0_set1_rain_data.xlsx')
    dt_s0=df_s0.values[:,1:]
    dt_a0=df_a0.values[:,1:]
    dt_r0=df_r0.values[:,1:]
    data_test1=[]
    data_test1.append(np.concatenate((dt_s0,dt_a0,dt_r0),axis=1))
    for it in [0,1,2,3,4,5,6,7,8,9]:
        for jt in [1]:   
            df_s0 = pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_state_data.xlsx')
            df_a0= pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_action_data.xlsx')
            df_r0= pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_rain_data.xlsx')
            dt_s0=df_s0.values[:,1:]
            dt_a0=df_a0.values[:,1:]
            dt_r0=df_r0.values[:,1:]
            tem=np.concatenate((dt_s0,dt_a0,dt_r0),axis=1)
            data_test1.append(tem)
    
    #Prepare training data2 (set0)
    df_s0 = pd.read_excel(r'./save_data/excelfile/rain0_set0_state_data.xlsx')
    df_a0= pd.read_excel(r'./save_data/excelfile/rain0_set0_action_data.xlsx')
    df_r0= pd.read_excel(r'./save_data/excelfile/rain0_set0_rain_data.xlsx')
    dt_s0=df_s0.values[:,1:]
    dt_a0=df_a0.values[:,1:]
    dt_r0=df_r0.values[:,1:]
    
    data1=[]
    data1.append(np.concatenate((dt_s0,dt_a0,dt_r0),axis=1))     
    for it in [0,1,2,3,4,5,6,7,8,9]:
        for jt in [0,2]:   
            df_s0 = pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_state_data.xlsx')
            df_a0= pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_action_data.xlsx')
            df_r0= pd.read_excel(r'./save_data/excelfile/rain'+str(it)+'_set'+str(jt)+'_rain_data.xlsx')
            dt_s0=df_s0.values[:,1:]
            dt_a0=df_a0.values[:,1:]
            dt_r0=df_r0.values[:,1:]
            tem=np.concatenate((dt_s0,dt_a0,dt_r0),axis=1)
            data1.append(np.concatenate((data,tem),axis=0))
    
    return data,data_test1,data1


class env_Linear:
    def __init__(self, date_time, date_t):
        
        self.data,self.data_test1,self.data1=Pre_data()
        X_train=self.data[:self.data.shape[0]-2,:]
        Y_train=self.data[1:self.data.shape[0]-1,:]
        
        self.date_time=date_time
        self.date_t=date_t
        self.T=len(self.date_t)
        
        #Linear
        self.reg = linear_model.LassoLars(alpha=0.1)
        self.reg.fit(X_train,Y_train)
        
        self.iten=0
                
        self.pump_list={'CC-storage':['CC-Pump-1','CC-Pump-2'],'JK-storage':['JK-Pump-1','JK-Pump-2'],'XR-storage':['XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']}
        self.limit_level={'CC-storage':[0.9,3.02,4.08],'JK-storage':[0.9,3.02,4.08],'XR-storage':[0.9,1.26,1.43,1.61,1.7]}
        self.max_depth={'CC-storage':5.6,'JK-storage':4.8,'XR-storage':7.72}
        self.pool_list=['CC-storage','JK-storage','XR-storage']
        self.action_space=[1.0 , 0.0]
    
    def reset(self,raindata):
        self.iten=0
        self.T=raindata.shape[0]
        self.Rain=raindata
        
        st=np.array([0.134,0.131,0.554,0.245])
        a=np.array([0,0,0,0,0,0,0,0])
        S=np.hstack((st,a,raindata[0])).reshape((1,-1))
        
        Y=self.reg.predict(S)     

        flooding=np.abs(Y[0][0])
        total_in=np.abs(Y[0][1])
        store=np.abs(Y[0][2])
        outflow=np.abs(Y[0][3])
        rain_sum=np.sum(raindata[:self.iten])
        
        self.st=[flooding,total_in,store,outflow]#np.abs(state[0][:4])#???????????????
        state=np.array([total_in,flooding,store,outflow,rain_sum])

        return state,flooding
    
    def step(self,a,raindata):
        #??????statf???date?????????iten????????????
        #??????action
        #???????????????????????????
        self.iten+=1
        S=np.hstack((self.st,a,raindata[self.iten])).reshape((1,-1))
        Y=self.reg.predict(S)     

        flooding=np.abs(Y[0][0])
        total_in=np.abs(Y[0][1])
        store=np.abs(Y[0][2])
        outflow=np.abs(Y[0][3])
        rain_sum=np.sum(raindata[:self.iten])
        
        self.st=[flooding,total_in,store,outflow]#np.abs(state[0][:4])#???????????????
        state=np.array([total_in,flooding,store,outflow,rain_sum])
        
        reward_sum=0
        if total_in==0.0:
            reward_sum+=0.0
        else:
            reward_sum+=(total_in-flooding)/(total_in)
        
        
        if self.iten==self.T-2:
            done=True
        else:
            done=False

        return state,reward_sum,done,{},flooding




if __name__=='__main__':
    '''
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
    date_t=[0,10,20,30,40,50,\
            60,70,80,90,100,110,\
            120,130,140,150,160,170,\
            180,190,200,210,220,230,240]
    '''
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50',\
               '16:00']
    
    rainData1=np.loadtxt('./sim/testRainFile.txt',delimiter=',')/6#????????????????????????
    rainData2=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')*10#????????????????????????
    rainData=np.vstack((rainData1[:,:120],rainData2[:4,:]))#???8?????????
    print(rainData.shape)
    #plt.plot(rainData.T)
    
    env=env_Linear(rainData[0])
    st=np.array([0.134,0.131,0.554,0.245])
    a=np.array([0,0,0,0,0,0,0,0])
    env.reset(rainData[0])
    
    
    R=[]
    floodings=[]
    for t in range(rainData[0].shape[0]-2):
        a=np.array([1])
        state,reward_sum,done,_,flooding=env.step(a,rainData[0])
        floodings.append(flooding)
        R.append(reward_sum)
    
    plt.figure()
    plt.plot(R)
    
    plt.figure()
    plt.plot(floodings)
    