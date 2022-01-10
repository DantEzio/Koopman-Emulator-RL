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
import tensorflow as tf
import DLEDMD

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


class env_DLEDMD:
    def __init__(self,date_time,date_t):
        
        self.data,self.data_test1,self.data1=Pre_data()
        
        self.date_t=date_t
        self.date_time=date_time
        
        X_train=self.data[:self.data.shape[0]-2,:4]
        X_train_F=self.data[:self.data.shape[0]-2,4:]
        Y_train=self.data[1:self.data.shape[0]-1,:4]
                
        #DLEDMD
        params={}
        #params['enlayers']=[[X_train.shape[1]+X_train_F.shape[1],100],[100,100],[100,100]]
        params['enlayers']=[[4+9,10],[10,50],[50,100],[100,100]]
        params['K']=[100,100]
        #params['delayers']=[[100,100],[100,100],[100,Y_train.shape[1]]]
        params['delayers']=[[100,100],[100,50],[50,10],[10,4]]
        
        params['input_layer']=4#X_train.shape[1]
        params['f_layer']=9#X_train_F.shape[1]
        params['output_layer']=4#Y_train.shape[1]
        params['lr']=0.0001
        
        
        X_train=self.data[:self.data.shape[0]-2,:4]
        X_train_F=self.data[:self.data.shape[0]-2,4:]
        Y_train=self.data[1:self.data.shape[0]-1,:4]
        
        g_DLEDMD=tf.Graph()
        with g_DLEDMD.as_default():
            self.DL=DLEDMD.DLEDMD(params,X_train,Y_train,X_train_F)
            self.ybar=self.DL.forward_net(self.DL.x,self.DL.f)
            self.loss=self.DL.Loss(self.DL.x,self.DL.f,self.DL.y)
            self.op=self.DL.opt(self.loss)
            self.sess=tf.compat.v1.Session()
            
            init_op = tf.compat.v1.global_variables_initializer()
            self.sess.run(init_op)
            saver=tf.compat.v1.train.Saver()
            
            '''
            epoch=20000
            r1=[]
            for step in np.arange(epoch):
                for i in np.arange(10):
                    self.sess.run(self.op,feed_dict={self.DL.x:X_train,self.DL.f:X_train_F,self.DL.y:Y_train})
                r1.append(self.sess.run(self.loss,feed_dict={self.DL.x:X_train,self.DL.f:X_train_F,self.DL.y:Y_train}))
                #r2.append(sess.run(model.Loss,feed_dict={model.x:X_test,model.y_bar:Y_test,model.f:X_test_F}))
                
            plt.plot(r1)
            print(np.min(r1))
            self.sess.run(self.op,feed_dict={self.DL.x:X_train,self.DL.f:X_train_F,self.DL.y:Y_train})
            saver.save(self.sess,'./emulator model/DLEDMD_model.ckpt')
            '''
            saver.restore(self.sess,'./dqn_DLEDMD_test_result/emulator model/DLEDMD_model.ckpt')
                    
        self.iten=0
        #self.Rain=rainData
               
        self.pump_list={'CC-storage':['CC-Pump-1','CC-Pump-2'],'JK-storage':['JK-Pump-1','JK-Pump-2'],'XR-storage':['XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']}
        self.limit_level={'CC-storage':[0.9,3.02,4.08],'JK-storage':[0.9,3.02,4.08],'XR-storage':[0.9,1.26,1.43,1.61,1.7]}
        self.max_depth={'CC-storage':5.6,'JK-storage':4.8,'XR-storage':7.72}
        self.pool_list=['CC-storage','JK-storage','XR-storage']
        self.action_space=[1.0 , 0.0]
    
    def reset(self,raindata):
        self.iten=0
        self.T=raindata.shape[0]
        self.Rain=raindata
        self.T=raindata.shape[0]
        
        st=np.array([0.134,0.131,0.554,0.245]).reshape((1,-1))
        r=np.array([raindata[0]])
        a=np.array([0,0,0,0,0,0,0,0])
        F=np.hstack((a,r)).reshape((1,-1))
        #S=np.hstack((st,a,raindata[0])).reshape((1,-1))
        
        Y=self.sess.run(self.ybar,feed_dict={self.DL.x:st,self.DL.f:F})

        flooding=np.abs(Y[0][0])
        total_in=np.abs(Y[0][1])
        store=np.abs(Y[0][2])
        outflow=np.abs(Y[0][3])
        rain_sum=np.sum(raindata[:self.iten])
        
        self.st=[flooding,total_in,store,outflow]#np.abs(state[0][:4])#用于模拟的
        state=np.array([total_in,flooding,store,outflow,rain_sum])
        return state,flooding
    
    def step(self,a,raindata):
        #修改statf的date，根据iten逐步向前
        #加入action
        #开始模拟，存储结果
        
        self.iten+=1
        F=np.hstack((a,raindata[self.iten])).reshape((1,-1))
        Y=self.sess.run(self.ybar,feed_dict={self.DL.x:np.array(self.st).reshape((1,-1)),self.DL.f:F})
        
        flooding=np.abs(Y[0][0])
        total_in=np.abs(Y[0][1])
        store=np.abs(Y[0][2])
        outflow=np.abs(Y[0][3])
        rain_sum=np.sum(raindata[:self.iten])
        
        self.st=[flooding,total_in,store,outflow]#np.abs(state[0][:4])#用于模拟的
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
    date_t=[]
    for i in range(len(date_time)):
        date_t.append(int(i*10))
    
    rainData1=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据
    rainData2=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')*20#读取真实降雨数据
    rainData=np.vstack((rainData1[:,:120],rainData2[:4,:]))#共8场降雨
    print(rainData.shape)
    #plt.plot(rainData.T)
    
    env=env_DLEDMD(date_time,date_t)
    st=np.array([0.134,0.131,0.554,0.245])
    a=np.array([0,0,0,0,0,0,0,0])
    env.reset(rainData[0])
    
    
    R=[]
    floodings=[]
    for t in range(rainData[0].shape[0]-2):
        a=np.array([1,1,1,1,1,1,1,1])
        state,reward_sum,done,_,flooding=env.step(a,rainData[0])
        floodings.append(flooding)
        R.append(reward_sum)
    
    plt.figure()
    plt.plot(R)
    
    plt.figure()
    plt.plot(floodings)
    