# -*- coding: utf-8 -*-

import DDQN
import env_SWMM
import env_MLP
import time
import numpy as np
import tensorflow as tf


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
    
    
    
    # Superparameters
    batch_size=240
    step=1000
    rain_num=4
    test_num=4
    
    envD=env_MLP.env_MLP(date_time, date_t)
    g1=tf.Graph()
    with g1.as_default():
        D_DQN = DDQN.DDQN(step=step,batch_size=batch_size,num_rain=rain_num,env=envD,t='dqn_MLP',raindata='test')
        #D_DQN.load_model()
        t1=time.time()
        D_DQN.train(True)
        t2=time.time()
        print(t2-t1)
    
    
    envS=env_SWMM.env_SWMM(date_time, date_t)
    g2=tf.Graph()
    with g2.as_default():
        D_DQN = DDQN.DDQN(step=step,batch_size=batch_size,num_rain=rain_num,env=envS,t='dqn_MLP',raindata='test')
        D_DQN.load_model()
        D_DQN.test(test_num)
        
        D_DQN = DDQN.DDQN(step=step,batch_size=batch_size,num_rain=rain_num,env=envS,t='dqn_MLP',raindata='real')
        D_DQN.load_model()
        D_DQN.test(test_num)
    