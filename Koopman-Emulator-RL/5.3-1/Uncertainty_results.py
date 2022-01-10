# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

import get_rpt

#获取文件名
file_name=['./5.2/dqn_DLEDMD_test_result/','./5.2/ppo_DLEDMD_test_result/',
           './5.2/dqn_Linear_test_result/','./5.2/ppo_Linear_test_result/',
           './5.2/dqn_MLP_test_result/','./5.2/ppo_MLP_test_result/',
           './5.2/dqn_SWMM_test_result/','./5.2/ppo_SWMM_test_result/']


Ps=[i for i in range(1,51)]
x=[]
for p in range(50):
    x.append(float(Ps[int(20*p/50)]))


font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 10,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 18,}

colors=['r','g','b','c','m','k']
labels=[['DQN'],['DDQN'],['PPO1'],['PPO2'],['A2C'],['Voting']]
it=0

results=[]
for name in file_name:
    all_rate=[]
    
    for i in range(50):
        f=name+str(i)+'.rpt'
        total_in,CSO,_,_,_,_=get_rpt.get_rpt(f)#读取total_inflow和CSO
        all_rate.append(CSO/total_in)#计算比值RIC
         
    
    #找到5%，50%，95%的结果
    all_rate.sort()
    print(name+':')
    print('5%:',all_rate[int(50*5/100)],' ',
          '50%:',all_rate[int(50*50/100)],' ',
          '95%:',all_rate[int(50*95/100)])
    results.append([all_rate[int(50*5/100)],all_rate[int(50*50/100)],all_rate[int(50*95/100)]])

pd.DataFrame(results).to_csv('results_Table.csv')

'''
fig = plt.figure(figsize=(15,15))
for name in file_name:
    plts=fig.add_subplot(3,2,it+1)
    
    all_rate=[]
    
    for i in range(50):
        f=name+str(i)+'.rpt'
        total_in,CSO,_,_,_,_=get_rpt.get_rpt(f)#读取total_inflow和CSO
        all_rate.append(CSO/total_in)#计算比值RIC
         
    
    #画6.5的图
    plt.scatter(x,all_rate,c=colors[it])
    #plt.title(labels[it],fontdict=font2)
    
    # 增加标签
    plt.xlabel('P',fontdict=font1)
    plt.ylabel('RCI',fontdict=font1)
    
    # 增加刻度
    plt.xticks(x, x)
    
    # 设置图例
    n=labels[it]
    print(n)
    plt.legend(n,loc='best',prop=font2)
    
    k=linregress(x,all_rate)[0]
    b=linregress(x,all_rate)[1]
    print(k,b)
    x1=np.array(x)
    mean=k*x1+b
    plt.plot(x,mean,color=colors[it],lw=1.5,ls='--',zorder=2)
    
    it=it+1
fig.savefig('4.1.png', bbox_inches='tight', dpi=500) 
'''  