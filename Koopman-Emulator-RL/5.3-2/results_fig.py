# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#每个df包含所有降雨结果
#获取所有测试数据
def collect_data(filename):
    data={}
    for i in range(4):
        data[i]=[]
        
    for testid in ['test']:
        for rid in range(20):
            tem=pd.read_csv('./5.2/'+filename+'_test_result/'+testid+' '+str(rid)+' '+filename+'flooding_vs_t.csv').values
            for i in range(4):
                if testid=='test':
                    k=i
                else:
                    k=i+4
                data[k].append(tem[1:,i].tolist())
    return data

dptest='dqn'
dfDL=collect_data(dptest+'_DLEDMD')
dfL=collect_data(dptest+'_Linear')
dfM=collect_data(dptest+'_MLP')
dfS=collect_data(dptest+'_SWMM')


dfhc=pd.read_csv('./5.2/HC/test hc_DLEDMD_flooding_vs_t.csv').values[1:,:]

#datas=[dfdqn,dfddqn,dfppo1,dfppo2,dfa2c,dfvt]

font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 25,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 25,}

font3 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 25,}


def draw(data,dfhc):
    a=np.max(data,axis=0)
    b=np.min(data,axis=0)
    for i in range(1,a.shape[0]-1):
        if a[i]<a[i-1]:
            a[i]=a[i-1]
        if b[i]<b[i-1]:
            b[i]=b[i-1]
    
    xf = [i for i in range(a.shape[0])]

    plt.fill_between(xf,b[xf],a[xf],color='b',alpha=0.75)
    plt.xticks([0,48],['08:00','16:00'])
    #plt.plot(dfop,'k',label='Optimization model',alpha=0.5)
    plt.plot(dfhc,'k:',label='Water level system',alpha=0.75)
    #plt.legend(prop=font1)

fig = plt.figure(figsize=(20,24))
line=0
im=1
while im < 16:
    
    if line ==0:
        data=dfDL
        ytitle='DLEDMD'
    elif line==1:
        data=dfL
        ytitle='Linear'
    elif line==2:
        data=dfM
        ytitle='MLP'
    else:
        data=dfS
        ytitle='SWMM'
        
    plts=fig.add_subplot(5,4,im)
    
    draw(data[0],dfhc[:,0])
    if im<5:
        plt.title('Rain'+str(im),fontdict=font3)
    if im>12:
        plt.xlabel('Time (minutes)',font1)
    if np.mod(im-1,4)==0:
        plt.ylabel(ytitle,font1)
    plt.xticks([0,47],['0','480'],fontsize=15)
    plt.yticks(fontsize=15)
    im+=1
    
    plts=fig.add_subplot(5,4,im)
    draw(data[1],dfhc[:,1])
    if im<5:
        plt.title('Rain'+str(im),fontdict=font3)
    if im>12:
        plt.xlabel('Time (minutes)',font1)
    if np.mod(im-1,4)==0:
        plt.ylabel(ytitle,font1)
    plt.xticks([0,47],['0','480'],fontsize=15)
    plt.yticks(fontsize=15)
    im+=1
    
    plts=fig.add_subplot(5,4,im)
    draw(data[2],dfhc[:,2])
    if im<5:
        plt.title('Rain'+str(im),fontdict=font3)
    if im>12:
        plt.xlabel('Time (minutes)',font1)
    if np.mod(im-1,4)==0:
        plt.ylabel(ytitle,font1)
    plt.xticks([0,47],['0','480'],fontsize=15)
    plt.yticks(fontsize=15)
    im+=1
    
    plts=fig.add_subplot(5,4,im)
    draw(data[3],dfhc[:,3])
    if im<5:
        plt.title('Rain'+str(im),fontdict=font3)
    if im>12:
        plt.xlabel('Time (minutes)',font1)
    if np.mod(im-1,4)==0:
        plt.ylabel(ytitle,font1)
    plt.xticks([0,47],['0','480'],fontsize=15)
    plt.yticks(fontsize=15)
    im+=1
    
    line=line+1
  
plt.text(-204, 50, 'The sum of CSO and flooding volume (10$^{3}$ m$^{3}$)',rotation=90,fontdict=font2)

#plt.text(-215, 200, r'DQN', fontdict=font1)
#plt.text(-215, 175, r'DDQN', fontdict=font1)
#plt.text(-215, 150, r'PPO1', fontdict=font1)
#plt.text(-215, 100, r'PPO2', fontdict=font1)
#plt.text(-215, 75, r'A2C', fontdict=font1)
#plt.text(-215, 25, r'Voting', fontdict=font1)

fig.savefig(dptest+'-uncertainty.png', bbox_inches='tight', dpi=500)
