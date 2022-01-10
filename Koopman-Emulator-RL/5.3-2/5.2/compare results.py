# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


#dqn比较同一sampling size不同方法的效果
rl='dqn'
step=2000
testid='test'
d_D=pd.read_csv('./'+rl+'_DLEDMD_test_result '+str(step)+'/'+testid+' '+rl+'_DLEDMDflooding_vs_t.csv').values
d_L=pd.read_csv('./'+rl+'_Linear_test_result '+str(step)+'/'+testid+' '+rl+'_Linearflooding_vs_t.csv').values
d_G=pd.read_csv('./'+rl+'_GP_test_result '+str(step)+'/'+testid+' '+rl+'_GPflooding_vs_t.csv').values
d_M=pd.read_csv('./'+rl+'_MLP_test_result '+str(step)+'/'+testid+' '+rl+'_MLPflooding_vs_t.csv').values
d_S=pd.read_csv('./'+rl+'_SWMM_test_result 100/'+testid+' '+rl+'_SWMMflooding_vs_t.csv').values

d_H=pd.read_csv('./dqn_DLEDMD_test_result '+str(step)+'/'+testid+' dqn_DLEDMDhc_flooding_vs_t.csv').values


data=pd.DataFrame(np.vstack((d_D[-1],d_M[-1],d_L[-1],d_G[-1],d_S[-1],d_H[-1])).T,columns=['DLEDMD','MLP','Linear','GP','SWMM','HC'])
print(rl)
print(step)
print(testid)
print(data)
