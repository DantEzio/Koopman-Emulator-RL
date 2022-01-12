# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt


testid='dqn'#'dqn'

index=str(1000)

df_DLEDMD1=pd.read_csv('./final-results/'+testid+'_DLEDMD_test_result '+index+'/test '+testid+'_DLEDMDflooding_vs_t.csv')
df_DLEDMD2=pd.read_csv('./final-results/'+testid+'_DLEDMD_test_result '+index+'/real '+testid+'_DLEDMDflooding_vs_t.csv')

df_Linear1=pd.read_csv('./final-results/'+testid+'_Linear_test_result '+index+'/test '+testid+'_Linearflooding_vs_t.csv')
df_Linear2=pd.read_csv('./final-results/'+testid+'_Linear_test_result '+index+'/real '+testid+'_Linearflooding_vs_t.csv')

df_MLP1=pd.read_csv('./final-results/'+testid+'_MLP_test_result '+index+'/test '+testid+'_MLPflooding_vs_t.csv')
df_MLP2=pd.read_csv('./final-results/'+testid+'_MLP_test_result '+index+'/real '+testid+'_MLPflooding_vs_t.csv')

df_SWMM1=pd.read_csv('./final-results/'+testid+'_SWMM_test_result 2000/test '+testid+'_SWMMflooding_vs_t.csv')
df_SWMM2=pd.read_csv('./final-results/'+testid+'_SWMM_test_result 2000/real '+testid+'_SWMMflooding_vs_t.csv')

dfhc1=pd.read_csv('./final-results/HC/test hc_DLEDMD_flooding_vs_t.csv')
dfhc2=pd.read_csv('./final-results/HC/real hc_DLEDMD_flooding_vs_t.csv')

font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 15,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 20,}

time_point=['0','480']
x=[0,dfhc1.shape[0]]

fig = plt.figure(figsize=(15,15))
for im in range(1,9):
    plts=fig.add_subplot(4,2,im)
    if im<=4:
        if im ==1:
            plt.plot(df_DLEDMD1.iloc[:,im-1],'r:',label='DLEDMD-'+testid)
            plt.plot(df_Linear1.iloc[:,im-1],'b:',label='Linear-'+testid)
            plt.plot(df_MLP1.iloc[:,im-1],'y:',label='MLP-'+testid)
            plt.plot(df_SWMM1.iloc[:,im-1],'k:',label='SWMM-'+testid)
            plt.plot(dfhc1.iloc[:,im-1],'k.-',label='Water level system')
        else:
            plt.plot(df_DLEDMD1.iloc[:,im-1],'r:')
            plt.plot(df_Linear1.iloc[:,im-1],'b:')
            plt.plot(df_MLP1.iloc[:,im-1],'y:')
            plt.plot(df_SWMM1.iloc[:,im-1],'k:')
            plt.plot(dfhc1.iloc[:,im-1],'k.-')
    else:
        plt.plot(df_DLEDMD2.iloc[:,im-1-4],'r:')
        plt.plot(df_Linear2.iloc[:,im-1-4],'b:')
        plt.plot(df_MLP2.iloc[:,im-1-4],'y:')
        plt.plot(df_SWMM2.iloc[:,im-1-4],'k:')
        plt.plot(dfhc2.iloc[:,im-1-4],'k.-')
    plt.title('Rain'+str(im),font2)
    
    if im in [7,8]:
        plt.xlabel('Time (minutes)',font2)
    # 增加刻度
    plt.xticks(x, time_point,fontsize=15)
    plt.yticks(fontsize=15)

plt.text(-73, 33, 'The sum of CSO and flooding volume (10$^{3}$ m$^{3}$)',rotation=90,fontdict=font2)
fig.legend(prop=font1,bbox_to_anchor=(0.45,0.01),loc=8,ncol= 5)
fig.savefig('5.2.1-'+index+' '+testid+'.png', bbox_inches='tight', dpi=500)
