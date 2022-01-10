import pandas as pd
import matplotlib.pyplot as plt

dic=['DLEDMD','Linear','MLP']
testid=['dqn','ppo']
rains=['test','real']
data={}

index=str(1000)

for it in dic:
    for tid in testid:
        for rain in rains:
            data[it+tid+rain]=pd.read_csv('./final-results/'+tid+'_'+it+'_test_result '+index+'/'+rain+' '+tid+'_'+it+'flooding_vs_t.csv').values[-1]

data['HC'+'test']=pd.read_csv('./final-results/HC/test hc_DLEDMD_flooding_vs_t.csv').values[-1]
data['HC'+'real']=pd.read_csv('./final-results/HC/real hc_DLEDMD_flooding_vs_t.csv').values[-1]

results=[]
for it in dic:
    for tid in testid:
        results.append(data[it+tid+'test'].tolist()+data[it+tid+'real'].tolist())

results.append(data['HC'+'test'].tolist()+data['HC'+'real'].tolist())
pd.DataFrame(results).to_excel('results '+index+'.xlsx')
