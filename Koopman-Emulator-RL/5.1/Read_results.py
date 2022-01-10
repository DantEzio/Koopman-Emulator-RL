# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel('./results/result1.xlsx',sheet_name='out').values
dfs=pd.read_excel('./results/result1.xlsx',sheet_name='sout').values

print(df.shape,dfs.shape)

print(np.max(np.square(df-dfs)))

plt.figure()
plt.plot(df[:,20])
plt.title('test')
plt.figure()
plt.title('true')
plt.plot(dfs[:,20])

