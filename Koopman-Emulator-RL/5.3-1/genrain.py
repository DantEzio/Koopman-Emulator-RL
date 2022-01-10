# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

Ps=[i for i in range(1,51)]
K=0.5

rains=[]
for p in range(50):
    rain=[]
    P=Ps[int(20*p/50)]
    n=random.randint(85,97)/100
    b=random.randint(16,22)
    A=random.randint(21,35)
    C=random.randint(939,1200)/1000
    for it in range(120):
        t=it
        if it<=(K*120):
            tem=(A*(1+C*np.log(P)))*((1-n)*t/K+b)/np.power(((120*K-it)+b),(1+n))
        else:
            tem=(A*(1+C*np.log(P)))*((1-n)*t/(1-K)+b)/np.power((((it-120*K))+b),(1+n))
            
        
        rain.append(tem)
    print(np.array(rain).shape)
    plt.plot(rain)
    rains.append(rain)
np.savetxt('testRainFile.txt',rains)