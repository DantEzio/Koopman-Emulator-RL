# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt

#constants = load(open('./constants.yml', 'r', encoding='utf-8'))


def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()


def replace_line(line, title,rain,t):

    node=line.split()
    if(node[0]=='Oneyear-2h'):
        t=t+1
        tem=node[0]+' '*8+node[1]+' '+node[2]+' '*6+str(rain)
        #print(tem)
        line=tem
        return t,line
    else:
        return t,line


def handle_line(line, flag, title,rain,t):
    if line.find(title) >= 0:
        flag = True
    elif flag and line == "":
        flag = False
    elif line.find(';') == -1 and flag:
        t,line = replace_line(line, title,rain,t)
    return t,line, flag


def change_rain(rain,infile):
    temfile=infile+'tem_rain.inp'
    output = open(temfile, 'wt')
    with open(infile, 'rt') as data:
        rain_flag =  False
        t=0
        for line in data:
            line = line.rstrip('\n')
            t,line, flag = handle_line(line, rain_flag, '[TIMESERIES]',rain[t],t)
            rain_flag = flag
            output.write(line + '\n')
    output.close()
    copy_result(infile,temfile)


def gen_rain(t,A,C,P,b,n,R,deltt):
    '''
    t是生成雨量时间序列步数上限
    delt是时间间隔，取1
    '''
    rain=[]
    for i in range(t):
        if i <int(t*R):
            rain.append(A*(1+C*math.log(P))/math.pow(((t*R-i)+b),n))
        else:
            rain.append(A*(1+C*math.log(P))/math.pow(((i-t*R)+b),n))
    
    return rain

if __name__ == '__main__':
    infile='ot.inp'
    outfile='tem.inp'
    A=10#random.randint(5,15)
    C=13#random.randint(5,20)
    P=2#random.randint(1,5)
    b=1#random.randint(1,3)
    n=0.5#random.random()
    R=0.5#random.random()
    deltt=1
    t=240
    #change_rain(A,C,P,b,n,infile,outfile)
    rain=gen_rain(t,A,C,P,b,n,R,deltt)
    plt.plot(range(240),rain)
    #copy_result(infile,'arg-original.inp')
    change_rain(rain,infile)