# -*- coding: utf-8 -*-
import os
import struct
import numpy as np

def read_out(filename):
    '''
    t:which time point you want to return 
    '''
    RECORDSIZE=4
    version=0
    NflowUnits=0
    Nsubcatch=0
    Nnodes=0
    Nlinks=0
    Npolluts=0
    
    
    magic1=0
    magic2=0
    magic3=0
    err=0
    startPos=0
    nPeriods=0
    errCode=0
    IDpos=0
    propertyPos=0
    
    pollutantUnit=''
    
    sub_name=[]
    node_name=[]
    link_name=[]
    poll_name=[]
    reportInterval=[]
    subcatchResultValueList=[]
    nodeResultValueList=[]
    linkResultValueList=[]
    systemResultValueList=[]
    data={}
    
    #各个要素id
    
    br=open(filename,'rb')
    #判断文件是否正常打开
    if(br==None or os.path.getsize(filename)):
        err=1
    
    #读取末尾的位置属性
    br.seek(os.path.getsize(filename)-RECORDSIZE*6)
    IDpos=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    propertyPos=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    startPos=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    nPeriods=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    errCode=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    magic2=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    
    #print(IDpos,propertyPos,startPos,nPeriods,errCode,magic2)
    
    #读取开头的magic变量
    br.seek(0)
    magic1=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    
    if(magic1!=magic2 or errCode!=0 or nPeriods==0):
        err=1
    else:
        err=0
        
    if(err==1):
        br.close()
        return sub_data,node_data,link_data
    else:
            
        #读取版本号，单位，汇水区个数，节点个数，管道个数，污染物个数
        version=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        NflowUnits=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        Nsubcatch=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        Nnodes=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        Nlinks=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        Npolluts=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        
        #print(version,NflowUnits,Nsubcatch,Nnodes,Nlinks,Npolluts)
        
        #读取各个id列表
        br.seek(IDpos)
        
        
        for i in range(Nsubcatch):
            numSubIdNames=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
            subcatchByte=br.read(numSubIdNames)
            sub_name.append(subcatchByte.decode(encoding = "utf-8"))
        
        for i in range(Nnodes):
            numNodeIdNames=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
            nodeByte=br.read(numNodeIdNames)
            node_name.append(nodeByte.decode(encoding = "utf-8"))
        
        for i in range(Nlinks):
            numlinkIdNames=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
            linkByte=br.read(numlinkIdNames)
            link_name.append(linkByte.decode(encoding = "utf-8"))
        
        for i in range(Npolluts):
            numpollutsIdNames=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
            pollutsByte=br.read(numpollutsIdNames)
            poll_name.append(pollutsByte.decode(encoding = "utf-8"))
        
        #读取污染物单位
        unit=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        #print(unit)
        if unit==0:
            pollutantUnit='mg/L'
        if unit==1:
            pollutantUnit='ug/L'
        if unit==2:
            pollutantUnit='counts/L'
         
        #读取各个属性个数
        br.seek(propertyPos)
        numSubcatProperty=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        offsetTemp1=numSubcatProperty*Nsubcatch
        br.seek((offsetTemp1+1)*4,1)
        numNodeProperty=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        offsetTemp2=numNodeProperty*Nnodes
        br.seek((offsetTemp2+3)*4,1)
        numLinkProperty=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        
        #print(numSubcatProperty,numNodeProperty,numLinkProperty)
        
        #读取各个属性
        subcatchProNameList=[]
        subcatchProValueList=[]
        nodeProNameList=[]
        nodeProValueList=[]
        linkProNameList=[]
        linkProValueList=[]
        
        br.seek(propertyPos+4)
        subcatchProNameList.append(int.from_bytes(br.read(RECORDSIZE),byteorder='little'))
        for i in range(Nsubcatch):
            subcatchProValueList.append(struct.unpack('f',br.read(RECORDSIZE)))
            #txtSubcatchPro.Text+=subcatchProValueList[i].To
        
        br.read(RECORDSIZE)
        for k in range(3):
            nodeProNameList.append(int.from_bytes(br.read(RECORDSIZE),byteorder='little'))
        for i in range(Nnodes*3):
            nodeProValueList.append(struct.unpack('f',br.read(RECORDSIZE)))
            
        #print(nodeProValueList)
            
        br.read(RECORDSIZE)
        for k in range(5):
            linkProNameList.append(int.from_bytes(br.read(RECORDSIZE),byteorder='little'))
        for i in range(Nlinks*5):
            linkProValueList.append(struct.unpack('f',br.read(RECORDSIZE)))
        
        #print(nodeProValueList)
        
        '''
        computing result
        '''
        #读取计算结果
        br.seek(startPos)   
        for i in range(nPeriods):
            dt=struct.unpack('f',br.read(RECORDSIZE))
            reportInterval.append(dt)
            br.read(RECORDSIZE)
            tem=[]
            for su in range(Nsubcatch):
                tem1=[]
                for su1 in range(8+Npolluts):
                    tem1.append(struct.unpack('f',br.read(RECORDSIZE)))
                tem.append(tem1)
            subcatchResultValueList.append(tem)
        
            tem=[]
            for no in range(Nnodes):
                tem1=[]
                for no1 in range(6+Npolluts):
                    tem1.append(struct.unpack('f',br.read(RECORDSIZE)))
                tem.append(tem1)
            nodeResultValueList.append(tem)
            
            tem=[]
            for li in range(Nlinks):
                tem1=[]
                for li1 in range(5+Npolluts):
                    tem1.append(struct.unpack('f',br.read(RECORDSIZE)))
                tem.append(tem1)
            linkResultValueList.append(tem)
        
            tem=[]
            for sy in range(15):
                tem.append(struct.unpack('f',br.read(RECORDSIZE)))
            systemResultValueList.append(tem)
        
        br.close()   
        ''' 
        k=0
        for item in sub_name:
            sub_data[item]=subcatchResultValueList[-1][k]
            k+=1
        k=0
        for item in link_name:
            data[item]=linkResultValueList[t][k][0]
            k+=1
        k=0
        for item in node_name:
            data[item]=[nodeResultValueList[t][k][6],nodeResultValueList[t][k][0]]
            k+=1
        '''
        
        return nodeResultValueList,node_name

'''
基础解析方法
'''
def get_depth(o_data,node_name,t):
    '''
    t时刻所有节点的水位
    '''
    k=0
    depth={}
    for item in node_name:
        depth[item]=o_data[t][k][0][0]
        k+=1
    return depth

def get_head(o_data,node_name,t):
    '''
    t时刻所有节点的Head
    '''
    k=0
    head={}
    for item in node_name:
        head[item]=o_data[t][k][1][0]
        k+=1
    return head


def get_volume(o_data,node_name,t):
    '''
    t时刻所有节点的volume
    '''
    k=0
    volume={}
    for item in node_name:
        volume[item]=o_data[t][k][0][0]
        k+=1
    return volume


def get_lateral_inflow(o_data,node_name,t):
    '''
    t时刻所有节点流入量
    '''
    k=0
    inflow={}
    for item in node_name:
        inflow[item]=o_data[t][k][3][0]
        k+=1
    return inflow

def get_total_inflow(o_data,node_name,t):
    '''
    到t时刻所有节点的总流量
    '''
    k=0
    t_inflow={}
    for item in node_name:
        t_inflow[item]=o_data[t][k][4][0]
        k+=1
    return t_inflow


def get_flood(o_data,node_name,t):
    '''
    t时刻所有节点的flooding
    '''
    k=0
    flood={}
    for item in node_name:
        flood[item]=o_data[t][k][5][0]
        k+=1
    return flood

def get_cod(o_data,node_name,t):
    '''
    t时刻所有节点COD
    '''
    k=0
    COD={}
    for item in node_name:
        COD[item]=o_data[t][k][6][0]
        k+=1
    return COD




'''
功能解析方法，调用上述方法进行解析
'''
def depth(filename,pool_list,t):
    '''
    t时刻out文件中的前池水位
    '''
    data,name=read_out(filename)
    depth=get_depth(data,name,t-1)
    pool_d={}
    for pool in pool_list:
        pool_d[pool]=depth[pool]

    return pool_d

if __name__=='__main__':
    filename='./sim/ot.out'
    data,name=read_out(filename)
    
    #将data写入txt
    T=len(data)
    N=len(data[0])
    M=len(data[0][0])
    print(T,N,M)
    t=10
    '''
    text=''
    n=N-1
    for t in range(T):
        for m in range(M):
            text+=str(data[t][n][m][0])+'\t\t'
        text+='\n'
    
    output = open('data.txt', 'wt')
    output.write(text)
    output.close()
    '''
    
    print(name)
    print(".................................")
    print(get_cod(data,name,t))
    print(".................................")
    print(get_depth(data,name,t))
    print(".................................")
    print(get_flood(data,name,t))
    print(".................................")
    print(get_total_inflow(data,name,t))
    print(".................................")
    print(get_lateral_inflow(data,name,t))
    print(".................................")
    print(get_volume(data,name,t))
    print(".................................")
    print(get_head(data,name,t))
    
    pool_list=['CC-storage']

    print(depth(filename,pool_list,t))
    
    