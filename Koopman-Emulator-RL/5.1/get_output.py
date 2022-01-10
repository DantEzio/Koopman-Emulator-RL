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
        
        return nodeResultValueList,node_name,linkResultValueList,link_name,subcatchResultValueList,sub_name

'''
基础解析方法
'''
import pandas as pd

def get_node_data(node_data,node_name,filename):
    '''
    所有节点的水位、流量、COD
    '''
    #针对每个节点构建字典
    data={}
    for iname in range(len(node_name)):
        data[node_name[iname]+'depth']=[]
        data[node_name[iname]+'total_inflow']=[]
        data[node_name[iname]+'COD']=[]

    
    for t in range(len(node_data)):
        for iname in range(len(node_name)):
            depth=node_data[t][iname][0][0]#节点Depth数据
            total_inflow=node_data[t][iname][4][0]#节点流量数据
            COD=node_data[t][iname][6][0]#节点COD数据
            data[node_name[iname]+'depth'].append(depth)
            data[node_name[iname]+'total_inflow'].append(total_inflow)
            data[node_name[iname]+'COD'].append(COD)
    
    #将上述字典合并为矩阵，保存为excel
    datapd=pd.DataFrame(data)
    writer = pd.ExcelWriter(filename+'_node_data.xlsx')
    datapd.to_excel(writer)
    writer.save()
    
def get_link_data(link_data,link_name,filename):
    '''
    所有节点的水位、流量、COD
    '''
    #针对每个节点构建字典
    data={}
    for iname in range(len(link_name)):
        data[link_name[iname]+'depth']=[]
        data[link_name[iname]+'flow']=[]
        data[link_name[iname]+'COD']=[]

    
    for t in range(len(link_data)):
        for iname in range(len(link_name)):
            depth=link_data[t][iname][1][0]#节点Depth数据
            inflow=link_data[t][iname][0][0]#节点流量数据
            COD=link_data[t][iname][5][0]#节点COD数据
            data[link_name[iname]+'depth'].append(depth)
            data[link_name[iname]+'flow'].append(inflow)
            data[link_name[iname]+'COD'].append(COD)
    
    #将上述字典合并为矩阵，保存为excel
    datapd=pd.DataFrame(data)
    writer = pd.ExcelWriter(filename+'_link_data.xlsx')
    datapd.to_excel(writer)
    writer.save()
    
def get_sub_data(sub_data,sub_name,filename):
    '''
    所有节点的水位、流量、COD
    '''
    #针对每个节点构建字典
    data={}
    for iname in range(len(sub_name)):
        data[sub_name[iname]+'depth']=[]
        data[sub_name[iname]+'flow']=[]
        data[sub_name[iname]+'COD']=[]

    
    for t in range(len(sub_data)):
        for iname in range(len(sub_name)):
            depth=sub_data[t][iname][1][0]#节点Depth数据
            inflow=sub_data[t][iname][0][0]#节点流量数据
            COD=sub_data[t][iname][5][0]#节点COD数据
            data[sub_name[iname]+'depth'].append(depth)
            data[sub_name[iname]+'flow'].append(inflow)
            data[sub_name[iname]+'COD'].append(COD)
    
    #将上述字典合并为矩阵，保存为excel
    datapd=pd.DataFrame(data)
    writer = pd.ExcelWriter(filename+'_sub_data.xlsx')
    datapd.to_excel(writer)
    writer.save()
    
def get_lateralin_data(node_data,node_name,filename):
    '''
    所有节点的水位、流量、COD
    '''
    #针对每个节点构建字典
    data={}
    for iname in range(len(node_name)):
        data[node_name[iname]+'lateral_inflow']=[]
        data[node_name[iname]+'lateral_inCOD']=[]
    
    for t in range(len(node_data)):
        for iname in range(len(node_name)):
            inflow=node_data[t][iname][3][0]#节点入流流量数据
            COD=node_data[t][iname][5][0]#节点COD数据
            data[node_name[iname]+'lateral_inflow'].append(inflow)
            data[node_name[iname]+'lateral_inCOD'].append(COD)
    
    #将上述字典合并为矩阵，保存为excel
    datapd=pd.DataFrame(data)
    writer = pd.ExcelWriter(filename+'_inflow_data.xlsx')
    datapd.to_excel(writer)
    writer.save()
    

if __name__=='__main__':
    filename='./ori_model.out'
    node_data,node_name,link_data,link_name,sub_data,sub_name=read_out(filename)
    
    #将data写入txt
    T=len(node_data)
    N=len(node_data[0])
    M=len(node_data[0][0])
    print(T,N,M)
    
    print(len(node_name))
    #get_node_data(node_data,node_name,filename)
    #get_link_data(link_data,link_name,filename)
    #get_lateralin_data(node_data,node_name,filename)
    get_sub_data(sub_data,sub_name,filename)
    