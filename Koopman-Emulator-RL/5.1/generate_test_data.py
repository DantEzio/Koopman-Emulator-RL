import numpy as np
import rain_gen as rg
import SWMM_operator as SO
import shutil as sh
import pandas as pd
import get_output as go#从out文件读取水动力初值
import get_rpt as gr#从rpt文件读取flooding值
import matplotlib.pyplot as plt

#import datetime

from pyswmm import Simulation
#import pandas as pd

#整个代码需要两个inp文件，
#一个是orifile，仅仅包含管网数据；
#一个是simfile，是由orifile经过set_pump和change_rain之后，用于模拟的文件

class training_data_gen:
    def __init__(self,ori_filename):
        
        self.ori_filename=ori_filename+'.inp'#原始inp文件，只包含管网，子汇水区信息
        self.rain_filename=ori_filename+'_rain.inp'#包含ori_filename与降雨信息
        self.sim_filename=ori_filename+'_sim.inp'#模拟使用文件名，包含ori_filename与降雨、泵运行信息
        self.sim_filename_rpt=ori_filename+'_sim.rpt'
        self.sim_filename_out=ori_filename+'_sim.out'
        
        self.date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50',\
               '16:00','16:10','16:20','16:30','16:40','16:50',\
               '17:00','17:10','17:20','17:30','17:40','17:50',\
               '18:00']
        self.pump_list=['CC-Pump-1','CC-Pump-2',
                        'JK-Pump-1','JK-Pump-2',
                        'XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']
        
        rainData1=np.loadtxt('./source data/testRainFile.txt',delimiter=',')
        rainData2=np.loadtxt('./source data/real_rain_data.txt',delimiter=' ')*15
        self.rainData=np.vstack((rainData1[:,:120],rainData2[:4,:]))
        
        self.num_set=10
        self.num_M=self.rainData.shape[0]-1
    
    def simulation(self,filename):
        with Simulation(filename) as sim:
            #stand_reward=0
            for step in sim:
                pass    

    def copy_result(self,outfile,infile):
        output = open(outfile, 'wt')
        with open(infile, 'rt') as data:
            for line in data:
                output.write(line)
        output.close()
        
    def one_data(self,rain,pump_actions,rainlog,i):#针对一场降雨和一组泵的运行生成一组数据
        #先改降雨，泵的顺序不能变
        SO.change_rain(rain,self.date_time,self.ori_filename,self.rain_filename)
        SO.set_pump(pump_actions,self.date_time,self.pump_list,self.rain_filename,self.sim_filename) 
        self.simulation(self.sim_filename)#进行模拟
        
        #保存模拟结果
        savefile='./save_data/outfile/TEST_rain'+str(rainlog)+'_set'+str(i)
        self.copy_result(savefile+'.inp',self.sim_filename)
        self.copy_result(savefile+'.rpt',self.sim_filename_rpt)
        sh.copy(self.sim_filename_out,savefile+'.out')
        
    def one_setdata(self,rain,rainlog):#针对一场降雨，随机生成一组泵的运行数据，调用one_data生成数据
        
        for i in range(self.num_set):
            actions=[]
            
            for time in range(len(self.date_time)):
                actions.append(np.random.randint(0,2,len(self.pump_list)))
            
            filename='./save_data/outfile/TEST_rain'+str(rainlog)+'_set'+str(i)
            savename='./save_data/excelfile/TEST_rain'+str(rainlog)+'_set'+str(i)
            n_action=np.array(actions)[1:,:]
            np.savetxt(filename,n_action)
            
            datapd=pd.DataFrame(n_action)
            writer = pd.ExcelWriter(savename+'_action_data.xlsx')
            datapd.to_excel(writer)
            writer.save()
            
            self.one_data(rain,actions,rainlog,i)
            
    def Get_data(self):
        Rains=self.rainData
        print('rainfall data:', Rains.shape)
        #选其中self.num_M场降雨
        for rainlog in range(Rains.shape[0]):
            print(rainlog)
            self.one_setdata(Rains[rainlog],rainlog)
            
    def data_transfer(self):
        #将out文件转化为excel，方便后续计算。转化excel包括三个：节点时间序列，管段时间序列，节点入流数据时间序列
        #filename:包含降雨序列，泵控制序列，但是不含后缀

        for rainlog in range(self.num_M):
            for i in range(self.num_set):
                print('rain:',rainlog,' set:',i)
                filename='./save_data/outfile/TEST_rain'+str(rainlog)+'_set'+str(i)
                savename='./save_data/excelfile/TEST_rain'+str(rainlog)+'_set'+str(i)
                node_data,node_name,link_data,link_name,sub_data,sub_name=go.read_out(filename+'.out')
                
                sdate=edate='08/28/2015'
                stime=self.date_time[0]
                states=[]
                floodings=[]
                for i in range(1,len(self.date_time)):
                    etime=self.date_time[i]
                    SO.set_date(sdate,edate,stime,etime,filename+'.inp')
                    self.simulation(filename+'.inp')#进行模拟
                    total_in,flooding,store,outflow,_,_=gr.get_rpt(filename+'.rpt')
                    s=[flooding,total_in,store,outflow]
        
                    floodings.append(flooding)
                    states.append(s)
                
                floodings=np.array(floodings)
                #plt.plot(floodings)
                np.savetxt(filename,states)
                
                datapd=pd.DataFrame(states)
                writer = pd.ExcelWriter(savename+'_state_data.xlsx')
                datapd.to_excel(writer)
                writer.save()
        
        

if __name__=='__main__':
    ori_filename='ori_model'
    TDG=training_data_gen(ori_filename)
    TDG.Get_data()
    TDG.data_transfer()