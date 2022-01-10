# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def change_rain(rain,time,orifile,outfile):
    #orifile中仅有管网数据，没有泵的运行数据
    #outfile是orifile加上降雨数据之后得到的
    
    def handle_line(line, title):
        flag=False
        if line.find(title) >= 0:
            flag = True
        return flag
    
    output = open(outfile, 'wt')
    with open(orifile, 'rt') as data:
        time_flag=  False
        k=0
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            
            if not time_flag:
                time_flag = handle_line(line, '[TIMESERIES]')
                
            if time_flag:
                if k==0:
                    k+=1
                    output.write(line + '\n')
                else:
                    if line.find(';')>=0 and k<2:
                        k+=1
                        output.write(line + '\n')
                    else:
                        tem=line + '\n'
                        it=0
                        for item in time:
                            tem+='10y'+' '*25+'08/28/2015 '+item+' '*6+str(rain[it])+'\n'
                            it+=1
                        tem+=';'+'\n'
                        output.write(tem)
                        time_flag=False
            else:
                output.write(line + '\n')
    output.close()
    


def set_pump(action,t,pump_list,orifile,outfile):
    #orifile中仅有管网数据，没有泵的运行数据
    #outfile是orifile加上泵的运行数据之后得到的
    
    def handle_line(line, title):
        flag=False
        if line.find(title) >= 0:
            flag = True
        return flag
    
    output = open(outfile, 'wt')
    with open(orifile, 'rt') as data:
        control_flag=time_flag=  False
        k,kc=0,0
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            
            if not time_flag:
                time_flag = handle_line(line, '[TIMESERIES]')
            if not control_flag:
                control_flag = handle_line(line, '[CONTROLS]')
                
            if time_flag:
                if k==0:
                    k+=1
                    output.write(line + '\n')
                else:
                    if line.find(';')>=0 and k<2:
                        k+=1
                        output.write(line + '\n')
                    else:
                        tem=line + '\n'
                        for pump_ind in range(len(pump_list)):
                            action_ind=0
                            for item in t:
                                tem+='pump_'+str(pump_ind)+' '*32+item+' '*6+str(action[action_ind][pump_ind])+'\n'
                                action_ind+=1
                            tem+=';'+'\n'
                        output.write(tem)
                        time_flag=False
        
            
            elif control_flag:
                if kc==0:
                    kc+=1
                    output.write(line + '\n')
                else:
                    for pik in range(len(pump_list)):
                        line='RULE R'+str(pik)+'\n'\
                            +'IF SIMULATION TIME > 0'+'\n'\
                            +'THEN PUMP '+pump_list[pik]+' SETTING = TIMESERIES pump_'+str(pik)+'\n'
                        
                        output.write(line + '\n')
                    control_flag=False
            else:
                output.write(line + '\n')
    output.close()
    
def set_date(sdate,edate,stime,etime,infile):
    
    def copy_result(outfile,infile):
        output = open(outfile, 'wt')
        with open(infile, 'rt') as data:
            for line in data:
                output.write(line)
        output.close()
    
    temfile=infile+'tem_date.inp'
    output = open(temfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            node=line.split()
            if node!=[]:
                if(node[0]=='START_DATE'):
                    tem=node[0]+' '*11+sdate
                    line=tem
                elif(node[0]=='END_DATE'):
                    tem=node[0]+' '*13+edate
                    line=tem
                elif(node[0]=='REPORT_START_DATE'):
                    tem=node[0]+' '*4+sdate
                    line=tem
                elif(node[0]=='REPORT_START_TIME'):
                    tem=node[0]+' '*4+stime
                    line=tem
                elif(node[0]=='START_TIME'):
                    tem=node[0]+' '*11+stime
                    line=tem
                elif(node[0]=='END_TIME'):
                    tem=node[0]+' '*13+etime
                    line=tem
                else:
                    pass
            else:
                pass
            output.write(line + '\n')
    output.close()
    copy_result(infile,temfile)


if __name__ == '__main__':
    infile='ori_model.inp'
    outfile='out_model.inp'
    date_time=['07:00','08:30','09:00','09:30',\
           '09:40','10:00','10:20',\
           '10:40','11:00','12:00','13:00']
    rain=[10.0 for _ in range(240)]

    change_rain(rain,date_time,infile,outfile)
    
    action=[[1,1,1,0,0,0,0,0,1,1,0]]*11
    pump_list=['CC-Pump-1','CC-Pump-2','JK-Pump-1','JK-Pump-2','XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']
    arg_input_path0 = 'ori_model.inp'
    arg_input_path1 = 'out_model.inp'
    #copy_result(arg_input_path0,'arg-original.inp')
    set_pump(action,date_time,pump_list,arg_input_path0,arg_input_path1)