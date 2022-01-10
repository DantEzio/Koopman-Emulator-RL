# -*- coding: utf-8 -*-

def handle_line(line, title):
    flag=False
    if line.find(title) >= 0:
        flag = True
    return flag


def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()


def set_pump(action,t,pump_list,infile):
    temfile=infile+'tem_pump.inp'
    output = open(temfile, 'wt')
    with open(infile, 'rt') as data:
        control_flag=time_flag=  False
        k=0
        for line in data:
            line = line.rstrip('\n')
            
            if not time_flag:
                time_flag = handle_line(line, '[TIMESERIES]')
            if time_flag:
                
                if k==0:
                    k+=1
                else:
                    if line.find(';')>=0 and k<=2:
                        #print(line)
                        #output.write(line + '\n')
                        k+=1
                    else:
                        for pump_ind in range(len(pump_list)):
                            tem=''
                            action_ind=0
                            for item in t:
                                tem+='pump_'+str(pump_ind)+' '*11+'8/28/2015'+' '*2+item+' '*6+str(action[action_ind][pump_ind])+'\n'
                                action_ind+=1
                            tem+=';'+'\n'
                            output.write(tem)
                        time_flag=False
        
            control_flag = handle_line(line,  '[CONTROLS]')
            output.write(line + '\n')
            if control_flag:
                for pik in range(len(pump_list)):
                    line='RULE R'+str(pik)+'\n'\
                        +'IF SIMULATION TIME > 0'+'\n'\
                        +'THEN PUMP '+pump_list[pik]+' SETTING = TIMESERIES pump_'+str(pik)+'\n'
                    
                    output.write(line + '\n')
                control_flag=False
    output.close()
    copy_result(infile,temfile)
    

if __name__ == '__main__':
    
    date_time=['07:00','08:30','09:00','09:30',\
           '09:40','10:00','10:20',\
           '10:40','11:00','12:00','13:00']
    action=[[1,1,1,0,0,0,0,0]]*11
    #pump_list=['CC-Pump-1','CC-Pump-2','JK-Pump-1','JK-Pump-2','XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']
    pump_list=['CC-Pump-1','CC-Pump-2']
    arg_input_path0 = './ot.inp'
    arg_input_path1 = './tem.inp'
    #copy_result(arg_input_path0,'arg-original.inp')
    set_pump(action,date_time,pump_list,arg_input_path0)
    
    