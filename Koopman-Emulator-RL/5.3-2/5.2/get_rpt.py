# -*- coding: utf-8 -*-

def handle_line(line, flag, title):
    if line.find(title) >= 0:
        flag = True
    elif flag and line == "":
        flag = False
    return flag

def get_rpt(filename):
    with open(filename, 'rt') as data:
        total_in=0
        flooding=0
        store=0
        outflow=0
        upflow=0
        downflow=0
        pumps_flag = outfall_flag= False
        k_out=0
        k_pump=0
        
        for line in data:
            line = line.rstrip('\n')
            #pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            if not pumps_flag:
                pumps_flag = handle_line(line, pumps_flag, 'Flow Routing Continuity')
            if not outfall_flag:
                outfall_flag=handle_line(line, outfall_flag, 'Outfall Loading Summary')
            
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[]:
                if line.find('External Outflow')>=0 or\
                   line.find('Exfiltration Loss')>=0 or \
                   line.find('Mass Reacted')>=0:
                    outflow+=float(node[4])

                elif line.find('Flooding Loss')>=0:
                    flooding=float(node[4])
                elif line.find('Final Stored Volume')>=0:
                    store=float(node[5])
                elif line.find('Dry Weather Inflow')>=0 or \
                     line.find('Wet Weather Inflow')>=0 :
                    total_in+=float(node[5])
                    
                elif line.find('Groundwater Inflow')>=0 or line.find('RDII Inflow')>=0 or line.find('External Inflow')>=0:
                    total_in+float(node[4])
                elif line.find('Quality Routing Continuity')>=0:
                    pumps_flag=False
                    
                if line.find('******************')>=0:
                    k_pump+=1
                if k_pump>2:
                    pumps_flag=False
                    
            if outfall_flag and node!=[]:
                if node[0]=='4' or node[0]=='5' or node[0]=='3':
                    upflow+=float(node[4])
                    flooding+=float(node[4])
                elif line.find('WSC')>=0:
                    downflow+=float(node[4])
                    
                if line.find('---------------')>=0:
                    k_out+=1
                if k_out>3:
                    outfall_flag=False

    return total_in,flooding,store,outflow,upflow,downflow
        

if __name__ == '__main__':

    filename='./sim/staf.rpt'
    #arg_output_path0 = './sim/arg-original.rpt'
    print(get_rpt(filename))

