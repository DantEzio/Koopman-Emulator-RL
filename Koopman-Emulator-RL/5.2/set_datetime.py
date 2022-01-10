# -*- coding: utf-8 -*-

def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()


def set_date(sdate,edate,stime,etime,infile):
    temfile=infile+'tem_date.inp'
    output = open(temfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
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
    infile='ot.inp'
    temfile='tem.inp'
    sdate=edate='08/28/2015'
    stime='07:00:00'
    etime='13:00:00'
    #copy_result(infile,'arg-original.inp')
    set_date(sdate,edate,stime,etime,infile)