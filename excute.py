import time
# from datetime import datetime
import pandas as pd
from datetime import datetime
import numpy as np
from csv import DictWriter
from csv import writer
import calendar


elements_1=['Activity','End','Start','Total(m)']
"""
   the is the counter of ANY kind of Activity in the day
    """

activity_list={'act_1':['erfan']}

class myclass(object):    
    
    global DF
    global first_time_day 
    first_time_day=datetime.strptime('2022/12/08 16:15','%Y/%m/%d %H:%M')

    def __init__(self):
        pass
    
    def get_timer(self):
        time=datetime.now()
        return time


    def start(self,assumpt,**data):
        global start_current
        main_F=input('Is there any main focus for today? ')
        time_current=datetime.now()
        start_current=time_current.strftime('%Y/%m/%d %H:%M')

        # =['Day_name','Activity','Start','Days','Assumption_take','Total(m)']
        DF=pd.DataFrame(columns=['Activity'])
        DF=DF.append(pd.Series(None),ignore_index=True) 

        DF['Day_name']=calendar.day_name[time_current.weekday()] 
        DF['Assumption_take(H)']=(f'{assumpt}')
                
        for key,value in data.items():
            DF.loc[DF.index[-1],'Activity']=value.capitalize()
        DF.loc[DF.index[-1],'Start']=time_current.strftime('%Y/%m/%d %H:%M')

        DF_1=DF.squeeze()
        #writing the input data        
        data=pd.read_csv('Activity_2.csv')
        
        data=data.append(DF_1)

        if main_F  not in('no' or 'n'):
            if main_F in ('no' or 'nothing' or 'n'):
                data.loc[data.index[-1],'Main_Focus']= 'None'
            
            else: data.loc[data.index[-1],'Main_Focus']= main_F

        data.to_csv('Activity_2.csv', index=False)


        return time_current
        
    def stop(self):
        global DF
        data=pd.read_csv('Activity_2.csv')
        end_time=datetime.strptime(datetime.now().strftime('%Y/%m/%d %H:%M'),
                                                    '%Y/%m/%d %H:%M') #when we turn string datetime format to datetime, the year shift to right in printing.

        total_time=end_time - datetime.strptime(data.loc[data.index[-1],'Start'],'%Y/%m/%d %H:%M')

        counter_day= (end_time - first_time_day).days
        
        diff=(total_time.seconds /60) /60  # Convert Different time to minut /hour
        
        
        data.loc[data.index[-1],'Total(H)']= f'{diff:.2f}'
        data.loc[data.index[-1],'Days']= counter_day
        data.to_csv('Activity_2.csv',index=False)

        

    def main_focus(self):
        act_am=[]
        act_pm=[]
        
        while True:     
            checker=input('AM or PM').lower()
            act_name=input('Well write the activity:').lower()

            # AM Activities
            if checker in ('am'):
                act_am.append(act_name)

            # PM Activities
            elif checker in ('pm'):
                act_pm.append(act_name)
            
            ender=input('Any more for today?').lower()

            if ender in  ( 'y' ,'yes' ):
                pass
            
            else: break

        # var3 = " -- ".join(a)
        data=pd.read_csv('Activity_2.csv')

        # check if some list is not empty -> raise Error after excution if any of the list was empty.
        if act_am:
            data.loc[data.index[-1],'focus_am']= ' -- '.join(act_am)
        if act_pm:
            data.loc[data.index[-1],'focus_pm']= ' -- '.join(act_pm)
        
        data.to_csv('Activity_2.csv',index=False )
    
    def awake(self):
        get_time=datetime.today()
        
        weekend=get_time.strftime('%A')

        df_1=pd.DataFrame({'Weekend':weekend,
                            'Awake_time':get_time.strftime('%H:%M'),
                            'Status':'Start'},index=[0])
    # a[0].values[0][1]
        df_1=df_1.squeeze()
    
        awake_1=pd.read_csv('Awaking.csv')
        awake_1=awake_1.append(df_1)
        awake_1.to_csv('Awaking.csv',index=False)
    
    def awake_off(self):
        awake_1=pd.read_csv('Awaking.csv')
        end_time=datetime.today()
        try:
            first_time=datetime.strptime(awake_1.loc[awake_1.index[-1],'Awake_time'],'%H:%M')
            
            diff=(end_time - first_time).seconds
            diff=diff /3600 # Hours

            awake_1.loc[awake_1.index[-1],'Awake_time']=f'{diff:.2f}'
            awake_1.loc[awake_1.index[-1],'Status']='End'
            awake_1.to_csv('Awaking.csv',index=False)

        except ValueError as o:
            print(f'-> THe Awake_time is the standard format <- \n'
                 f'          ___Try again___')

        return print('done')

    def awake_status(self):
        awake_1=pd.read_csv('Awaking.csv')
        check=awake_1.loc[awake_1.index[-1],'Status']
        if check ==  'Start':
            print('awake_time has been Started! ')
        else: print('Please run the Starter. ')


obj=myclass()