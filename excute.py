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

class myclass():    
    
    global DF
    global first_time_day 
    
    first_time_day=datetime.strptime('2022/12/08 16:15','%Y/%m/%d %H:%M')

    def __init__(self):
        pass
    
    def get_timer(self):
        time=datetime.now()
        return time


    def start(self):
        global start_current

        print('1.Coding 2.Reading 3.Toefl 4. Thinking')    
        block_time=input('Type Block of Activity?').capitalize()
        activity=input('Activity name...  ').capitalize()
        Assumption_timer=int(input('Estimate the time of the Activity  '))

        Block_time=['Thinking','Coding','Reading','Toefl','Think','Code','Read','English']
        # Time block
        Block_time_thinking=['Thinking','think','Though',]
        # reading blokc
        Block_time_reading=['Reading','Read']
        # Codign block
        Block_time_code=['Coding','Code']
        # Toefl block
        Block_time_toefl=['English','Toefl']


        
        if block_time in Block_time_code:
            block_time= 'Coding'
        elif block_time in Block_time_reading:
            block_time= 'Reading'
        elif block_time in Block_time_toefl:
            block_time= 'Toefl'            
        elif block_time in Block_time_thinking:
            block_time= 'Thinking'

        else: print('Try Again, Block time is invalid')

        

        time_current=datetime.now()
        start_current=time_current.strftime('%Y/%m/%d %H:%M')

        # =['Day_name','Activity','Start','Days','Assumption_take','Total(m)']
        DF=pd.DataFrame()
        # DF=pd.concat([DF,pd.Series(None)], axis=0, ignore_index=True, join='outer')
        # DF=DF.append(pd.Series(None),ignore_index=True) 

        DF.loc[-1,'Day_name']=calendar.day_name[time_current.weekday()] 
        DF.loc[-1,'Activity']=activity
        DF.loc[-1,'Start']=time_current.strftime('%Y/%m/%d %H:%M')
        DF.loc[-1,'Assumption_take(H)']=(f'{Assumption_timer}')
        DF.loc[-1,'Block_time']=(f'{block_time}')

     
        DF_1=DF.squeeze()
        # Writing the input data        
        act_data=pd.read_csv('Task.csv')

        result = pd.concat([act_data,DF], axis=0, ignore_index=True, join='outer')
        result.to_csv('Task.csv', index=False)  
        
    def stop(self):
        global DF
        data=pd.read_csv('Task.csv')
        end_time=datetime.strptime(datetime.now().strftime('%Y/%m/%d %H:%M'),
                                                    '%Y/%m/%d %H:%M') #when we turn string datetime format to datetime, the year shift to right in printing.

        total_time=end_time - datetime.strptime(data.loc[data.index[-1],'Start'],'%Y/%m/%d %H:%M')

        counter_day= (end_time - first_time_day).days
        
        diff=(total_time.seconds /60) /60  # Convert Different time to minut /hour
        
        
        data.loc[data.index[-1],'Total(H)']= f'{diff:.2f}'
        data.loc[data.index[-1],'Days']= counter_day
        data.to_csv('Task.csv',index=False)

        print('Stop activity...') 

         
    def awake(self):
        get_time=datetime.today()
        weekend=get_time.strftime('%A')

        df_1=pd.DataFrame({'Weekend':weekend,
                            'Awake_time':get_time.strftime('%H:%M'),
        
                            'Status':'Start'},index=[0])
    # a[0].values[0][1]
        # df_1=df_1.squeeze()
    
        awake_1=pd.read_csv('Awaking.csv')
        awake_1=pd.concat([awake_1,df_1],axis=0,ignore_index=True)
        awake_1.to_csv('Awaking.csv',index=False)
        print('awake_start')
    
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



# git remote -v #checker the fetch and push URL
# git remote add <name> url # adding a new repo
#hi
# # Commit a file into a repo
# git add <name_file> 
# git commit -m '<comment>' 

# #after comming the last file:
# git push <remote>  master
