import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path
from datetime import datetime
class Chart():
    def __init__(self) -> None:
        pass

    def chart_tracker(self):
        
        

        

        path_file="A:\Algorithm_Trading_01/chart_result.csv"
        if path.exists(path_file):
            # check for existing path_data
            print ("File exist")    

            start_current=datetime.now()
            start=start_current.strftime("%Y/%m/%d") # with minute

            currency=input('Currency: ')
            type_positon=input('Type Positon: ')
            Price= input('Enter Price: ')
            Leverage=input('Leverage: ')
            Sl=input('SL: ')
            Tp=input('TP: ')
            
            DF=pd.read_csv(path_file) #            
            
            # DF=pd.DataFrame(columns=['time','Currency','type_postion','Price',
            #                             'Leverage','Sl','Tp'])            

            
            
            DF.loc[-1,'Time']=start.format()
            DF.loc[-1,'Currency']=currency.upper()
            DF.loc[-1,'type_position']=type_positon
            DF.loc[-1,'Price']=Price
            DF.loc[-1,'Leverage']=Leverage
            DF.loc[-1,'Sl']=Sl
            DF.loc[-1,'Tp']=Tp
            
            
            # result = pd.concat([DF,df], axis=0, ignore_index=True, join='outer')
            DF.to_csv('A:\Algorithm_Trading_01/chart_result.csv',index=False)

            # print('data has been updated. ')
            # print(df)

            
        else:
            print ("File not exist")
            df=pd.DataFrame(columns=['Time','Currency','type_position','Price',
                                        'Leverage','Sl','Tp'])
            
            df.to_csv('A:\Algorithm_Trading_01/chart_result.csv',index=False)
        # data=pd.read_csv('chart_result')
        # return result        


obj=Chart()
obj.chart_tracker()