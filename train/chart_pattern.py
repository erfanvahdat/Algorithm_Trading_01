import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

 
class Chart_pattern:
    def __init__(self,data):
        self.data=data
        self.result={"two_pairs":[],
                        "cross_high":[],
                        "cross_low":[],
                        "three_pair":[],
                        "duplicated_high":[],
                        "duplicated_low":[],
                        "Draw_line":[],
                        "check_time_enter":[],
                        "EnterPrice":[],
                        "postion_inf": [],
                        "fake_signal":[]
                        }
        
        self.swing_low=[]
        self.swing_high=[]
        self.two_pairs=[]
        
        self.duplicate_low=[]
        self.duplicate_high=[]
        
        self.cross_high=[]
        self.cross_low=[]

        self.two_pair()
        self.draw_trend()
        

    def time_eval(self):
        
        # check if our datetime fromat is hourly
        if  self.data.index[2] - self.data.index[1] ==datetime.timedelta(hours=1):
            time_st_40=datetime.timedelta(hours=40)
            time_st_30=datetime.timedelta(hours=30)
            
            status_interval='hour'


        elif  self.data.index[2] - self.data.index[1] ==datetime.timedelta(days=1):
            time_st_40=datetime.timedelta(days=40)
            time_st_30=datetime.timedelta(days=30)
            
            status_interval='day'

            
        else:
            # df.index[2] - df.index[1] ==datetime.timedelta(minutes=30):
            time_st_40=datetime.timedelta(minutes=1200)
            time_st_30=datetime.timedelta(minutes=900)

            status_interval='minute'

        return time_st_40, time_st_30, status_interval

    def two_pair(self):
        
        self.data['base_line']=ta.trend.ichimoku_base_line(high=self.data.High,low=self.data.Low)
        self.data['convertion_line']=ta.trend.ichimoku_conversion_line(high=self.data.High,low=self.data.Low)

        # we track the close price
        
        low_status=False
        high_status=False
        status_t= False
        status_chik= False

        

        time_st_40,time_st_30,_=self.time_eval()
        print(f'{_} Timeframe data as input')

        for i in range(len(self.data)):
            
            # Maximum Price in 5 period of candle
            # C=df.Close.iloc[i:i+5].groupby(np.arange(len(df.Close.iloc[i:i+5 ])) // 5).max()
            

            # Price chekcer for find the best swing,
            """
                By default we choose to investigate the 10 candle to finding the best swing in each period.
                but also we can decrease this period to 5 and it's optional to increase to more than 10 candle.

            """
            
            

            C,C_index=self.data.Close.iloc[i: i+1].max(),self.data.Close.iloc[i:i+1].idxmax()
            Z,Z_index=self.data.Close.iloc[i: i+1].min(),self.data.Close.iloc[i:i+1].idxmin()
            
            
        
        
            # maximum price in 40/30 candle before
            validity,third_candl=self.data.Close.loc[ C_index - time_st_40 :C_index], self.data.Close.loc[C_index : C_index + time_st_30 ]

            
            
            if C_index - time_st_40   < self.data.index[0]:
                continue

            # High swing
            if (C > validity).sum() ==40  and   (C > third_candl ).sum() ==30:
            
                # we have updated low_siwng
                if high_status == True and low_status == False:
                    print('duplicated  high')
                    # Replace the new swing to prevous one
                    self.result['duplicated_high'].append(self.swing_high[-1] )
                    self.swing_high.pop()
                    self.swing_high.append( [ C,C_index] )
                    
                else:
                    self.swing_high.append([ C,C_index])
                    high_status= True


            # Low swing
            if (Z < validity).sum() ==40  and   (Z < third_candl).sum() ==30:
                

                if low_status  == True and high_status == False:
                    
                    # Replace the new swing to prevous one
                    self.result['duplicated_low'].append(self.swing_low[-1])
                    self.swing_low.pop()
                    self.swing_low.append([Z,Z_index])


                else:
                    self.swing_low.append([Z,Z_index])
                    low_status=True


            if  low_status == True and high_status == True:
                if self.swing_high[-1][1] < self.swing_low[-1][1]:
                    continue

            # Founding the two pair
            if low_status == True and high_status == True:
                print('we have found valid two pair...')
                self.result['two_pairs'].append( [self.swing_low[-1],self.swing_high[-1]] )
            
            # reset the status
                low_status,high_status = False,False
            
            

        if len( self.result['two_pairs']) == 0:
            print('no valid two pair')
                

            
        #################### Finding Crosses #########################        
        
        self.data['chikou_span']=self.data.Close.shift(-30)

        for xi,(chik,t) in enumerate(zip(self.data['chikou_span'],self.data['convertion_line'])):
                    
            # finding tenkan_sen -> corssing from below

            if t >  chik  and  status_t==False:
                self.result['cross_high'].append([t,self.data['chikou_span'].index[xi] ])

                # cross_below_date.append(ich_time[i])
                status_t=True

            elif t < chik : # kijun_sen cross the t in top
                status_t=False
            

            if chik >  t  and  status_chik==False: # we have Cross in bottom
                self.result['cross_low'].append([ chik,self.data['chikou_span'].index[xi] ])
                
                # cross_high_time.append(ich_time[i])
                status_chik=True

            elif t > chik : # tenkan_sen cross the k in bottom
                status_chik=False

        #################### Finding Crosses #########################        

    def draw_trend(self):
        draw_series=pd.Series(data=np.array(self.result['cross_high'])[:,0],index=np.array(self.result['cross_high'])[:,1])

        _,_,status_interval=self.time_eval()
        
        if status_interval == 'day':
            Daily_shifted=datetime.timedelta(days=30)

        elif status_interval == 'hour':
            Daily_shifted=datetime.timedelta(hours=50)

        else:
            # 20 candle waiting for enter
            Daily_shifted=datetime.timedelta(minutes=600)


        for i,xi in  enumerate(np.array(self.result['two_pairs'])):

            first_time=xi[0][1]
               # Daily shifted added to protect the soon entery in sell positon

            last_time=xi[1][1] + Daily_shifted
            last_price=xi[1][0]
                
            Dat=draw_series[first_time:last_time]
            
            if len(Dat) == 0:
                print('we found no crosses in high swing')
                continue
            
            
            first_datapoint=(self.data.Low[self.data.index ==np.array(xi)[0,1]].values[0]  ,xi[0][1] )

            ## --->> lines 50 candle is shifted in minute timeframe
            line_shifted=datetime.timedelta(minutes=1500)
            second_datapoint=(Dat[-1],Dat.index[-1]+ line_shifted)
            

            
            x_1=pd.date_range(first_datapoint[1],second_datapoint[1],freq='30min')
            y_1=np.linspace(first_datapoint[0],second_datapoint[0],num=len(x_1))

            
            check_price=np.array(self.result['two_pairs'])[-1,1,0]
            check_date=np.array(self.result['two_pairs'])[-1,1,1]

            self.result['Draw_line'].append(
                                    [y_1,
                                        x_1])
            
            print('we found a valid two_pair')
            self.result['check_time_enter'].append([last_price,last_time])
            
            self.Find_Enter()        

        
            
    def Find_Enter(self):

        # for i in X:
        #     if len(i) == 0:
        #         print('not able to find any singal')
        #         break

        
        
        
        
        
        price,date=self.result['check_time_enter'][-1][0],self.result['check_time_enter'][-1][1]
        
        
        line=pd.Series(data=self.result['Draw_line'][-1][0], index=self.result['Draw_line'][-1][1])
        high_checker=self.data.High.loc[date:]
        

        price_tracker= self.data.High.loc[date:]
        # we are above the draw_line            
        if (price > line ).sum()  ==len (line):
            
            for j,(yprice,indexi) in enumerate(zip(self.data.High.loc[date:],price_tracker.index  )):
                
                if yprice -  line.loc[indexi]   <= 0 :
                    self.result['EnterPrice'].append([yprice,  self.data.index[self.data.High == yprice ] ])
                    break

                # Setting stop sl/tp
                # if len (self.result['EnterPrice'][-1]) != 0:
                #     first_sl=np.array(self.result['two_pairs'])[i,1,0]
                #     first_tp=( np.array(self.result['two_pairs'])[i,0,0] + price ) /2
                #     second_tp= ( np.array(self.result['two_pairs'])[i,0,0] + first_tp)/2

                #     # postive tp devided by negetive sl
                #     risk_reward_first_tp= ((yprice - first_tp) / first_tp) /  ((np.array(self.result['two_pairs'])[i,1,0]  - yprice) / yprice)
                #     risk_reward_second_tp= ((yprice - second_tp) / second_tp) /   ((np.array(self.result['two_pairs'])[i,1,0]  - yprice) / yprice)
                #     self.result['postion_inf'].append([first_sl,first_tp,second_tp,risk_reward_first_tp,risk_reward_second_tp])

                

        # we are below the line
        # else:
        #     print('we found the fake singal')

                # if  len(self.result['Draw_line'][-1]) !=0:
                    
                    # self.result['fake_signal'].append(self.result['Draw_line'][])
                    # self.result['Draw_line'].pop(-1)
            


obj=Chart_pattern(data)
result=obj.result
data=obj.data

def plot_result(data,result):

    fig=plt.figure(figsize=(25  ,7))
    ax=fig.add_subplot(1,1,1)
    plt_candle(data=data,ax=ax,period=18,path_data=False,full=True,ich=False)
    

    alpha= data.Close.std() - 0.01
    
    plt.scatter(np.array(result['two_pairs'])[:,0][:,1],np.array(result['two_pairs'])[:,0][:,0] -  alpha,c='green',marker='^',label='swing_low')
    plt.scatter(np.array(result['two_pairs'])[:,1][:,1],np.array(result['two_pairs'])[:,1][:,0] + alpha ,c='red',marker='v',label='swing_high')

    
    date_refrence=np.array(result['two_pairs'])[0,1,1]
    time_extend= datetime.timedelta(hours=30)
    time_extend_inf= datetime.timedelta(hours=20)

    font = {'family' : 'Arial',
            'weight' : 'bold',
            'size'   : 10}


    # Enter candle
    # plt.vlines(x=np.array(result['EnterPrice'])[:,0] , ymax=np.array(result['EnterPrice'])[:,1]+ alpha, ymin=np.array(result['EnterPrice'])[:,1],
    #                                 linewidth=2 ,color='red')   
    
    # Lines
    for i in result['Draw_line']:
        plt.plot(i[1],i[0],label='valid_signal',c='black')

    for i in result['fake_signal']:
        plt.plot(i[1],i[0],label='fake_signal',c='red')

    if len(result['EnterPrice'])  != 0:
        enter_price,enter_date=result['EnterPrice'][-1][1],result['EnterPrice'][-1][0]

        # i=result['postion_inf'][-1]
        # sl_price=i[0]
        # tp_price=i[1]
        # tp2_price=i[2]
        # r1=i[3]
        # r2=i[-1]
        
        for i in result['EnterPrice']:  
            plt.annotate(text=f'enter',xy=(i[1],i[0]) ,xytext=(i[1],i[0]+ alpha),arrowprops=dict(arrowstyle= '<|-|>',
                                color='blue',
                                lw=1.25,
                                ls='-'),**font)


        # ax.annotate(text='Enter',xy=(enter_date,enter_price - alpha*alpha) ,arrowprops=dict(arrowstyle= '<|-|>',
        #                         color='blue',
        #                         lw=1,
        #                         ls='-'),**font,xytext=(enter_date,  enter_price + alpha*2),label='Enter')
                                    
    # First sl
    # ax.hlines(y=sl_price,xmin=date_refrence,xmax=date_refrence + time_extend,
    #                     color='#990000', linestyle='solid',linewidth=2)
    # plt.annotate(text=f'Sl -> {sl_price :.2f}',xy=(date_refrence + time_extend_inf , sl_price) ,**font)

    # # First tp
    # ax.hlines(y=tp_price,xmin=date_refrence,xmax=date_refrence + time_extend,
    #                     color='green', linestyle='dashed')
    # plt.annotate(text=f'tp_1 -> {tp_price :.2f}',xy=(date_refrence + time_extend_inf , tp_price) ,**font)
    
    # # Second tp
    # ax.hlines(y=tp2_price,xmin=date_refrence,xmax=date_refrence + time_extend,
    #                     color='green', linestyle='solid',)
    
    # plt.annotate(text=f'tp_2 -> {tp2_price :.2f}',xy=(date_refrence + time_extend_inf , tp2_price), **font)

    # # Enter point
    # ax.hlines(y=enter_price,xmin=date_refrence,xmax=date_refrence + time_extend,
    #                     color='#D8D056', linestyle='solid',label='Enter')
    
    # plt.annotate(text=f'Enter point -> {enter_price :.2f}',xy=(date_refrence + time_extend_inf , enter_price), **font)
    
    
    ax.set_title(loc='Center',label='Test',font=font)
    # ax.gca().set_title('title')
    ax.legend(loc='best',prop={'size':10})
    
    # plt.savefig('plotting.jpeg')
    plt.scatter(np.array(result['check_time_enter'])[:,1],np.array(result['check_time_enter'])[:,0])
    
    plt.plot(data['convertion_line'])
    plt.plot(data['chikou_span'])
    plt.plot()
    

plot_result(data=data,result=result)
