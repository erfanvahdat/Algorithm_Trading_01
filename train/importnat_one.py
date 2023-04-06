import os
import torch
import pandas
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import transforms
# from torch.nn.functional import F
import mplfinance as mpf
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset

################mplfinance to plot the chart###################
# df=pd.read_csv('A:/algorithm_trading_01/train/crptoh/AR.csv')
# s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6})
# fig = mpf.figure(figsize=(25,15),style=s) #style=s) # pass in the self defined style to the whole canvas
# ax = fig.add_subplot(2,1,1,) 
# # av = fig.add_subplot(2,1,2, sharex=ax)  # volume chart subplot
# mpf.plot(df,ax=ax,type='candle')
# mpf.plot(df+df.std()[0], type='line', ax=ax)
##############################################################


##################adding macd to mplfinance chart####################
# def MACD(df, window_slow, window_fast, window_signal):

#     macd = pd.DataFrame()

#     macd['ema_slow'] = df['Close'].ewm(span=window_slow).mean()
#     macd['ema_fast'] = df['Close'].ewm(span=window_fast).mean()
#     macd['macd'] = macd['ema_slow'] - macd['ema_fast']
#     macd['signal'] = macd['macd'].ewm(span=window_signal).mean()
#     macd['diff'] = macd['macd'] - macd['signal']
#     macd['bar_positive'] = macd['diff'].map(lambda x: x if x > 0 else 0)
#     macd['bar_negative'] = macd['diff'].map(lambda x: x if x < 0 else 0)
#     return macd

# macd = MACD(df, 12, 26, 9)
# macd_plot = [

# mpf.make_addplot((macd['macd']), color='#606060', panel=2, ylabel='MACD', secondary_y=False),
# mpf.make_addplot((macd['signal']), color='#1f77b4', panel=2, secondary_y=False),
# mpf.make_addplot((macd['bar_positive']), type='bar', color='#4dc790', panel=2),
# mpf.make_addplot((macd['bar_negative']), type='bar', color='#fd6b6c', panel=2),
# ]
"""" also can see the web_page for more information
https://plainenglish.io/blog/plot-stock-chart-using-mplfinance-in-python-9286fc69689"""
##############################################################






##################################  SPlit the Data point ####################################
ch_dev=int(len(df.chikou_span)/4)


split_rate=int(len(df.chikou_span)/4) 


first_value=0
devider_value=2
high=[]
low=[]
open=[]
close=[]

turn =4
        # devide the whole data into turns value
devided_value=int(len(df.chikou_span)/turn)
counter=0

# Splited Date
split_dates=[]
# we miss the first peirod and the last period. to fix this issue we need to add those period manually

count_count=[]
for  i in range(turn):
    counter += devided_value
    if counter >= len(df['chikou_span'])-2:
            print('break',counter)
            break
    
    count_count.append(df.index[counter])
    #first period
    if i==0:
        # split_dates.append(chikou_value[chikou_value.index[0:devided_value]]  ) 
        split_dates.append(df.loc[df.index[0:devided_value]])

        
    # Last period
    elif i == turn -1:
        a=counter - devided_value
        split_dates.append(df.loc[df.index[a:-1]])
        

    
    
    else:
         split_dates.append(df.loc[df.index[counter - devided_value:counter]])




############################## Finding the cross over ##############################

Chik_low=[]
Chik_high=[]

stauts_t=False
status_k=False

cross_below=[]
cross_high=[]


C_high=[]
C_low=[]


for i in split_dates:
    Chik_high.append((i.chikou_span.max(),i.chikou_span.idxmax()))
    Chik_low.append((i.chikou_span.min(),i.chikou_span.idxmin()))

    for xi,(t,k) in enumerate(zip(i['tenkan_sen'],i['kijun_sen'])):

        if t > k and  stauts_t==False:
            cross_below.append((t,i.index[xi]))

            # cross_below_date.append(ich_time[i])
            stauts_t=True

        elif t < k : # kijun_sen cross the t in top
            stauts_t=False
        

        if k >t  and  status_k==False: # we have Cross in bottom
            cross_high.append(( k,i.index[xi] ))
            
            # cross_high_time.append(ich_time[i])
            status_k=True

        elif t > k : # tenkan_sen cross the k in bottom
            status_k=False

    C_high.append(cross_high[:])
    C_low.append(cross_below[:])
    cross_below.clear()
    cross_high.clear()
    



#################################### Finding the lines in the chart  ####################


import datetime
lines=[]
lines_date=[]


max_cross_date=[]
max_cross_price=[]
for i,xlabel in enumerate(Chik_low):
    
    
    if len(C_high[i]) ==0:
        print('we found empty cell')
        continue
         

    # we plote the two at least for one major of chikou_span
    real_time=  xlabel[1]+ datetime.timedelta(days=18) 
    
    max_cross_date=np.array(C_high[i])[:,1].max()
    max_cross_price=np.array(C_high[i])[:,0].max()

    if max_cross_date > real_time or max_cross_price < xlabel[0]:
        print(f'not found in section :{i}')
    

    # the first is price and the second is date
    lines.append([xlabel[0],max_cross_price])
    lines_date.append([xlabel[1],max_cross_date])
    print(f'found in seciton : {i}')



    
# 
fig=plt.figure(figsize=(20,10))
ax=fig.add_subplot(1,1,1)


################################# plotting  the swing in chikou_span ####################################

from ichimoku import  plt_candle,cal_ich

plt_candle('./eth_save.csv',ax=ax,ich=True)


ax.plot(df['chikou_span'],c='green')

plt.scatter(np.array(Chik_low)[:,1],np.array(Chik_low)[:,0] -50 ,marker='^',color='gold')
plt.scatter(np.array(Chik_high)[:,1],np.array(Chik_high)[:,0] +50,marker='v',color='gold')


# # plot the k and t lines
# plt.plot(ich_value['kijun_sen'],c='red',label='kijun_sen')
# plt.plot(ich_value['tenkan_sen'],c='blue',label='tenkan')
# # plotting the corss of k and 
# for high,low in zip(C_high,C_low):
#     for high_a,low_a in zip(high,low):
#         ax.scatter(high_a[1],high_a[0] +100 ,marker='v',c='red')
#         ax.scatter(low_a[1],low_a[0] - 100 ,marker='^',c='blue',)

# # Add the splitted Data

for i in count_count:
    ax.axvline(x=i)

for price,date in zip(lines,lines_date):
    plt.plot(date,price,marker='o',c='black')

# plt.legend()
# plt.xticks(rotation=45)
# plt.show()

    
            