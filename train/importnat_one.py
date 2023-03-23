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