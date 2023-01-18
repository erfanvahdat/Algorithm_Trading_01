import pandas_ta as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch


# Calculate MACD
class MACD():
    def __init__(self):
        super(MACD, self).__init__()
        self.data=pd.read_csv('BTC-USD.csv') # Data
        self.X_list=[]
        self.label=[]
        self.first_index=[]
        self.last_index=[]

        self.buy_counter=[]
        self.sell_counter=[]
    def signal_macd(self):
        # Calculate MACD values using the pandas_ta library
        

        df=pd.read_csv('BTC-USD.csv')
        df=self.data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        
        # Create empty columns for buy and sell signals
        self.data["Buy"] = None
        self.data["Sell"] = None
        
        
        # Fill in buy and sell columns
        for i in range(len(self.data)):
            if i > 0:
                if self.data.at[i, "MACD_12_26_9"] > self.data.at[i, "MACDs_12_26_9"] and self.data.at[i-1, "MACD_12_26_9"] < self.data.at[i-1, "MACDs_12_26_9"]:
                    self.data.at[i, "signal"] =1
                elif self.data.at[i, "MACD_12_26_9"] < self.data.at[i, "MACDs_12_26_9"] and self.data.at[i-1, "MACD_12_26_9"] > self.data.at[i-1, "MACDs_12_26_9"]:
                    self.data.at[i, "signal"] = 0

     

        # # Buy label
        for index,z in enumerate(self.data['signal']): 
            if z ==0: # Sell signal
                self.sell_counter.append(1)
                self.first_index=index

            if z ==1: # Buy signal
                self.buy_counter.append(1)
                self.last_index=index
                
            if len(self.sell_counter) >=1  and len(self.buy_counter) <2:
                # we have a buy cross
                self.last_index=index
                # Cleare the counter for next round
                self.buy_counter.clear()
                self.sell_counter.clear()

                self.X_list.append((self.data.loc[self.first_index:self.last_index+1,
                            ['Open','High','Low','Close','Adj Close','Volume']]).values ) # Turn int numpy array
                self.label.append(1)


        # Sell label
        for index,z in enumerate(self.data['signal']): 
            if z ==1: # Buy signal
                self.buy_counter.append(1)
                self.first_index=index

            if z ==0: # Sell signal
                self.sell_counter.append(1)
                
            if len(self.buy_counter) >=1 and len(self.sell_counter) <2:
                # we have a Sell cross
                self.last_index=index
                # Cleare the counter for next round
                self.buy_counter.clear()
                self.sell_counter.clear()
                # Print(self.last_index,self.first_index)
                self.X_list.append((self.data.loc[self.first_index:self.last_index+1,
                                    ['Open','High','Low','Close','Adj Close','Volume']]).values ) # Turn int numpy array
                self.label.append(0)


                

        # plt.figure(figsize=(10, 4)) 
        # ax1 = plt.subplot(3, 1, 1)
        # plt.plot(self.data.Date, self.data.Close)
        # ax1.set_title('Close Price')

        # ax2 = plt.subplot(3, 1, 2)
        # plt.plot(self.data.Date, self.data['MACD_12_26_9'], c='black')
        # plt.plot(self.data.Date, self.data['MACDs_12_26_9'], c='red')
        # plt.scatter(np.arange(len(self.data)),self.data.Buy,marker='^',c='green')
        # plt.scatter(np.arange(len(self.data)),self.data.Sell,marker='v',c='red')
        # ax2.set_title('MACD and Signal')
        # plt.show()

        X=[torch.tensor(x) for x in self.X_list] # flaot64 format
        y=torch.tensor(np.array(self.label)).long()  # int64 format

        # return self.X_list,np.array(self.label)
        return X,y
        
                
# Excute
macd=MACD()
