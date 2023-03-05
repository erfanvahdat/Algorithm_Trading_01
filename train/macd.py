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
        self.label.clear()
        self.X_list.clear()

        # df=pd.read_csv('ETH-USD.csv')
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
        box=[]
        
        # Buy Signal
        for index,z in enumerate(self.data['signal']): 

            if len(self.sell_counter) < 2 : # if we didint have any sell we track the index
                if z ==0: # Sell signal
                    self.sell_counter.append(0)
                    # Track the first sell
                    self.first_index=index

            elif  z ==1: # Buy signal
                self.buy_counter.append(1)
                # Track the last Sell
                self.last_index=index
        
            # elif len(self.sell_counter) >=1  and len(self.buy_counter) <2:
                # we have a buy cross
                
                # Cleare the counter for next round
                self.buy_counter.clear()
                self.sell_counter.clear()
                # print(last_index,first_index)
                # X=torch.concat(,)  # Turn int numpy array
                
                values=(self.data.loc[self.first_index:self.last_index+1,['Open','High','Low','Close','Adj Close','Volume']]).values
                self.X_list.append((self.data.loc[self.first_index:self.last_index+1,['Open','High','Low','Close','Adj Close','Volume']]).values)
                len_current_value_buy=len(values)
                self.label.append(np.ones(len_current_value_buy))
                
                


        # Sell Signal
        for index,z in enumerate(self.data['signal']): 

            if len(self.buy_counter) < 2 : # if we didint have any Buy  we track the index
                if z == 1: # Buy signal
                    self.buy_counter.append(0)
                    self.first_index=index

            elif  z ==0: # Sell signal
                self.buy_counter.append(1)
                self.last_index=index
                        
                # Cleare the counter for next round
                self.buy_counter.clear()
                self.sell_counter.clear()
                # print(last_index,first_index)
                # X=torch.concat(,)  # Turn int numpy array


                values=(self.data.loc[self.first_index:self.last_index+1,['Open','High','Low','Close','Adj Close','Volume']]).values
                self.X_list.append((self.data.loc[self.first_index:self.last_index+1,['Open','High','Low','Close','Adj Close','Volume']]).values)
                len_current_value_sell=len(values)

                self.label.append(np.zeros(len_current_value_sell))

                

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

        # X=[torch.from_numpy(self.X_list[i]).type(torch.float) for i in range(len(self.X_list))] # flaot64 format
        # # y=torch.tensor(np.array(self.label))  # int64 format

        # for indexi  in range(len(X)):
        #     z=torch.full( len(X[indexi]),y[indexi] )


        from sklearn import preprocessing 
        # X=[(self.X_list[i]) for i in range(len(self.X_list))] # float32 format

        A=[]

        # for indexx ,valuey  in zip(range(len(self.X_list)), self.label):
        #     if valuey==1:

        #         A.append(torch.ones(len(self.X_list[indexx])* valuey))
        #     else:
        #         A.append(torch.zeros(len(self.X_list[indexx])) )

        # label=[]
        # for iny  in range(len(self.label)):
        #     label_y=self.label[iny][0]
        #     label.append(label_y)

        # return self.X_list,A
        return self.X_list,self.label

        
# Excute
macd=MACD()


X,y=macd.signal_macd()



# from sklearn.model_selection import train_test_split as split





""" labeling binomial for each rows of candle """"
# X_con=torch.tensor(torch.tensor(X[0])).float()
# y_con=torch.tensor(torch.tensor(y[0])).float()

# for i in range(len(X)):
#     if i>=1:
#         X_con=torch.concat([torch.Tensor(X[i]),X_con],axis=0)    
#         y_con=torch.concat([torch.Tensor(y[i]),y_con],axis=0)    
    
# X_train, X_test, y_train, y_test= split(X_con, y_con, test_size= 0.4, random_state=0)  


""" Labeling binomial for each flatten period of Buy and Sell """
# X_con=[]
# y_con=[]
# for i in range(len(X)):
#     X_con.append(torch.flatten(torch.tensor(X[i]).float()))
#     y_con.append(y[i][0])
# y_con=torch.tensor(y_con).float()

# X_train, X_test, y_train, y_test= split(X_con, y_con, test_size= 0.4, random_state=0)  


# print(len(y_con),len(X_con))




