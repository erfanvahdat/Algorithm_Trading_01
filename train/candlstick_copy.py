# # Calculate MACD values using the pandas_ta library
# import pandas_ta as ta
# import pandas as pd
# from candlestick import candlestick
# import matplotlib.pyplot as plt


# class Candle():
#     def __init__ (self):
#         self.data= 0

#     def candstick(self,data):
#         # macd
#         # self.data= pd.read_csv(f'{data}.csv',parse_dates=True,index_col='Date')
#         # macd=self.data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
#         self.data=data
#         # Defining OHLC
#         ohlc=["Open", "High", "Low", "Close"]
#         # candlstick pattern with candlestick
#         self.data=candlestick.doji_star(self.data,ohlc,target='doji_start')
#         self.data=candlestick.bearish_engulfing(self.data,ohlc,target='bearish_engulfing')
#         self.data=candlestick.bullish_engulfing(self.data,ohlc,target='bullish_engulfing')
#         self.data=candlestick.hammer(self.data,ohlc,target='hammer')
#         self.data=candlestick.gravestone_doji(self.data,ohlc,target='gravestone_doji')
#         self.data=candlestick.dragonfly_doji(self.data,ohlc,target='dragonfly_doji')
#         self.data=candlestick.morning_star_doji(self.data,ohlc,target='morningstar')        
#         return self.data

# # Excute_1
# ca=Candle()

import finplot as fplt
import pandas as pd
import mplfinance as mtp
data=pd.read_csv('training/cryptoh/ADA.csv',parse_dates=True,index_col=0) # Data

# fplt.candlestick_ochl(data[['Open', 'Close', 'High', 'Low']])
# fplt.show()
mtp.plot(data)
mtp.plot(data)
mtp.show()