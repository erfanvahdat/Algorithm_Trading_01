import pandas as pd
import os
import numpy as np
import yfinance as yf
from tradingview_ta import TA_Handler

class YF():
    def __init__(self):
        pass

    def yfdownlaod(self,tickers,interval,period,
                   start=None,
                   end=None,
                   ):
        print('  hint ->\n Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max \n',
            'Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo ')
        # for i in tickers:
        data = yf.download(tickers, period=f'{period}' ,interval=f'{interval}')
        data.to_csv(f'{tickers}.csv')
        print(f'data has been save to {tickers}.csv in current path')
        return data

    def live(self,symbol,interval):
        handler = TA_Handler(
        

        symbol=f"{symbol}",
        exchange= 'KRAKEN' or 'BITTREX ' or 'BINANCE' or 'COINEX' or 'BINANCEUS' or 'GATEIO',
        
        screener="crypto",
        # valid  interval: (ex: 1m, 5m, 15m, 1h, 4h, 1d, 1W, 1M)
        interval=f"{interval}",timeout=None)

        analysis = handler.get_analysis()
        Rsi=analysis.indicators["RSI"]
        Macd_slow=analysis.indicators["MACD.macd"]
        Macd_fast=analysis.indicators["MACD.signal"]
        Open=analysis.indicators["open"]
        High=analysis.indicators["high"]
        Low=analysis.indicators["low"]
        Close=analysis.indicators["close"]
        change=analysis.indicators["change"]
        exchange=analysis.exchange        

        # # put them into aray to pass it to DF
        data=np.array([Open,High,Low,Close,Rsi,Macd_slow,Macd_fast,change,exchange])

        # Dataframe Data
        data=pd.DataFrame([data],columns=['Open','High','Low','Clos','Rsi','Macd_slow','Macd_fast','change','Exchange'])
    
        return data
            
# Excuted
yfin=YF()
# yfin.yfdownlaod(tickers='BTC-USD',interval='1h',period='1mo')

import yfinance as yf

df = yf.download(
tickers = ["DOGE-USD", "BTC-USD"],
interval = "15m")



