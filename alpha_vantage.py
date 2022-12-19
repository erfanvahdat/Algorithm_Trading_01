import numpy as np
import pandas as pd
import requests 
import yfinance as yf
API_key=['OW7QOVU34ZFY3MNE']
class_name=['none','bullish_hammer','bearish_hammer','three white soldier','bulish_engulfing','bearish_engulfing']

class yahoo:
    
    def __init__(slef):
        pass

    def get_currency(self,currency):
        for i in currency:
        
                data=yf.download(tickers=f'{i}',period='5y')
                data.to_csv(f'{i}.csv',index='False')

            
            



