# data=pd.read_csv('./cryptoh/BAT.csv',parse_dates=True,index_col=0)
# import yfinance as yf

import requests
import json

def crypto_api():
    api_key='F482043C-CE3B-47AE-8952-3F1DAEC2843A'


    url = 'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history?period_id=1DAY&time_start=2016-01-01&time_end=2016-03-11'
    headers = {'X-CoinAPI-Key' : api_key}
    response =( requests.get(url, headers=headers)).json()

    with open('jsondata.json', 'w') as f:
        json.dump(response, f)

    json_1=pd.read_json("jsondata.json",)

    df=json_1
    df.index=pd.to_datetime(df['time_period_start'])
    df=df.iloc[:,4:-1]
    df=df.rename({'price_open': 'Open', 'price_high': 'High','price_low':'Low','price_close':'Close','volume_traded':'Volume'}, axis='columns')

    return data
datya=crypto_api()
