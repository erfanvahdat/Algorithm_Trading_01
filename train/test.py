import chart_pattern  as cp

import yfinance as yf

data=yf.download('btc-usd',start='2023-03-20',end='2023-03-25',interval='30m')
obj=cp.Chart_pattern(data)
result=obj.result
data=obj.data

print(result)
cp.plot_result(data=data,result=result)
plt.plot()