# # import backtrader as bt
# # import backtrader.indicators as btind

# # # Create a strategy
# # class MyStrategy(bt.Strategy):
# #     params = (("macd_period_fast", 12), ("macd_period_slow", 26), ("macd_period_signal", 9))

# #     def __init__(self):
# #         self.data_close = self.datas[0].close
# #         self.macd = btind.MACD(self.data.close, period_me1=self.params.macd_period_fast, period_me2=self.params.macd_period_slow, period_signal=self.params.macd_period_signal)

# #     def next(self):
# #         print(f'Close Price: {self.data_close[0]}')
# #         print(f'MACD: {self.macd.macd[0]}')
# #         print(f'Signal Line: {self.macd.signal[0]}')
# #         print(f'Histogram: {self.macd.histo[0]}')

# # # Create a cerebro
# # cerebro = bt.Cerebro()

# # # Add the strategy
# # cerebro.addstrategy(MyStrategy)

# # # Load the data
# # data = bt.feeds.PandasData(dataname=)
# # cerebro.adddata(data)

# # # Run the strategy
# # cerebro.run()

