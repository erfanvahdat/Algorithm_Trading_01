import torch 
import numpy as np
import pandas as pd
from torch import nn
import mplfinance as mpl
import matplotlib.pyplot as plt
from  sklearn.preprocessing import Normalizer
from matplotlib.pyplot import figure, show
from mpl_interactions import ioff, panhandler, zoom_factory
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split as split


# strd_end = np.random.randint(100, 150)
# strd_end = np.append(strd_end, np.random.randint(300, 350))
# sample = df.iloc[strd_end.min():strd_end.max(), :]

##############Turn the series to DF################
# smaple_1=sample['Close']
# ss = smaple_1.to_frame(name='my_column')
###################################################

# upper_close=sample['High'].max()
# lower_close=sample['High'].min()

# # for ind in sample.High:
# UPPER = sample['High'].rolling(window=10).max()
# LOWER = sample['Low'].rolling(window=10).min()

# # two_points = [(sample.index[0], upper_close),(sample.index[-1], lower_close),(sample.index[-1], lower_close+100)]

# # allines = [(upper.index, upper.values), (lower.index, lower.values)]
# colors = ['r', 'g']
# higher=[]
# lower=[]
# for i,(value,ind) in enumerate(zip(sample.High,sample.index)):
#     # print(value,ind)
#     higher.append((ind,value+40))

# for i,(value,ind) in enumerate(zip(sample.Low,sample.index)):
#     # print(value,ind)
#     lower.append((ind,value+-40))


#######################  ZOOOMING ######################################
class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 0.5):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = 1 / base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print (event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion
    
##########################################################################################

# fig = mpl.figure(figsize=(15,5), style='yahoo')
# ax = fig.add_subplot(111,autoscale_on=True) 
# ax.set_title('Click to zoom')
scale = 1.1


# x,y,s,c = np.random.rand(4,200)
# s *= 200
# ax.scatter(x,y,s,c)
# sample = df.iloc[strd_end.min():strd_end.max(), :]


# apds =[mpl.make_addplot(LOWER - 40,color='g',ax=ax,),
#        mpl.make_addplot(UPPER + 50,color='r',ax=ax,marker='v')
#        ]
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

# mpl.plot(sample,addplot=apds, type='candle',ax=ax)
# plt.show()



# plt.title('Sample_Data',fontdict=font)
# plt.grid(False)
# plt.show()
#########################funcirton of chart plotting######################################




def plt_candle(path_data,ich=None,full=None,figsize=None,ax=None):

    data=pd.read_csv(path_data,index_col=0,parse_dates=True)
    print('Data is loaded')
    # Ichimoku-> chikou and kijen_sen -> 26 period
    period26_high = data.High.rolling(window=26).max()
    period26_low = data.Low.rolling(window=26).min()
    kijun_sen = pd.Series((period26_high + period26_low) / 2).rename('kijun_sen')

    # Tenkan_sen -> 9 period
    period9_high = data.High.rolling( window=9).max()
    period9_low = data.Low.rolling(window=9).min()
    tenkan_sen = ((period9_high + period9_low) / 2).rename('tenkan_sen')

    chikou_span = data.Close.shift(-30).rename('chikou_span')
    global df
    df=pd.concat([data,chikou_span,kijun_sen,tenkan_sen],axis=1,join='inner')

    if full==True:
        # Increase and decrease value of Data
        inc= df.Close > df.Open
        dec= df.Close < df.Open

        # OHLC value
        open=np.array(df.Open)
        close=np.array(df.Close)
        low=np.array(df.Low)
        high=np.array(df.High)
        date=np.array(df.index)

        bar_width=3

        ax.vlines(x=date, ymin=low, ymax=high, color='k', linestyle='-',
                linewidth=1)

        ax.vlines(date[inc],open[inc],close[inc],color='green',linewidth=bar_width)
        ax.vlines(date[dec],open[dec],close[dec],color='red',linewidth=bar_width)

        ax.plot(df.chikou_span,c='g', linewidth=1)
        ax.plot(df.kijun_sen,c='red',linewidth=1)
        ax.plot(df.tenkan_sen,c='blue',linewidth=1)

        # Adding zoompan in plotting
        zp = ZoomPan()
        figZoom = zp.zoom_factory(ax, base_scale = scale)
        figPan = zp.pan_factory(ax)
        

    elif  ich == True:
        
        ax.plot(df.chikou_span,c='g', linewidth=1)
        ax.plot(df.kijun_sen,c='red',linewidth=1)
        ax.plot(df.tenkan_sen,c='blue',linewidth=1)
        
    
    
    else:
        print('fill the ich or full method')
        

# print('data path is incorect, try Again!')


def add_plot(ax=None): 
    
    ax.plot(df.Close,c='k')


def cal_ich(data_path):
    data=pd.read_csv(data_path,index_col=0,parse_dates=True)
    # 9 period
    period9_high = data.High.rolling( window=9).max()
    period9_low = data.Low.rolling(window=9).min()
    tenkan_sen = ((period9_high + period9_low) / 2).rename('tenkan_sen')
    # 26  period
    period26_high = data.High.rolling(window=26).max()
    period26_low = data.Low.rolling(window=26).min()
    kijun_sen = pd.Series((period26_high + period26_low) / 2).rename('kijun_sen')
    
    chikou_span = data.Close.shift(-30).rename('chikou_span')
    df=pd.concat([data,chikou_span,kijun_sen, tenkan_sen],axis=1,join='inner')

    return df


#####################################################################

    





# # 
# fig=plt.figure(figsize=(20,10))
# ax=fig.add_subplot(1,1,1)


# ################################# plotting  the swing in chikou_span ####################################

# from ichimoku import  plt_candle,cal_ich

# plt_candle('./ETH-USD.csv',ax=ax,ich=True)

# # chikou=cal_ich('./ETH-USD.csv',)
# # ax.plot(chikou['chikou_span'],c='green')

# ax.scatter(high_swing_date,np.array(high_swing) +100 ,marker='v',color='gold')
# ax.scatter(low_swing_date,np.array(low_swing) -50,marker='^',color='gold')


# ############################## Finding the cross over ##############################
# ich_time=np.array(ich_value.index)
# cross_below=[]
# cross_below_time=[]

# cross_high=[]
# cross_high_time=[]

# stauts_t=False
# status_k=False
# for i,(t,k) in  enumerate(zip(ich_value['tenkan_sen'],ich_value['kijun_sen'])):
#     if t > k and  stauts_t==False:
#         cross_below.append(t)
#         cross_below_time.append(ich_time[i])
#         stauts_t=True

#     elif t < k : # kijun_sen cross the t in top
#         stauts_t=False
    

#     if k >t  and  status_k==False: # we have Cross in bottom
#         cross_high.append(k)
#         cross_high_time.append(ich_time[i])
#         status_k=True

#     elif t > k : # tenkan_sen cross the k in bottom
#         status_k=False





# # plot the k and t lines
# plt.plot(ich_value['kijun_sen'],c='red',label='kijun_sen')
# plt.plot(ich_value['tenkan_sen'],c='blue',label='tenkan')
# # plotting the corss of k and 
# plt.scatter(cross_high_time,np.array(cross_high) +100 ,marker='v',c='red',label='cross_t')
# plt.scatter(cross_below_time,np.array(cross_below) -100 ,marker='^',c='blue',label='cross_t')
# # Add the splitted Data
# for i in save_dates:
#     ax.axvline(x=i)

# plt.legend(loc="upper left")
# plt.xticks(rotation=45)
# plt.show()
