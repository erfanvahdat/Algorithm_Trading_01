

## To do list
1. add the function counter form the __one.ipynb__ and add it it __Tester.py main class__
2. find a way to predict the feature label, the features engineer about the dataset
3. calculating hte macd by pandas.bt (only Sell)
4. Test the archeticture



### where are we and what are we doing....
    our featuers is the the signal of macd. trained to understand what and where is 0 and 1 in the chart. then we pick a atleast 30 windows of last
     candle to predict the next candl.not all the candle must would end with the probabality of above 60 percent. just bunch of a little  candle would
     eventually end with the signal buy and sell

----
## Done
1. do some backtrader on the indicator
2. calculating hte macd by pandas.bt (only Buy)
3. 
 

###  Want to do at the end of the year

- 


## PyAututoGui with CV2 for recording the screen
import cv2
import numpy as np
import pyautogui

fourcc = cv2.VideoWriter_fourcc(*'XVID')
screen = (1920, 1080)
out = cv2.VideoWriter('screen_recording.avi', fourcc, 20.0, screen)

screen_size = (1920, 1080)

while True:
    img = pyautogui.screenshot()
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cv2.destroyAllWindows()



import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, SpanSelector

# Create the figure and axes for the subplot
fig, ax = plt.subplots()

# Plot the Close price on the bottom axis
ax.plot(short.index, short.Close, label='Close')

# Create the top axes for the macd and signal values
ax2 = ax.twinx()

# Plot the macd and signal values on the top axis
ax2.plot(short.index, short.macd, c='g', label='macd')
ax2.plot(short.index, short.signal, c='r', label='signal')

# Add crosshairs to the plot
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# Add data dragging to the plot
def onselect(xmin, xmax):
    print(f'Selected x range: {xmin} to {xmax}')

span = SpanSelector(ax, onselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))

# Show the legend
ax.legend()
ax2.legend()

# Display the plot
plt.show()


