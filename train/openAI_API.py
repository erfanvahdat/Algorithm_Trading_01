# # open_API="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"
# # import requests
# # import argparse
# import os
# import urllib.request
# from PIL import Image
# import matplotlib.pyplot as plt
# import replicate

# model = replicate.models.get("sczhou/codeformer")
# # export REPLICATE_API_TOKEN=r8_Zk4g6NZCm9hwkakHXwUy8nd5YUfu54x2qVmKv

# def opt_image(path,save_path,name_file):
  
#   version = model.versions.get("7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56")
#     # https://replicate.com/sczhou/codeformer/versions/7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56#input
#   # for index,file in enumerate(os.listdir(path)):

#   path=r'C:\Users\Erfan\Desktop/'
#   file_name="download.jpeg"
#   inputs = {
#         'image': open(os.path.join(path,file_name),'rb'),
#         'codeformer_fidelity': 0.5,
#         'background_enhance': True,
#         'face_upsample': True,
#         'upscale': 2,
#     }
#   output = version.predict(**inputs)
#   print(output)

#     # urllib.request.urlretrieve(output, f"{save_path}\{name_file}_{index}.png")  



# opt_image(path=None,
#           save_path=None,
#           name_file=None)



import plotly.graph_objs as go
import pandas as pd

# Load data
df=pd.read_csv('./ETH-D.csv')

fig = go.Figure()
fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                         y=[df['High'].max(), df['Low'].min()],
                         mode='lines',
                         name='Fibonacci Retracement'))

fig.add_shape(type='line',
              x0=df['Date'].iloc[0],
              y0=df['High'].max(),
              x1=df['Date'].iloc[-1],
              y1=df['Low'].min(),
              line=dict(color='gray', width=1))

levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
for level in levels:
    fig.add_shape(type='line',
                  x0=df['Date'].iloc[0],
                  y0=df['High'].max() - level * (df['High'].max() - df['Low'].min()),
                  x1=df['Date'].iloc[-1],
                  y1=df['High'].max() - level * (df['High'].max() - df['Low'].min()),
                  line=dict(color='gray', width=1, dash='dash'))
# Show the figure
fig.show()



