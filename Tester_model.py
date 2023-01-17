import torch
import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split as split
from torch.utils.data import TensorDataset, DataLoader
from  tqdm import tqdm, trange, tqdm_notebook



# Load and preprocess the data
path=f'cryptoData/'
df=pd.read_csv(f'{path}BNB-USD.csv',parse_dates=True,index_col='Date')
data_YY=df['Close'].shift(30)
data_YY=torch.tensor(data_YY.fillna(data_YY.mean()))
data_XX=torch.tensor(df.values)

data =df.astype(float) # Convert the data to float type
data = (df -df.mean()) / df.std() # Normalize the data
data = torch.tensor(data.values).float() # Create a tensor from the data

# f
window_size = 30
data_X = []
data_y = []
for i in range(len(data)-window_size):
    data_X.append(data_XX[i:window_size+i])
    data_y.append(data_YY[i:window_size+i])
    

data_X = torch.stack(data_X)
data_y = torch.stack(data_y)

# print(data_X,data_y)
# Define the model architecture -> 

class FinancialModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(FinancialModel, self).__init__()
        # self.rnn = torch.nn.GRU(input_size, 64, num_layers=2, batch_first=True)
        # self.fc = torch.nn.Linear(64, output_size)
        self.fc = torch.nn.Linear(6, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        # self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # x, _ = self.rnn(x)
        # x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        x=self.fc2(x)
        return x

X_train,X_test,y_train,y_test= split(data_X,data_y,test_size=0.2,shuffle=False)
model = FinancialModel(input_size=X_train.shape[1], output_size=30)


train_data = TensorDataset(X_train, y_train)
train_loader=DataLoader(train_data,shuffle=False)


# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

# Define the number of training iterations
num_iterations = 5

# print(X_train.shape[1]) # must be 30

for iteration in range(num_iterations):
    Loss=[]
    loss= 0 
    for X_batch, y_batch in tqdm_notebook((train_loader)):

        optimizer.zero_grad()
        output = model(X_batch.squeeze().float())

        # plt.ion()
        loss = loss_fn(output.squeeze().float(), y_batch.squeeze().float())
        Loss.append(loss)
        loss.backward()
        optimizer.step()
        # loss += loss
    
    with torch.no_grad():
        if iteration %2 ==0:
            print('loss ->',loss.item())   
            

        # with torch.no_grad():
        #     plt.clf() # clear the plot
        #     plt.plot(np.arange(len(Loss)), Loss,c='red')
        #     plt.show()
        #     plt.pause(0.001) # pause for a short time

    
















# """
#  testing the raw model and the trained model
# """


# path=f'cryptoData/'
# df=pd.read_csv(f'{path}ADA-USD.csv',parse_dates=True,index_col='Date')
# data =df.astype(float) # Convert the data to float type
# data = (df -df.mean()) / df.std() # Normalize the data
# data = torch.tensor(data.values).float() # Create a tensor from the data
# #

# # X_data
# data_X=np.delete(data,[-2],axis=1) # expect Adj_close
# data
# # Y_data
# data_y=data[:,-2] # Adj_close
# X_train,X_test,y_train,y_test= split(data_X,data_y,test_size=0.8,shuffle=False)

# # Define the bach size
# batch_size = 32
# num_batches = len(X_train) // batch_size

# from torch.utils.data import TensorDataset, DataLoader

# # create TensorDataset
# train_data = TensorDataset(X_train, y_train)
# # create DataLoader
# train_loader = DataLoader(train_data, batch_size=batch_size,) # shuffle=True



# class MyModel(torch.nn.Module):
#     def __init__(self, input_size, output_size):
#         super(MyModel, self).__init__()
#         self.fc1 = torch.nn.Linear(input_size, 64)
#         self.fc2=torch.nn.Linear(64,64)
#         self.fc3=torch.nn.Linear(64,output_size)

#     def forward(self,x):
#         x=torch.nn.functional.relu(self.fc1(x))
#         x=torch.nn.functional.relu(self.fc2(x))
#         x=self.fc3(x)
#         return x

# raw_model = MyModel(input_size=X_test.shape[1],output_size=1)
# model_trained = MyModel(input_size=X_test.shape[1],output_size=1)




# def Eval(X_test,y_test):
#     loaded_model=torch.load('my_model.pt')
#     model_trained.load_state_dict(loaded_model)

#     loaded_model_raw=torch.load('raw_model.pt')
#     raw_model.load_state_dict(loaded_model_raw)
    
#     with torch.no_grad():
#         raw_Z=raw_model(X_test.float())
#         trained_Z=model_trained(X_test.float())
#         Loss_raw,Loss_trained= 0.0 , 0.0

#         Loss_raw += torch.nn.functional.mse_loss(raw_Z,torch.unsqueeze(y_test,1)).item()
#         Loss_trained += torch.nn.functional.mse_loss(trained_Z,torch.unsqueeze(y_test,1)).item()

#     print(f'raw_model Loss-----> {Loss_raw}\n',
#           f'laoded_model Loss ----> {Loss_trained}')


# Eval(X_test,y_test)