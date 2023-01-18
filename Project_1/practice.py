import torch
import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split as split
from  torch import nn


from macd import  macd
X,y= macd.signal_macd()


# # Define the model architecture
class FinancialModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(FinancialModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 120) # 5 is the number of the columns
        self.fc2 = torch.nn.Linear(120, 64)
        self.fc3 = torch.nn.Linear(64,output_size)
        # self.fc4 = torch.nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        # x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc3(x)
        return x
# #dd
# import matplotlib.pyplot as plt



    # ---------------> Evaluation Section

    
# # Load and preprocess the data

# path=f'cryptoData/'
# df=pd.read_csv(f'{path}ADA-USD.csv',parse_dates=True,index_col='Date')
# data =df.astype(float) # Convert the data to float type
# data = (df -df.mean()) / df.std() # Normalize the data
# data = torch.tensor(data.values).float() # Create a tensor from the data


# # X_data
# data_X=np.delete(data,[-2],axis=1) # expect Adj_close
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




#Create the model and optimizer  ------>>>
# model = FinancialModel(input_size=X_train.shape[1], output_size=1)
# optimizer = torch.optim.Adam(model.parameters())

seq_los=nn.L1Loss() # Main absolute Error
cross_loss=nn.CrossEntropyLoss() 



# # for X_batch, y_batch in train_loader:
# #     pass
# #     print(X_batch.shape)




# # Runn the Evaluation
# obj=Train_eval()
# obj.Eval(X_test=X_test, y_test=y_test)

# # torch.save(model.state_dict(), 'raw_model.pt')

# # loaded_model=torch.load('my_model.pt')

# # # loaded_model = torch.load('models/my_model_full.pt')
# # print(f'\nloaded_model\n',loaded_model)
# # print(f'\nfirst model\n',model.state_dict())


print(len(X),len(y))
