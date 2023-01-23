import torch
# import gym
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sn
# from sklearn.model_selection import train_test_split as split
# from  torch import nn



from macd import  macd
X,y= macd.signal_macd()
print(X[0].shape)

# # print(torch.concat((torch.tensor(X[1]),torch.tensor(X[0]))).shape)

# # print(y.unique())

# # # Define the model architecture
# class FinancialModel(torch.nn.Module):
#     def __init__(self, input_size, output_size):
#         super(FinancialModel, self).__init__()
#         # self.fc1 = torch.nn.Linear(input_size, 120) # 5 is the number of the columns
#         # self.fc2 = torch.nn.Linear(120, 2)
#         # self.fc3 = torch.nn.Linear(64,1)
#         # self.fc4 = torch.nn.Linear(64, output_size)
#         self.fc1 = nn.Linear(6, 60)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(60,2)
#         self.softmax = nn.Softmax(dim=1)


#     def forward(self, x):
#         # x = torch.nn.functional.relu(self.fc1(x))
#         # x = (self.fc2(x))
#         # x = (self.fc3(x))
        
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.softmax(out)
#         return out
        
# # #dd
# # import matplotlib.pyplot as plt



#     # ---------------> Evaluation Section

    
# # # Load and preprocess the data
# # path=f'cryptoData/'
# # df=pd.read_csv(f'{path}ADA-USD.csv',parse_dates=True,index_col='Date')
# # data =df.astype(float) # Convert the data to float type
# # data = (df -df.mean()) / df.std() # Normalize the data
# # data = torch.tensor(data.values).float() # Create a tensor from the data
# # # X_data
# # data_X=np.delete(data,[-2],axis=1) # expect Adj_close
# # # Y_data
# # data_y=data[:,-2] # Adj_close
# # X_train,X_test,y_train,y_test= split(data_X,data_y,test_size=0.8,shuffle=False)
# # # Define the bach size
# # batch_size = 32
# # num_batches = len(X_train) // batch_size
# # from torch.utils.data import TensorDataset, DataLoader

# # create TensorDataset
# from  torch.utils.data import TensorDataset,DataLoader
# X_train,X_test,y_train,y_test=split(X,y,test_size=0.3,shuffle=True,random_state=42)
# # train_data = TensorDataset(X_train, y_train)
# # create DataLoader
# # batch_size=10
# # train_loader = DataLoader(train_data, batch_size=batch_size) # shuffle=True


# #Create the model and optimizer  ------>>>
# model = FinancialModel(input_size=X_train[0].shape[1], output_size=2)
# optimizer = torch.optim.Adam(model.parameters())

# seq_loss=nn.L1Loss() # Main absolute Error
# cross_loss=nn.CrossEntropyLoss() 



# # for X_batch, y_batch in train_loader:
# #     pass
# #     print(X_batch.shape)


# # Runn the Evaluation
# EPOCH=10

# # print(y_train[30])


# # # print((torch.tensor([1]).float()).dtype)

# # for i in  tqdm(range(EPOCH)):
# #     for i in range(len(X_train)): # this is because we have a unshape structure data. need to pass all the tensor to model manually...
# #         outcome=model(X_train[i].float())
# #         print(outcome)


# #         optimizer.zero_grad()
# #         # loss.backward()
# #         optimizer.step()

# #         # print(outcome.dim())
# # #         print(outcome)
# #         break
        
#         # print(torch.softmax(outcome),dim)
#         # output = cross_loss(outcome, y_train[i])

# #     output.backward()
#         # print('input: ', X_train[i])
#         # print('target: ', y_train[i])
#         # print('output: ', output)
#         # break
        


# # torch.save(model.state_dict(), 'raw_model.pt')

# # loaded_model=torch.load('my_model.pt')

# # # loaded_model = torch.load('models/my_model_full.pt')
# # print(f'\nloaded_model\n',loaded_model)
# # print(f'\nfirst model\n',model.state_dict())


# # Runner

