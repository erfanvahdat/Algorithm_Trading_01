import torch
import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split as split
from  torch import nn
import netron
from torchviz import make_dot


from macd import  macd
X,y= macd.signal_macd()

from sklearn import preprocessing
norm=preprocessing.Normalizer()


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


X_train,X_test,y_train,y_test=split(X,y,test_size=(0.2),shuffle=True,random_state=42)
    
    

class FinancialModel(torch.nn.Module):
    def __init__(self):
        super(FinancialModel, self).__init__()
        self.fc1 = nn.Linear(6, 80) 
        self.relu=nn.ReLU()
        self.tahn=nn.Tanh()
        self.sigmoid=nn.Sigmoid()

        self.fc2=nn.Linear(80,100)
        # self.fc3=nn.Linear(100,400)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        # x = self.flatten(x)

        x = self.fc1(x)
        # x=self.sigmoid(x)
        x=self.relu(x)
        x=self.sigmoid(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        return x

# #Create the model and optimizer  ------>>>
# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
# loss_fn=nn.BCEWithLogitsLoss() # buil-in sigmoid funciton


# One Apporach ---->>>>>

X_train_norm=[torch.nn.functional.normalize(torch.Tensor(X_train[indexx]),dim=0,p=2).float() for indexx in range(len(X_train))]

# input size of the matrix



# Define the Model
model = FinancialModel()
optimizer = torch.optim.Adam(model.parameters())



input_sizes = [torch.flatten(torch.Tensor(A)).shape[0] for A in X_train] # for X_train
evalue=[]

EPOCH=200      
# Training Loop ==========>>>>
for qw in range(EPOCH):
 
    for indexx  in range(len(X_train)):
        
        new_fc1 = torch.nn.Linear(input_sizes[indexx], 80)

        # Get the weights and biases of the old fc1 layer (original model)
        # old_fc1_weights = model.fc1.weight.data
        # old_fc1_biases = model.fc1.bias.data

        # # Assign the weights and biases from the old fc1 layer to the new fc1 layer
        # new_fc1.weight.data = old_fc1_weights
        # new_fc1.bias.data = old_fc1_biases
        model.fc1=new_fc1
        
        # Replace the old fc1 layer in the model with the new fc1 layer
        
        # torch.save(model.state_dict(),'model.pth')
        y_logits = model(torch.flatten(X_train_norm[indexx])) # one value of our model # not changed
        
        
        evalue.append(y_logits)
    

    cc=torch.vstack([torch.tensor(evalue, requires_grad=True),torch.Tensor(y_train)]).T

    # loss function

    # we dont ned to pass y_Pred to sigmoid funciton becasue our loss_fn has build-in sigmoid funciton.
    loss=loss_fn(torch.sigmoid(cc[:,0]),cc[:,1]) # no need torch.sigmoid

    y_pred = torch.round(torch.sigmoid(cc[:,0])) # turn logits -> pred probs -> pred labls

    acc = accuracy_fn(y_true=torch.tensor(y_train,dtype=torch.int32),
    y_pred=y_pred ) 

    # print(len(torch.tensor(y_train,dtype=torch.int32)),len(y_pred))

    optimizer.zero_grad()
    # 4. Loss backwards
    loss.backward()

    optimizer.step()

    # saver_evalue_total.append(cc[:,0])
    # # clear the evlaue for the next enviroment to calcualte next loss function
    evalue.clear()

    if qw %10==0:
        print(f"Epoch: {qw} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% " )
    




    # ### Testing
    
        # with torch.inference_mode():
        #     # 1. Forward pass
        #     for i in range(len(X_test)):
        #         test_logits = model(X_test[i].float()).squeeze() 
        #         test_pred = torch.round(torch.sigmoid(test_logits))
        #     #     # 2. Caculate loss/accuracy
        #         test_loss = loss_fnlogic(test_logits,
        #                             y_test[i])

        #         test_acc = accuracy_fn(y_true=y_test[indexi],
        #                                 y_pred=test_pred)

        # # Print out what's happening every 10 epochs
        # if indexi % 10 == 0:
            # print(f"Epoch: {indexi} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
                # print(f"Epoch: {EPOCH} | Loss: {loss:.5f}, Accuracy: {acc:.2f}%" )
        


