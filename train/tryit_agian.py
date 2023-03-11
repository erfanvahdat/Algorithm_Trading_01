import os
import torch
import pandas
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import transforms
# from torch.nn.functional import F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset



x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)


x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.layer = nn.Linear(1, 1)
        # self.layer2= nn.Linear(50, 1)

        
    def forward(self, x):
        # Computes the outputs / predictions
        # return self.a + self.b * x
        # return self.layer2(self.layer(x))
        return self.layer(x)


device= 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = nn.MSELoss(reduction='mean') # now defining the loss funciton


lr = 0.1
n_epochs = 100
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

model = ManualLinearRegression().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# optimizer=optim.SGD(model.state_dict() )



dataset = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [80, 20])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(X_tensor, y_tensor):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(X_tensor)
        # Computes loss
        loss = loss_fn(y_tensor, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step # ---> return the loss.item.....

# Creates the train_step function for our model, loss function and optimizer
losses = []
val_losses = []
avg_loss_list = []
loss_count=0

batch_counter=0
train_step = make_train_step(model, loss_fn, optimizer)
a=0

# Engine of leanring --------> 1
for epoch in  range(n_epochs):
    
    for ep,( x_batch, y_batch) in enumerate(train_loader):
        y_batch = y_batch.to(device)

        # Calculating the loss
        loss = train_step(x_tensor, y_tensor)
        loss_count+=loss
        losses.append(loss)
        batch_counter+=1

    # print(loss, loss_count/batch_counter)
    
    loss_count=0
    batch_counter=0
    
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            model.eval()

            yhat = model(x_val)
            val_loss = loss_fn(y_val, yhat)
            val_losses.append(val_loss.item())

    # print(f"loss: {loss}, eval_loss: {val_loss}")



