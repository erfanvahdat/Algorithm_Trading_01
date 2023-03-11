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


# a=torch.rand(45,4)
# print(a)
# Data Generation
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)


# Uses `fi`rst 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# plt.scatter(x_train,y_train)
# plt.show()
#  Initializes parameters "a" and "b" randomly
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)



# print(a, b)

# Sets learning rate
lr = 1e-1
# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Computes our model's predicted output
    yhat = a + b * x_train
    
    # How wrong is our model? That's the error! 
    error = (y_train - yhat)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()
    
    # Computes gradients for both "a" and "b" parameters
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()
    
    # Updates parameters using gradients and the learning rate
    a = a - lr * a_grad
    b = b - lr * b_grad
    
# print(a, b)

# Sanity Check: do we get the same results as our gradient descent?
# from sklearn.linear_model import LinearRegression
# linr = LinearRegression()
# linr.fit(x_train, y_train)
# print(linr.intercept_, linr.coef_[0])





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# # and then we send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)


# # Here we can see the difference - notice that .type() is more useful
# # since it also tells us WHERE the tensor is (device)
# print(type(x_train), type(x_train_tensor), x_train_tensor.type())




# FIRST
# Initializes parameters "a" and "b" randomly, ALMOST as we did in Numpy
# since we want to apply gradient descent on these parameters, we need
# to set REQUIRES_GRAD = TRUE
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
# print(a, b) #-----> show

# SECOND
# But what if we want to run it on a GPU? We could just send them to device, right?
a = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
b = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)
# print(a, b) -----> show
# Sorry, but NO! The to(device) "shadows" the gradient...

# THIRD
# We can either create regular tensors and send them to the device (as we did with our data)
a = torch.randn(1, dtype=torch.float).to(device)
b = torch.randn(1, dtype=torch.float).to(device)
# and THEN set them as requiring gradients...
a.requires_grad_()
b.requires_grad_()
# print("\n first set the device then apply for gradient",a,b) ----->> show

# better apporach ot write them all in one line ----->>>>>>
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)


lr = 1e-1
n_epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)


for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    # print(error,"loss::",loss)
    

    # No more manual computation of gradients! 
    # a_grad = -2 * error.mean()
    # b_grad = -2 * (x_tensor * error).mean()
    
    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()
    # Let's check the computed gradients...
    # print(a.grad)
    # print(b.grad)
    
    # What about UPDATING the parameters? Not so fast...
    
    # FIRST ATTEMPT
    # AttributeError: 'NoneType' object has no attribute 'zero_'
    # a = a - lr * a.grad
    # b = b - lr * b.grad
    # print(a)

    # SECOND ATTEMPT
    # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
    # a -= lr * a.grad
    # b -= lr * b.grad        
    
    # THIRD ATTEMPT
    # We need to use NO_GRAD to keep the update out of the gradient computation
    # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
    
    # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
    a.grad.zero_()
    b.grad.zero_()
    
### put it all togehter....!!!
# THe smae------------------------------------------------------------------------------------------------------------

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

# print(a, b)
loss_fn = nn.MSELoss(reduction='mean') # now defining the loss funciton

x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()


lr = 1e-1
n_epochs = 1000

class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.layer2 = nn.Linear(1, 50)
        self.layer3 = nn.Linear(50, 1)

        
    def forward(self, X):
        # Computes the outputs / predictions
        # return self.a + self.b * x
        x=self.layer2(X)
        x=self.layer3(x)
        # return self.linear2(self.Linear(X))
        return x



model = ManualLinearRegression().to(device)
# We can also inspect its parameters using its state_dict
# print(model.state_dict())

# Defines a SGD optimizer to update the parameters
# optimizer = optim.SGD(model.state_dict(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for epoch in range(n_epochs):
    # No more manual prediction!
    # yhat = a + b * x_tensor
    yhat = model(x_train_tensor)
    # yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat

    # loss = (error ** 2).mean() # delete this to add the next line
    loss = loss_fn(y_train_tensor, yhat)


    loss.backward()    

    # No more manual update!
    # with torch.no_grad():
    #     a -= lr * a.grad
    #     b -= lr * b.grad
    optimizer.step()
    
    # No more telling PyTorch to let gradients go!
    # a.grad.zero_()
    # b.grad.zero_()
    optimizer.zero_grad()
    
    print(loss)

# print(a, b)

# Put all together------------------------------------------------------------------------------------------------------
# fx = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)


x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.linear = nn.Linear(1, 1)

        
    def forward(self, x):
        # Computes the outputs / predictions
        # return self.a + self.b * x
        return self.linear(x)


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









