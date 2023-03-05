import matplotlib.pyplot as mtp  
import pandas as pd
import numpy as np
import torch 
from torch import nn
from sklearn.model_selection import train_test_split as split
# Normalizaiton
# torch.nn.functional.normalize
from  sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression as LR
# norm=Normalizer()




data_set=pd.read_csv('Salary_Data.csv')

x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 1].values   
x_train, x_test, y_train, y_test= split(x, y, test_size= 0.4, random_state=0)  

X_torch=torch.Tensor(x_train)
y_torch=torch.Tensor(y_train)

regress=LR()
regress.fit(X_torch,y_torch)
y_pred= regress.predict(x_test)

# x_pred=regress.predict(X_torch)  
# mtp.scatter(x_train, y_train, color="green")   
# mtp.plot(x_train, x_pred, color="red")    
# mtp.title("Salary vs Experience (Training Dataset)")  
# mtp.xlabel("Years of Experience")  
# mtp.ylabel("Salary(In Rupees)")  
# mtp.show()   

# #visualizing the Test set results  
# mtp.scatter(x_test, y_test, color="blue")   
# mtp.plot(x_train, x_pred, color="red")    
# mtp.title("Salary vs Experience (Test Dataset)")  
# mtp.xlabel("Years of Experience")  
# mtp.ylabel("Salary(In Rupees)")  
# mtp.show()


from  macd import macd

X,y=macd.signal_macd()

print()


# mtp.imshow(X[5])
# mtp.show()
# regress.fit()
