# parth='/Desktop/Falling/none.png'
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
import os
import shutil
from distutils.dir_util import copy_tree
from torch.utils.data import Dataset, TensorDataset
import torch
import torch.nn.functional as F         
import torchvision.transforms as transforms
from PIL import Image




"""
    os.lisdr(parh) # full element in the dictionalry
    or ->  [f for f in os.listdir(path) if f.endswith('.png')]

        if we want to join something from the previus path:
    # os.rename(os.path.join(path, file), os.path.join(path, new_filename))
    # shutil.move(os.path.join(path,file),os.path.join(path,new_filename ))

    # for copy the folder or file
    copy_tree(path,new_path )
    """



def Image_opt():
    # copy_path="C:/Users/Erfan/Desktop/Symetric triangle/copy"
    # path="C:/Users/Erfan/Desktop/Symetric triangle/"
    # path="A:\Algorithm_Trading_01\DATA\copy_Train\Rising_Wedge/"
    # path="A:\Algorithm_Trading_01\DATA\Train\Rising_Wedge/"

    # path="A:\Algorithm_Trading_01\copy_Train/"
    path="C:/Users/Erfan/Desktop/new_dataset/"
    # simple_path="A:\Algorithm_Trading_01\copy_Train\Rising_Wedge"

    classes=os.listdir(path)
    
    # Copy the original file
    # copy_tree(path,copy_path)
    transform = transforms.Compose([transforms.Resize((500,900)),  # Resize image
            transforms.ToTensor()  # Convert image to PyTorch tensor 
                    ])
    
    
    for class_name in classes:
        for index,file in enumerate(os.listdir(os.path.join(path,class_name))):
            try:
                os.rename(os.path.join(path,class_name,file),
                      os.path.join(path,class_name,f"{class_name}_{index}.png"))
                
            except FileExistsError:
            # If the file already exists, ignore the error and move on
                pass

    
        # if index== 1:        
            # return shutil.move(os.path.join(path,file),'C:/Users/Erfan/Desktop/')
        

            # Tensor_image resiezed
#             tensor_image_resized=transform(open_image)
            # Turn the Tensor to Image
#             tensor_rgba = transforms.ToPILImage(mode='RGBA')(tensor_image_resized)

#             tensor_rgba.save(os.path.join(path, file))

# Image_opt()


print('hi')


#########################Tesing tinygrad#############################

# from   tinygrad.tensor  import Tensor
# import tinygrad.nn.optim as optim
# import torch

# class TinyBobNet:
#   def __init__(self):
#     self.l1 = Tensor.uniform(784, 128)
#     self.l2 = Tensor.uniform(128, 10)

#   def forward(self, x):
#     return x.dot(self.l1).relu().dot(self.l2).log_softmax()

# model = TinyBobNet()
# optim = optim.SGD([model.l1, model.l2], lr=0.001)

# # ... and complete like pytorch, with (x,y) data

# y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)

# out = model.forward(y)
# loss = out.mul(y).mean()
# optim.zero_grad()
# loss.backward()
# optim.step()
###############################################################
# ---------------------------------------------------------------------------------------------------------------------------------------
# optimizer = optim.Adam(model.parameters(), lr=0.02)
# input is of size N x C = 3 x 5
# torch.manual_seed(42)
# Epochs=1000
# for epoch in range(Epochs):
#     input = torch.randn(3, 5, requires_grad=True)
#     # each element in target has to have 0 <= value < C
#     target = torch.tensor([1, 0, 4])
#     output = F.nll_loss(F.log_softmax(input, dim=1), target)
#     output.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    # print(output)
    # print(target,input,torch.argmax(input,dim=1))

# ---------------------------------------------------------------------------------------------------------------------------------------

#