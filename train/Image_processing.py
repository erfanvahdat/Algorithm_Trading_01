# parth='/Desktop/Falling/none.png'
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
import os
import shutil
from distutils.dir_util import copy_tree
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
    path="/Algorithm_Trading_01/DATA/Rising_Wedge//"
    # Copy the original file
    # copy_tree(path,copy_path)
    transform = transforms.Compose([transforms.Resize((500,900)),  # Resize image
            transforms.ToTensor()  # Convert image to PyTorch tensor 
                    ])
    
    for index,file in enumerate(os.listdir(path)):
        if file.endswith('.png') or file.endswith('.jpg'):
            open_image=Image.open(os.path.join(path,file))
            print(open_image)

            # Tensor_image resiezed
#             tensor_image_resized=transform(open_image)
            # Turn the Tensor to Image
#             tensor_rgba = transforms.ToPILImage(mode='RGBA')(tensor_image_resized)

#             tensor_rgba.save(os.path.join(path, file))

Image_opt()