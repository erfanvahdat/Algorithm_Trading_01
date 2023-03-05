# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
 

# img1 = cv.imread('Resources/Photos/cats.jpg',1)

# # cv.imshow('cat',img)
# # print(img.shape)

# b,g,r = cv.split(img1)  
# img_split = cv.merge((b,g,r))  # as the same of the img but mergin

# scale=1.0
# # dim=(int(img_split.shape[0]*scale/100),int(img_split.shape[1]*scale/90))

# # resized = cv.resize(img_split,(img_split.shape[1],img_split.shape[0]), interpolation=cv.INTER_AREA)   #dim (width,height)
# # print(resized.shape,img1.shape)

# h,w=img1.shape[:2]
# center = (int(w / 2),int( h / 2))




# # fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10, 20))
# # # ax(0,0)
# # edge=cv.Canny(img1,100,200)

# # ax1.imshow(edge)
# # ax2.imshow(img1)
# # plt.show()


# cap=cv.VideoCapture('Resources\Videos\kitten.mp4')


# while(cap.isOpened()):
#     _,frame=cap.read()
#     gray=cv.cvtColor(frame,cv.COLOR_BGRA2BGR)    
#     cv.imshow('f',gray)
#     if cv.waitKey(1) & 0xFF == ord('q'):  
#         break

# # while(cap.isOpened()):  
# #     ret,frame = cap.read()  
# # #it will open the camera in the grayscale mode  
# #     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  
# #     # gray=cv.cvtColor(frame,cv.COLOR_BGRA2BGR)    
# #     cv.imshow('frame',gray)  
# #     if cv.waitKey(1) & 0xFF == ord('q'):  
# #         break



# # Line and triangle and TEXT
# # tring=cv.rectangle(img1,pt1=center,pt2=(20,20),color=(20,255,2))
# # line=cv.line(img1,pt1=(10,0),pt2=center,color=(0,255,100),thickness=2)  
# # cv.putText(img1,"hi there its a cat",(100,150),cv.FONT_HERSHEY_DUPLEX,1,(20,0,200),2)
# # cv.imshow('g',line)



# # Circle
# # c_r=cv.circle(img1,center=center,radius=50,color=(50,255,500),thickness=-1)
# # cv.imshow('f',c_r)


# # Rotation
# # h,w=img1.shape[:2]
# # center = (w / 2, h / 2)  
# # M = cv.getRotationMatrix2D(center, 45,scale)  
# # rotated90 = cv.warpAffine(img1, M, (h, w))  
# # cv.imshow('f',M)



# # print(r.shape,g.shape,r.shape,img.shape) # shaping the RGB channel
# # cv.imshow('c',cv.cvtColor(img1, cv.COLOR_RGB2GRAY )) # turn the image to gray sacale

# """ Making Borders for Images
# replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)  
# reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)  
# reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)  
# wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)  
# constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)  
# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')  
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')  
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')  
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')  
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')  
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')  
# plt.show()   """

# # status=cv.imwrite(img=img,filename='Resources/catii.jpg',)
# # cv.imshow('cats',cv.imread('Resources/catii.jpg',0))



# # blank = np.zeros((500,500,3), dtype='uint8')
# # cv.imshow('Blank', blank)
# # print(blank)

# # # 1. Paint the image a certain colour
# # blank[200:300, 300:400] = 0,0,255
# # cv.imshow('Green', blank)

# # # 2. Draw a Rectangle
# # cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
# # cv.imshow('Rectangle', blank)

# # # 3. Draw A circle
# # cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1)
# # cv.imshow('Circle', blank)

# # # 4. Draw a line
# # cv.line(blank, (100,250), (300,400), (255,255,255), thickness=3)
# # cv.imshow('Line', blank)

# # # 5. Write text
# # cv.putText(blank, 'Hello, my name is Jason!!!', (0,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
# # cv.imshow('Text', blank)


# # Excute the image and then with clsoe it Destroy all windows
# cv.waitKey(0)
# cv.destroyAllWindows()  



# parth='/Desktop/Falling/none.png'
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
import os
import shutil

import torch

# path='C:/Users/Erfan/Desktop/Symetric_triangle_Copy'

# # image_files = [f for f in os.listdir(path) if f.endswith('.png')]
# new_filename='Falling_Wedge'


# for file  in (os.listdir(path)):
#     # current_filename=file
    
#     img=cv2.imread(os.path.join(path,file))
#     plt.imshow(img)
#     plt.show()
