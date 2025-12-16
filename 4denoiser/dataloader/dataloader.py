import numpy as np

'''
This module contains all utilities needed to load training data and the live diffraction pattern for denoising into pytorch 

Functions will be added in due course

All assumes that data is held in memory as numpy files currently

If necessary, we could investigate modifying this to holding data as cupy arrays if there is a suitable GPU with enough memory to hold these
'''

#Importing libraries
import hyperspy.api as hs
import torch
import matplotlib.pyplot as plt

#Settings the inference area i.e. the area that will be uses as input to the NN
# 2 before, 2 after
inference_H=1
inference_W=5
x_offset=2
y_offset=0
"""
#For surrounding 8 instead
inference_H=3
inference_W=3
x_offset=1
y_offset=1
"""
#Coords of input pixels relative to output pixel. In [y,x] (numpy) order
input_coords=[]
for y in range(inference_H):
    for x in range(inference_W):
        coords=[x-x_offset,y-y_offset]
        if coords[0]!=0 or coords[1]!=0: #i.e. coords is not [0,0]
            input_coords.append(coords)

#Custom dataset object
class DataSet(torch.utils.data.Dataset):
    def __init__(self, file_paths):
        #file_paths here is a list of paths refers to a list of paths to files that are used as sources of data
        self.imgs=[]
        for file_path in file_paths:
            self.imgs.append(hs.load(file_path, reader="hspy"))

    #Height and width
    def img_H(self,img_index):
        return self.imgs[img_index].data.shape[2]

    def img_W(self,img_index):
        return self.imgs[img_index].data.shape[3]

    def index_location(self, index): #FInds a location in i, y, x (i being the img_index) of pixel number index
        if index>self.__len__():
            raise ValueError("Index too high")

        running_total=0
        for img_index in range(len(self.imgs)):
            new_running_total=running_total+((self.img_H(img_index)+1-inference_H)*(self.img_W(img_index)+1-inference_W))
            if index<new_running_total: #It's in this image
                difference=index-running_total
                x_pos=difference%self.img_W(img_index)
                y_pos=difference//self.img_W(img_index)
                return img_index, y_pos, x_pos
            else:
                running_total=new_running_total

    def __len__(self):
        running_total=0
        for img_index in range(len(self.imgs)):
            running_total+=(self.img_H(img_index)+1-inference_H)*(self.img_W(img_index)+1-inference_W)
        return running_total
    
    #Function that returns input/output pair 
    def getitem(self, index):

        img_index, y_pos, x_pos=self.index_location(index)
        
        item_output=torch.tensor(self.imgs[img_index].data[y_pos,x_pos],dtype = torch.float64)
        #For the inputs, we collect diffraction pattern from all input pixels and use the one from each pixel as a channel
        item_input=[]
        for coords in input_coords:
            item_input.append(self.imgs[img_index].data[y_pos+coords[0],x_pos+coords[1]])
        item_input=torch.tensor(item_input,dtype = torch.float64)
        
        return item_input,item_output
    
    def __getitem__(self,index):
        return self.getitem(index)
    
