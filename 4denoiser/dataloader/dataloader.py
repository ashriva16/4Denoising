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

# More complex selector, but assumes we are selecting from numpy and delivering to torch in small batches.  This sounds more appropriate
    # to the live denoising problem, where we only have a partial dataset at any one time.  Of course, to use shapes like square or diamond,
    # you need to be displaying a line or two behind live.  You could for live purposes also do a triangle of left, up and right.    

def selector(dataset, Rx, Ry, shape='5l'):
    '''
    Makes a selection of suitable points for video averaging from a dataset being recorded, 
    and works with a few different shapes
    
    Parameters
    ----------
    dataset: np.ndarray
        a 4-dimensional STEM dataset, with axes in order Rx, Ry, Qx, Qy (same as py4DSTEM)
    Rx: int
        index along the Rx direction (vertical down)
    Ry: int
        index along the Ry direction
    shape: str
        A predefined str for the shape of area to extract the patch from for video denoising
        Currently supported shapes:
            5l: a line 5 long along the horizontal direction, skipping the centre: xxoxx
            3d: a diamond 3 wide: oxo
                                  xox
                                  oxo
            3s: a square 3 wide:  xxx
                                  xox
                                  xxx
            5d: a diamond 5 wide: ooxoo
                                  oxxxo
                                  xxoxx
                                  oxxxo
                                  ooxoo
    
    Returns
    -------
    DPs: np.ndarray
        A 3D array of dimensions (n,Qx,Qy), n is the number of diffraction patterns returned for video denoising
    
    '''
    RQshape = dataset.shape
    assert shape in ['5l','3d','3s','5d'], 'Undefined Shape Code, please choose 5l, 3d, 3s or 5d'
    if shape == '5l':
        slicer = np.mgrid[0:1,-2:3]
        keep = np.ones_like(slicer).astype('bool')[0]
        keep[
            [0],
            [2]
        ] = False
        slicer = slicer[:,keep]
    elif shape == '3d' or '3s':
        slicer = np.mgrid[-1:2,-1:2]
        keep = np.ones_like(slicer).astype('bool')[0]
        if shape == '3d':
            keep[
                [0,2,1,0,2],
                [0,0,1,2,2]
            ] = False
        elif shape == '3s':
            keep[
                1,
                1
            ] = False
        slicer = slicer[:,keep]
    elif shape == '5d':
        slicer = np.mgrid[-2:3,-2:3]
        keep = np.ones_like(slicer).astype('bool')[0]
        keep[
            [0,1,3,4,0,4,2,0,4,0,1,3,4],
            [0,0,0,0,1,1,2,3,3,4,4,4,4]
        ] = False
        slicer = slicer[:,keep]
    
    # Shift the slicer to the chosen scan position
    shifted_slicer = (slicer.T+np.array([Rx,Ry])).T
    # Only keep selections that are within top and left boundaries
    keepTL = np.where(np.logical_and(shifted_slicer[0]>=0,shifted_slicer[1]>=0))[0]
    shifted_slicer_1 = shifted_slicer[:,keepTL]
    # Only keep selections that are inside the bottom and right boundaries
    keepBR = np.where(np.logical_and(shifted_slicer_1[0]<RQshape[0],shifted_slicer_1[1]<RQshape[1]))[0]
    shifted_slicer_2 = shifted_slicer_1[:,keepBR]
    
    # Now make the slice
    DPs = data_for_slicing[shifted_slicer_2[0],shifted_slicer_2[1]]
    
    return DPs
    


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
        # surely you are just loading one single filepath
        self.imgs=[]
        for file_path in file_paths:
            self.imgs.append(hs.load(file_path, reader="hspy"))
            # I recommend we simply load in my numpy object here, and we previously convert the hs dataset to numpy prior to even calling this

    #Height and width
    def img_H(self,img_index):
        return self.imgs[img_index].data.shape[2]
        # would be DPs.shape[1] in my object

    def img_W(self,img_index):
        return self.imgs[img_index].data.shape[3]
        # would be DPs.shape[2] in my object

    def index_location(self, index): #FInds a location in i, y, x (i being the img_index) of pixel number index
        # surely use by # would be DPs.shape[0] in my object 
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
            # this all sounds unnecesary.  We no longer care where the images came from in x and y, that was already sorted in the selector
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

