'''
This module contains all utilities needed to load training data and the live diffraction pattern for denoising into pytorch 

Functions will be added in due course

All assumes that data is held in memory as numpy files currently

If necessary, we could investigate modifying this to holding data as cupy arrays if there is a suitable GPU with enough memory to hold these
'''

#Importing libraries
import torch
import numpy as np
import h5py

#Custom dataset object
class DataSet(torch.utils.data.Dataset):
    '''
    Makes a Dataset object for torch given a path to a dataset in a hdf5 file

    It implements Rx and Ry which are the largest pixel index along the vertical (Rx) and 
    horizontal (Ry) axes (which preserves the standard axis order of numpy, which is a 
    right-handed coordinate system, and uses the terminology of py4DSTEM).  Calling horizontal
    x and vertical y when the vertical axis is index downwards flips the axis system to a 
    left-handed system and is best avoided (although that is the choice in hyperspy)

    A range of selectors are available for selecting the pixels around the one of interest for 
    the video denoising

    getitem uses the chosen selector in defining the pixels chosen for return of diffraction
    patterns in surrounding pixels in item_input
    
    the original pixel diffraction pattern is returned in item_output

    If you want standard shaped item_input tensors, then only call inside a range where none of 
    the sampled pixels would lie outside the scan box.  For instance, using 5d, you can only run
    a model in the range [Rx+2:-2, Ry+2:-2]
    '''
    def __init__(self, file_path):
        '''
        a deliberate choice is made to just select one file to map from, so there is only a 0
        selected in imgs
        '''
        #file_paths here is a list of paths refers to a list of paths to files that are used as sources of data
        self.imgs=[]
        
        f = h5py.File(file_path, 'r')
        self.imgs.append(f['Experiments/__unnamed__/data/'])

    #Height and width
    def Rx(self):
        '''
        Size in real space in vertical direction
        '''
        return self.imgs[0].shape[0]

    def Ry(self):
        '''
        Size in real space in horizontal direction
        '''
        return self.imgs[0].shape[1]

    def selector(self, Rx_pos, Ry_pos, samplershape='5l'):
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
        assert samplershape in ['5l','3d','3s','5d'], 'Undefined Shape Code, please choose 5l, 3d, 3s or 5d'
        if samplershape == '5l':
            slicer = np.mgrid[0:1,-2:3]
            keep = np.ones_like(slicer).astype('bool')[0]
            keep[
                [0],
                [2]
            ] = False
            slicer = slicer[:,keep]
        elif samplershape == '3d' or '3s':
            slicer = np.mgrid[-1:2,-1:2]
            keep = np.ones_like(slicer).astype('bool')[0]
            if samplershape == '3d':
                keep[
                    [0,2,1,0,2],
                    [0,0,1,2,2]
                ] = False
            elif samplershape == '3s':
                keep[
                    1,
                    1
                ] = False
            slicer = slicer[:,keep]
        elif samplershape == '5d':
            slicer = np.mgrid[-2:3,-2:3]
            keep = np.ones_like(slicer).astype('bool')[0]
            keep[
                [0,1,3,4,0,4,2,0,4,0,1,3,4],
                [0,0,0,0,1,1,2,3,3,4,4,4,4]
            ] = False
            slicer = slicer[:,keep]
        
        # Shift the slicer to the chosen scan position
        shifted_slicer = (slicer.T+np.array([Rx_pos,Ry_pos])).T
        # Only keep selections that are within top and left boundaries
        keepTL = np.where(np.logical_and(shifted_slicer[0]>=0,shifted_slicer[1]>=0))[0]
        shifted_slicer_1 = shifted_slicer[:,keepTL]
        # Only keep selections that are inside the bottom and right boundaries
        keepBR = np.where(np.logical_and(shifted_slicer_1[0]<self.Rx(),shifted_slicer_1[1]<self.Ry()))[0]
        coord_list = (shifted_slicer_1[:,keepBR]).T
        
        return coord_list

    def getitem(self, index, samplershape='5l'):
        '''
        gets the real space positions to select from the input of index

        Parameters
        ----------
        index: list of ints
            [Rx_pos, Ry_pos]
        samplershape: str
            Described in selector (above)

        Returns
        -------
        item_input:
            input to ML model, which will be a tensor of shape (n,Qx,Qy), where Qx and Qy
            are the sizes of the data in the diffraction directions vertically and horizontally.
            n will depend on samplershape and on the position in the scan
        item_output:
            currently just returns the diffraction pattern at the pixel at the index point
            as tensor of shape (Qx,Qy)
        '''
        Rx_pos, Ry_pos=index
        coord_list = self.selector(Rx_pos, Ry_pos, samplershape)
        item_output=torch.tensor(self.imgs[0][Rx_pos, Ry_pos],dtype = torch.float64)
        item_input=[]
        for coords in coord_list:
            item_input.append(self.imgs[0][coords[0],coords[1]])
        item_input=torch.tensor(item_input,dtype = torch.float64)

        return item_input, item_output
        
    def __getitem__(self,index):
        return self.getitem(index)