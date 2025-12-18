#Custom DataSet

'''
This module contains all utilities needed to load training data and the live diffraction pattern for denoising into pytorch

Functions will be added in due course

All assumes that data is held in memory as numpy files currently

If necessary, we could investigate modifying this to holding data as cupy arrays if there is a suitable GPU with enough memory to hold these
'''

#Importing libraries
import h5py
import numpy as np
import torch


class STEMDataSet(torch.utils.data.Dataset):
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
    '''
    def __init__(self, file_path, samplershape="5l"):
        '''
        A deliberate choice is made to just select one file to map from, so self.imgs only has one item.
        I would recommend upgrading this software so it can handle multiple "images" (actually 4D datasets)

        Parameters
        ----------
        filepath: string
            file path of the single 4D STEM data file to be loaded

        samplershape: string
            See self.selector().
            DO NOT MOVE THIS TO getitem OR THIS WILL BREAK
        '''
        self.imgs=[]
        f = h5py.File(file_path, 'r')
        self.imgs.append(np.array(f['Experiments/__unnamed__/data/']))

        self.samplershape=samplershape

        if self.samplershape == '5l':
            self.top_exclude=0
            self.bottom_exclude=0
            self.left_exclude=2
            self.right_exclude=2
            self.Rx=self.imgs[0].shape[0]
            self.Ry=self.imgs[0].shape[1]
            self.n=5
        elif self.samplershape == '3d' or '3s':
            self.top_exclude=1
            self.bottom_exclude=1
            self.left_exclude=1
            self.right_exclude=1
        if self.samplershape == '3d':
            self.n=5
        if self.samplershape == '3s':
            self.n=5
        self.Rx=self.imgs[0].shape[0]
        self.Ry=self.imgs[0].shape[1]
        self.Rx_cut=self.imgs[0].shape[0]-self.top_exclude-self.bottom_exclude
        self.Ry_cut=self.imgs[0].shape[1]-self.left_exclude-self.right_exclude

    def __len__(self):
        """
        This is just the name of usubale pixels in the downsamples frame.
        If this software is changed to multi-image, we will have to change this
        """
        return (self.imgs[0].shape[0]-self.top_exclude-self.bottom_exclude)*(self.imgs[0].shape[1]-self.left_exclude-self.right_exclude)

    #Note that this uses the sane person co-ordinate convention (x,y), while the rest of the program uses numpy convention (y,x), with weird results (e.g. the line for 5l is actually in the y-direction. We should probably fix this.)
    def selector(self, Rx_pos, Ry_pos):
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
                5l: a line 5 long along the horizontal direction: xxxxx
                3d: a diamond 3 wide: oxo
                                      xxx
                                      oxo
                3s: a square 3 wide:  xxx
                                      xxx
                                      xxx

        Returns
        -------
        DPs: np.ndarray
            A 3D array of dimensions (n,Qx,Qy), n is the number of diffraction patterns returned for video denoising

        '''
        samplershape=self.samplershape
        assert samplershape in ['5l','3d','3s',], 'Undefined Shape Code, please choose 5l, 3d, 3s'
                #Because of the sampler, we have to downsample, cannot train with all pixels of the image, we have to exclude some at the edges

        if samplershape == '5l':
            slicer = np.mgrid[0:1,-2:3]
            keep = np.ones_like(slicer).astype('bool')[0]
            slicer = slicer[:,keep]

        elif samplershape in ('3d', '3s'):
            slicer = np.mgrid[-1:2,-1:2]
            keep = np.ones_like(slicer).astype('bool')[0]
            if samplershape == '3d':
                keep[
                    [0,2,0,2],
                    [0,0,2,2]
                ] = False
            elif samplershape == '3s':
                pass
            slicer = slicer[:,keep]

        # Shift the slicer to the chosen scan position
        shifted_slicer = (slicer.T+np.array([Rx_pos,Ry_pos])).T
        # Only keep selections that are within top and left boundaries
        keepTL = np.where(np.logical_and(shifted_slicer[0]>=0,shifted_slicer[1]>=0))[0]
        shifted_slicer_1 = shifted_slicer[:,keepTL]
        # Only keep selections that are inside the bottom and right boundaries
        keepBR = np.where(np.logical_and(shifted_slicer_1[0]<self.Rx,shifted_slicer_1[1]<self.Ry))[0]
        coord_list = (shifted_slicer_1[:,keepBR]).T

        return coord_list

    def index_to_RxRy(self, index: int):
        Rx_pos = int(index / self.Ry_cut) + self.top_exclude
        Ry_pos = index % self.Ry_cut + self.left_exclude
        # print(f"Scannig location (x,y): {Rx_pos, Ry_pos} for index {index}")
        return Rx_pos, Ry_pos

    def getitem(self, index):
        '''
        gets the real space positions to select from the input of index

        Parameters
        ----------
        index: int
            Index corresponding to a usuable pixel on the downsampled image, starting at lowest
            x and y and increasing x first, then y.
        samplershape: str
            Described in selector (above)

        Returns
        -------
        item_input:
            input to ML model, which will be a tensor of shape (n,Qx,Qy), where Qx and Qy
            are the sizes of the data in the diffraction directions vertically and horizontally.
            n will depend on samplershape and on the position in the scan.
        item_output:
            currently just returns the diffraction pattern at the pixel at the index point
            as tensor of shape (Qx,Qy)
        '''
        samplershape=self.samplershape

        maxindex = self.Ry*self.Rx

        assert index<maxindex, 'index out of range'

        self.samplershape=samplershape

        Rx_pos, Ry_pos = self.index_to_RxRy(index)
        coord_list = self.selector(Rx_pos, Ry_pos)

        item_output = torch.from_numpy(self.imgs[0][Rx_pos, Ry_pos]).to(torch.float32) / 300

        arr = np.stack([self.imgs[0][c[0], c[1]] for c in coord_list], axis=0)
        item_input = torch.from_numpy(arr).to(torch.float32) / 300

        if item_input.shape[0]!=self.n:
            raise ValueError("Incorrect number of channels. This is probably because, somehow," \
            "values of Rx_pos and Ry_pos have been chosen that are near the edge of the" \
            "image, so the samples is off the edge of the image and cannot sample fully.")

        return item_input, item_output

    def __getitem__(self,index):
        return self.getitem(index)
