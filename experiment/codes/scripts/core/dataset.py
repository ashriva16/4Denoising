import hyperspy.api as hs
import numpy as np
import py4DSTEM
import torch
from torch.utils.data import Dataset


class STEMDataset(Dataset):
    """
    data_4d: (X, Y, Kx, Ky)
    Output: clip of shape (T, 1, Kx, Ky)
    """
    def __init__(self, filepath, T: int = 5):

        s = hs.load(filepath, reader='HSPY')
        datacube = py4DSTEM.DataCube(s.data)
        self.data_4d: np.ndarray = datacube.data
        self.D = self.data_4d
        self.X, self.Y, self.Kx, self.Ky = self.data_4d.shape
        self.T = T
        self.half = T // 2

        # flatten (x,y) -> z
        self.Dz = self.data_4d.reshape(self.X * self.Y, self.Kx, self.Ky)

        # valid centers
        self.valid_z = range(self.half, self.X * self.Y - self.half)

    def __len__(self):
        return len(self.valid_z)

    def __getitem__(self, idx):
        z = self.valid_z[idx]
        clip_np = self.Dz[z - self.half : z + self.half + 1]  # (T, Kx, Ky)
        clip = torch.from_numpy(clip_np).float().unsqueeze(1)  # (T,1,Kx,Ky)
        return clip, z
