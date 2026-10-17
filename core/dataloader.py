import numpy as np

'''
Dataloader for UDVD-MF 4D-STEM denoising.

Returns 9-channel output in 3x3 grid order for ALL model variants:

    Spatial mode (3x3 grid):          Temporal mode (double-arrow):
      0:(rx-1,ry-1) 1:(rx-1,ry) 2:(rx-1,ry+1)     0:(t-1,up) 1:(t-2)   2:(t-1,dn)
      3:(rx,  ry-1) 4: CENTER   5:(rx,  ry+1)     3:(t-1)    4:CENTER  5:(t+1)
      6:(rx+1,ry-1) 7:(rx+1,ry) 8:(rx+1,ry+1)     6:(t+1,up) 7:(t+2)   8:(t+1,dn)

    Center is always at index 4 → center_f=4 in denoise_mf.py.

    Model groups (all pass through center):
      Group [0, 4, 8] — diagonal  (spatial) / diagonal  (temporal)
      Group [1, 4, 7] — vertical  (spatial) / extended temporal
      Group [2, 4, 6] — anti-diag (spatial) / anti-diagonal (temporal)
      Group [3, 4, 5] — horizontal(spatial) / direct temporal

This format works with:
  - blind-video-net-4d-cross  (4 groups, shift,  1px blind spot)
  - blind-video-net-5d-cross  (4 groups, shift2, 2px blind spot)

The double-arrow temporal geometry keeps all neighbors within ±2 raster
positions of the center, avoiding the physical-distance problem of using
4 positions before and 4 after in a linear sequence.

Margin = 2 is required for both modes (spatial needs ±1, temporal needs ±2).
'''

import torch
import os
import h5py


# ============================================================================
# DataSetFromArray: Main dataset for training and inference
# ============================================================================

class DataSetFromArray(torch.utils.data.Dataset):
    """
    Dataset for 4D-STEM diffraction patterns stored as numpy array.

    Parameters
    ----------
    data : np.ndarray
        4D array of shape (Rx, Ry, Qx, Qy)
    neighbor_mode : str
        'spatial' or 'temporal'
    """
    def __init__(self, data, neighbor_mode='spatial'):
        self.data = data
        self.neighbor_mode = neighbor_mode

        valid_modes = ['spatial', 'temporal']
        if neighbor_mode not in valid_modes:
            raise ValueError(f"neighbor_mode must be one of {valid_modes}, got '{neighbor_mode}'")

        self.Rx_val, self.Ry_val, self.Qx_val, self.Qy_val = data.shape

        print(f"DataSetFromArray initialized:")
        print(f"  Shape: ({self.Rx_val}, {self.Ry_val}, {self.Qx_val}, {self.Qy_val})")
        print(f"  Neighbor mode: {neighbor_mode}")

    def Rx(self):
        return self.Rx_val

    def Ry(self):
        return self.Ry_val

    def Qx(self):
        return self.Qx_val

    def Qy(self):
        return self.Qy_val

    # ------------------------------------------------------------------
    # Helper: boundary-safe data access
    # ------------------------------------------------------------------

    def _safe(self, rx, ry, center):
        """Return data[rx, ry] if in bounds, else center frame."""
        if 0 <= rx < self.Rx_val and 0 <= ry < self.Ry_val:
            return self.data[rx, ry]
        return center

    # ------------------------------------------------------------------
    # 9-channel spatial: full 3x3 scan grid
    # ------------------------------------------------------------------

    def _get_3x3_spatial(self, rx, ry):
        """
        Full 3x3 spatial grid around (rx, ry). Center at index 4.

        Grid layout → index mapping:
            (rx-1,ry-1)=0   (rx-1,ry)=1   (rx-1,ry+1)=2
            (rx,  ry-1)=3   (rx,  ry)=4   (rx,  ry+1)=5
            (rx+1,ry-1)=6   (rx+1,ry)=7   (rx+1,ry+1)=8

        Returns list of 9 np.ndarray frames.
        """
        center = self.data[rx, ry]
        s = self._safe
        return [
            s(rx-1, ry-1, center), s(rx-1, ry, center), s(rx-1, ry+1, center),  # 0, 1, 2
            s(rx,   ry-1, center), center,               s(rx,   ry+1, center),  # 3, 4, 5
            s(rx+1, ry-1, center), s(rx+1, ry, center), s(rx+1, ry+1, center),  # 6, 7, 8
        ]

    # ------------------------------------------------------------------
    # 9-channel temporal: double-arrow geometry
    # ------------------------------------------------------------------

    def _get_doublearrow_temporal(self, rx, ry):
        """
        Double-arrow temporal geometry. Center at index 4.

        Physical layout (raster scan goes left→right along ry):

                     (rx-1,ry-1)          (rx-1,ry+1)
          (rx,ry-2)  (rx,  ry-1) [center] (rx,  ry+1)  (rx,ry+2)
                     (rx+1,ry-1)          (rx+1,ry+1)

        Mapped to 3x3 grid so model groups form lines through center:

            0:(rx-1,ry-1)   1:(rx,ry-2)     2:(rx+1,ry-1)
            3:(rx,  ry-1)  [4: center ]     5:(rx,  ry+1)
            6:(rx-1,ry+1)   7:(rx,ry+2)     8:(rx+1,ry+1)

        Resulting groups:
            [0,4,8] = (rx-1,ry-1), center, (rx+1,ry+1) — diagonal ↘
            [1,4,7] = (rx,  ry-2), center, (rx,  ry+2) — extended temporal
            [2,4,6] = (rx+1,ry-1), center, (rx-1,ry+1) — diagonal ↗
            [3,4,5] = (rx,  ry-1), center, (rx,  ry+1) — direct temporal

        All neighbors are within ±2 raster positions of center.
        """
        center = self.data[rx, ry]
        s = self._safe
        return [
            s(rx-1, ry-1, center),  # 0: t-1, row above      → group [0,4,8] start
            s(rx,   ry-2, center),  # 1: t-2                  → group [1,4,7] start
            s(rx+1, ry-1, center),  # 2: t-1, row below       → group [2,4,6] start
            s(rx,   ry-1, center),  # 3: t-1                  → group [3,4,5] start
            center,                  # 4: center
            s(rx,   ry+1, center),  # 5: t+1                  → group [3,4,5] end
            s(rx-1, ry+1, center),  # 6: t+1, row above       → group [2,4,6] end
            s(rx,   ry+2, center),  # 7: t+2                  → group [1,4,7] end
            s(rx+1, ry+1, center),  # 8: t+1, row below       → group [0,4,8] end
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def getitem(self, position, samplershape='3d'):
        """
        Get 9 frames in 3x3 grid order (center at index 4).

        Parameters
        ----------
        position : list
            [rx, ry]
        samplershape : str
            Kept for backward compatibility.

        Returns
        -------
        item_input : torch.Tensor (9, Qx, Qy)
            All 9 frames including center at index 4
        item_output : torch.Tensor (Qx, Qy)
            Center frame
        """
        rx, ry = position

        if self.neighbor_mode == 'spatial':
            frames = self._get_3x3_spatial(rx, ry)
        elif self.neighbor_mode == 'temporal':
            frames = self._get_doublearrow_temporal(rx, ry)
        else:
            raise ValueError(f"Unknown neighbor_mode: {self.neighbor_mode}")

        item_input = torch.tensor(np.stack(frames, axis=0), dtype=torch.float32)
        item_output = torch.tensor(frames[4], dtype=torch.float32)  # center at index 4
        return item_input, item_output

    def set_neighbor_mode(self, mode):
        valid_modes = ['spatial', 'temporal']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        self.neighbor_mode = mode

    def __getitem__(self, index):
        return self.getitem(index)


# ============================================================================
# AlternatingDataset
# ============================================================================

class AlternatingDataset(torch.utils.data.Dataset):
    """
    Wrapper that alternates neighbor mode (spatial/temporal) by epoch.
    Always produces 9-channel output with center at index 4.

    Parameters
    ----------
    base_dataset : DataSetFromArray
    mode : str
        'spatial', 'temporal', 'alternating_spatial', 'alternating_temporal', 'random'
    image_size : int or None
        Random crop size for training
    """
    def __init__(self, base_dataset, mode='alternating_spatial', image_size=None):
        self.base_dataset = base_dataset
        self.mode = mode
        self.image_size = image_size
        self.current_epoch = 0
        self.current_neighbor_mode = 'spatial' if 'spatial' in mode else 'temporal'
        self.sample_modes = None

        # Margin = 2 required: spatial needs ±1, temporal double-arrow needs ±2
        margin = 2
        self.positions = []
        for rx in range(margin, base_dataset.Rx() - margin):
            for ry in range(margin, base_dataset.Ry() - margin):
                self.positions.append([rx, ry])

        print(f"  AlternatingDataset: {len(self.positions)} positions, "
              f"9-channel grid output, margin={margin}")

    def set_epoch(self, epoch):
        """Call at start of each epoch."""
        self.current_epoch = epoch

        if 'alternating' in self.mode:
            if 'spatial' in self.mode:
                self.current_neighbor_mode = 'spatial' if epoch % 2 == 0 else 'temporal'
            else:
                self.current_neighbor_mode = 'temporal' if epoch % 2 == 0 else 'spatial'

            self.base_dataset.neighbor_mode = self.current_neighbor_mode
            print(f"  -> Epoch {epoch}: {self.current_neighbor_mode} neighbors")

        elif self.mode == 'random':
            np.random.seed(epoch)
            self.sample_modes = np.random.choice(
                ['spatial', 'temporal'], size=len(self.positions)
            )

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        rx, ry = self.positions[idx]

        # Handle random mode by temporarily switching
        if self.mode == 'random' and self.sample_modes is not None:
            old_mode = self.base_dataset.neighbor_mode
            self.base_dataset.neighbor_mode = self.sample_modes[idx]
            all_frames, center = self.base_dataset.getitem([rx, ry])
            self.base_dataset.neighbor_mode = old_mode
        else:
            all_frames, center = self.base_dataset.getitem([rx, ry])

        target = center.unsqueeze(0)  # (1, Qx, Qy)

        # Random crop
        if self.image_size is not None:
            _, H, W = all_frames.shape
            if H > self.image_size and W > self.image_size:
                h = np.random.randint(0, H - self.image_size)
                w = np.random.randint(0, W - self.image_size)
                all_frames = all_frames[:, h:h+self.image_size, w:w+self.image_size]
                target = target[:, h:h+self.image_size, w:w+self.image_size]

        return all_frames.float(), target.float()


# ============================================================================
# FixedModeDataset
# ============================================================================

class FixedModeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, positions, image_size=None):
        self.dataset = dataset
        self.positions = positions
        self.image_size = image_size

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        try:
            all_frames, center = self.dataset.getitem(self.positions[idx])
            target = center.unsqueeze(0)

            if self.image_size is not None:
                _, H, W = all_frames.shape
                if H > self.image_size and W > self.image_size:
                    h = np.random.randint(0, H - self.image_size)
                    w = np.random.randint(0, W - self.image_size)
                    all_frames = all_frames[:, h:h+self.image_size, w:w+self.image_size]
                    target = target[:, h:h+self.image_size, w:w+self.image_size]

            return all_frames.float(), target.float()
        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            raise


# ============================================================================
# Helper
# ============================================================================

def build_training_dataset(data, neighbor_mode='alternating_spatial', image_size=256):
    """
    Build the dataset for any 9-channel model variant.

    Parameters
    ----------
    data : np.ndarray (Rx, Ry, Qx, Qy)
    neighbor_mode : str
        'spatial', 'temporal', 'alternating_spatial', 'alternating_temporal', 'random'
    image_size : int or None
        Random crop size. Use None or value >= DP size for no crop.

    Returns
    -------
    train_dataset : Dataset
        Returns (9, Qx, Qy) inputs and (1, Qx, Qy) targets
    """
    base_mode = 'spatial' if 'spatial' in neighbor_mode else 'temporal'
    if neighbor_mode in ['spatial', 'temporal']:
        base_mode = neighbor_mode

    base_dataset = DataSetFromArray(data, neighbor_mode=base_mode)

    if neighbor_mode in ['alternating_spatial', 'alternating_temporal', 'random']:
        ds = AlternatingDataset(base_dataset, mode=neighbor_mode, image_size=image_size)
        ds.set_epoch(0)
        return ds
    else:
        # Fixed mode: use margin=1 for double-arrow temporal reach (minimal loss of info compared to margin=2)
        margin = 1
        positions = []
        for rx in range(margin, base_dataset.Rx() - margin):
            for ry in range(margin, base_dataset.Ry() - margin):
                positions.append([rx, ry])
        return FixedModeDataset(base_dataset, positions, image_size=image_size)


# ============================================================================
# HDF5-based DataSet (backward compatibility — returns 9-channel grid)
# ============================================================================

class DataSet(torch.utils.data.Dataset):
    """h5py-based DataSet that loads data into RAM. Returns 9-channel 3x3 grid."""
    def __init__(self, file_path):
        self.imgs = []
        f = h5py.File(file_path, 'r')
        self.imgs.append(np.array(f['Experiments/__unnamed__/data/']))
        f.close()
        self.Rx_val = self.imgs[0].shape[0]
        self.Ry_val = self.imgs[0].shape[1]

    def Rx(self):
        return self.Rx_val

    def Ry(self):
        return self.Ry_val

    def _safe(self, rx, ry, center):
        if 0 <= rx < self.Rx_val and 0 <= ry < self.Ry_val:
            return self.imgs[0][rx, ry]
        return center

    def getitem(self, index, samplershape='3d'):
        rx, ry = index
        center = self.imgs[0][rx, ry]
        s = self._safe
        frames = [
            s(rx-1, ry-1, center), s(rx-1, ry, center), s(rx-1, ry+1, center),
            s(rx,   ry-1, center), center,               s(rx,   ry+1, center),
            s(rx+1, ry-1, center), s(rx+1, ry, center), s(rx+1, ry+1, center),
        ]
        item_input = torch.tensor(np.stack(frames, axis=0), dtype=torch.float32)
        item_output = torch.tensor(center, dtype=torch.float32)
        return item_input, item_output

    def __getitem__(self, index):
        return self.getitem(index)


# ============================================================================
# Several datasets at once, with an arbitrary scan-position subset
# ============================================================================

def build_scan_mask(Rx, Ry, mode='all', corner=None, fraction=0.5, half='A',
                    block=4, path=None):
    """Boolean (Rx, Ry) mask of the scan positions used as training TARGETS.

    mode
        'all'          every position
        'corner'       the last ``corner`` x ``corner`` positions (old corner crop)
        'diagonal'     positions below (half 'A') or above (half 'B') a diagonal
                       cutting off ``fraction`` of the scan
        'checkerboard' ``block`` x ``block`` squares, half 'A' or 'B'
        'file'         a boolean (Rx, Ry) .npy at ``path``
    Training on half 'A' and evaluating on half 'B' is a leave-one-region-out
    test. Neighbours are still read from the whole scan, so keep a margin of a
    few positions between the halves if held-out targets must stay unseen.
    """
    ii, jj = np.mgrid[0:Rx, 0:Ry]
    if mode == 'all':
        m = np.ones((Rx, Ry), bool)
    elif mode == 'corner':
        n = corner or min(Rx, Ry) // 2
        m = np.zeros((Rx, Ry), bool)
        m[-n:, -n:] = True
    elif mode == 'diagonal':
        m = (ii / Rx + jj / Ry) < (2 * fraction)
        m = m if half == 'A' else ~m
    elif mode == 'checkerboard':
        m = ((ii // block + jj // block) % 2 == 0)
        m = m if half == 'A' else ~m
    elif mode == 'file':
        m = np.load(path).astype(bool)
        if m.shape != (Rx, Ry):
            raise ValueError(f"scan mask {path} has shape {m.shape}, scan is {(Rx, Ry)}")
    else:
        raise ValueError(f"unknown scan-mask mode '{mode}'")
    return m


class MultiCubeDataset(torch.utils.data.Dataset):
    """Pools several 4D datasets into one training set.

    Each dataset keeps its own DataSetFromArray, so the neighbour geometry is
    exactly the single-dataset one and neighbours never cross from one dataset
    into another. Returns ``(frames, target, source_id)``:
    frames (9, Qx, Qy), target (1, Qx, Qy), source_id int.

    Parameters
    ----------
    cubes : list of np.ndarray (Rx, Ry, Qx, Qy); scan shapes may differ
    masks : list of boolean (Rx, Ry) arrays or None (= all positions)
    neighbor_mode : 'spatial', 'temporal', 'alternating_spatial',
                    'alternating_temporal' or 'random'
    margin : positions excluded at every scan edge; default as in
             build_training_dataset (2 for alternating/random, 1 otherwise)
    """
    def __init__(self, cubes, masks=None, neighbor_mode='alternating_spatial',
                 image_size=None, margin=None):
        self.mode = neighbor_mode
        self.image_size = image_size
        if margin is None:
            margin = 2 if neighbor_mode in ('alternating_spatial', 'alternating_temporal',
                                            'random') else 1
        self.margin = margin
        self._spatial = [DataSetFromArray(c, neighbor_mode='spatial') for c in cubes]
        self._temporal = [DataSetFromArray(c, neighbor_mode='temporal') for c in cubes]
        qshapes = {(d.Qx(), d.Qy()) for d in self._spatial}
        if len(qshapes) != 1:
            raise ValueError(f"all datasets must share the detector shape, got {qshapes}")

        self.index = []                                   # (source, rx, ry)
        for ci, d in enumerate(self._spatial):
            m = np.ones((d.Rx(), d.Ry()), bool) if masks is None or masks[ci] is None \
                else np.asarray(masks[ci], bool)
            for rx in range(margin, d.Rx() - margin):
                for ry in range(margin, d.Ry() - margin):
                    if m[rx, ry]:
                        self.index.append((ci, rx, ry))
        if not self.index:
            raise ValueError("no training positions left (scan masks and margin)")
        self.sample_modes = None
        self.set_epoch(0)
        print(f"  MultiCubeDataset: {len(self.index)} positions from {len(cubes)} "
              f"dataset(s), margin={margin}, mode='{neighbor_mode}'")

    def set_epoch(self, epoch):
        """Call at the start of every epoch (same rules as AlternatingDataset)."""
        if self.mode in ('spatial', 'temporal'):
            self.current_neighbor_mode = self.mode
        elif self.mode == 'alternating_spatial':
            self.current_neighbor_mode = 'spatial' if epoch % 2 == 0 else 'temporal'
        elif self.mode == 'alternating_temporal':
            self.current_neighbor_mode = 'temporal' if epoch % 2 == 0 else 'spatial'
        elif self.mode == 'random':
            self.current_neighbor_mode = 'random'
            rng = np.random.RandomState(epoch)
            self.sample_modes = rng.choice(['spatial', 'temporal'], size=len(self.index))
        else:
            raise ValueError(f"unknown neighbor_mode '{self.mode}'")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ci, rx, ry = self.index[idx]
        mode = self.sample_modes[idx] if self.mode == 'random' else self.current_neighbor_mode
        base = self._temporal[ci] if mode == 'temporal' else self._spatial[ci]
        frames, center = base.getitem([rx, ry])
        target = center.unsqueeze(0)
        if self.image_size is not None:
            _, H, W = frames.shape
            if H > self.image_size and W > self.image_size:
                h = np.random.randint(0, H - self.image_size)
                w = np.random.randint(0, W - self.image_size)
                frames = frames[:, h:h + self.image_size, w:w + self.image_size]
                target = target[:, h:h + self.image_size, w:w + self.image_size]
        return frames.float(), target.float(), ci


def build_multi_dataset(cubes, masks=None, neighbor_mode='alternating_spatial',
                        image_size=None, margin=None):
    return MultiCubeDataset(cubes, masks=masks, neighbor_mode=neighbor_mode,
                            image_size=image_size, margin=margin)
