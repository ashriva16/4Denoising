import numpy as np
from pathlib import Path

"""
Import functions for specific formats.
New formats can be added here.
"""

def load_4dstem(filepath, crop_N=None):
    """
    Load 4D-STEM data from any supported format.
    
    Supported formats:
        .npy                         — numpy array (Rx, Ry, Qx, Qy)
        .h5/.hdf5 (FPD)             — fpd_expt/fpd_data/data
        .h5/.hdf5 (py4DSTEM)        — Experiments/__unnamed__/data/
        .h5/.hdf5 (generic)         — 'data' or 'datacube' key
        .h5 master + numbered files — reads via master.h5
    
    Parameters
    ----------
    filepath : str or Path
    crop_N : int or None
        Crop N pixels from each edge of the DP (detector padding removal)
    
    Returns
    -------
    data : np.ndarray (Rx, Ry, Qx, Qy), float32
    metadata : dict
    """
    filepath = Path(filepath)
    metadata = {}

    if filepath.suffix == '.npy':
        data = np.load(filepath).astype(np.float32)

    elif filepath.suffix in ('.h5', '.hdf5'):
        import h5py

        with h5py.File(filepath, 'r') as f:
            # Try formats in order of specificity
            if 'fpd_expt/fpd_data/data' in f:
                data, metadata = _load_fpd(f)
            elif 'Experiments/__unnamed__/data/' in f:
                data = np.array(f['Experiments/__unnamed__/data/'], dtype=np.float32)
            elif 'data' in f:
                data = np.array(f['data'], dtype=np.float32)
            elif 'datacube' in f:
                data = np.array(f['datacube'], dtype=np.float32)
            else:
                keys = _list_datasets(f)
                raise KeyError(
                    f"No recognized 4D-STEM dataset in {filepath}.\n"
                    f"Available datasets: {keys}"
                )
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Optional detector crop
    if crop_N is not None and crop_N > 0:
        data = np.ascontiguousarray(
            data[:, :, crop_N:-crop_N, crop_N:-crop_N]
        ).astype(np.float32)

    data = np.ascontiguousarray(data.astype(np.float32))

    print(f"Loaded: {filepath.name}, shape={data.shape}, "
          f"dtype={data.dtype}, range=[{data.min():.1f}, {data.max():.1f}]")

    return data, metadata

def _load_fpd(f):
    """Load FPD-format HDF5 (Fast Pixelated Detector)."""
    data = np.array(f['fpd_expt/fpd_data/data'], dtype=np.float32)
    
    metadata = {}
    
    # Dimensional scales
    dim_names = ['scan_x', 'scan_y', 'det_x', 'det_y']
    for i, name in enumerate(dim_names, 1):
        key = f'fpd_expt/fpd_data/dim{i}'
        if key in f:
            metadata[f'{name}_scale'] = f[key][:]

    # Auxiliary datasets
    aux = {
        'sum_image': 'fpd_expt/fpd_sum_im/data',
        'sum_diffraction': 'fpd_expt/fpd_sum_dif/data',
        'survey_image': 'fpd_expt/survey_image/data',
        'virtual_image': 'fpd_expt/virtual_image/data',
    }
    for name, path in aux.items():
        if path in f:
            metadata[name] = f[path][:]

    # Group attributes
    for group in ['fpd_expt', 'microscope', 'sample', 'user']:
        if group in f:
            attrs = dict(f[group].attrs)
            if attrs:
                metadata[f'{group}_attrs'] = attrs

    return data, metadata


def _list_datasets(f, prefix=''):
    """Recursively list all datasets in an HDF5 file."""
    import h5py
    datasets = []
    for key in f.keys():
        path = f'{prefix}/{key}' if prefix else key
        if isinstance(f[key], h5py.Dataset):
            datasets.append(f'{path} {f[key].shape} {f[key].dtype}')
        elif isinstance(f[key], h5py.Group):
            datasets.extend(_list_datasets(f[key], path))
    return datasets
def import_fpd_hdf5(filepath, load_metadata=True):
    """
    Import 4D-STEM data from FPD (Fast Pixelated Detector) HDF5 format.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file
    load_metadata : bool, default True
        Whether to load dimensional scaling and other metadata
    
    Returns
    -------
    datacube : py4DSTEM.DataCube
        4D-STEM datacube with shape (Rx, Ry, Qx, Qy)
    metadata : dict
        Dictionary containing dimensional scales and other metadata
        (only if load_metadata=True)
    
    Notes
    -----
    Expected HDF5 structure:
    - fpd_expt/fpd_data/data: (Rx, Ry, Qx, Qy) main 4D dataset
    - fpd_expt/fpd_data/dim1-4: dimensional scales
    - fpd_expt/fpd_sum_im/data: (Rx, Ry) sum image
    - fpd_expt/fpd_sum_dif/data: (Qx, Qy) sum diffraction pattern
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        # Verify expected structure
        if 'fpd_expt/fpd_data/data' not in f:
            raise ValueError(
                f"Expected dataset 'fpd_expt/fpd_data/data' not found in {filepath}"
            )
        
        # Load main 4D dataset
        main_data = f['fpd_expt/fpd_data/data']
        if len(main_data.shape) != 4:
            raise ValueError(
                f"Expected 4D data, got shape {main_data.shape}"
            )
        
        raw_data = main_data[:]
        print(f"Loaded 4D dataset: {raw_data.shape} ({raw_data.dtype})")
        
        # Create py4DSTEM DataCube
        datacube = py4DSTEM.DataCube(raw_data)
        
        if not load_metadata:
            return datacube
        
        # Load metadata
        metadata = {}
        
        # Dimensional scales
        try:
            dim_names = ['scan_x', 'scan_y', 'det_x', 'det_y']
            for i, dim_name in enumerate(dim_names, 1):
                dim_key = f'fpd_expt/fpd_data/dim{i}'
                if dim_key in f:
                    metadata[f'{dim_name}_scale'] = f[dim_key][:]
                    print(f"  {dim_name}_scale: {len(f[dim_key][:])} points")
        except Exception as e:
            print(f"Warning: Could not load dimensional scales: {e}")
        
        # Additional datasets
        extra_datasets = {
            'sum_image': 'fpd_expt/fpd_sum_im/data',
            'sum_diffraction': 'fpd_expt/fpd_sum_dif/data', 
            'survey_image': 'fpd_expt/survey_image/data',
            'virtual_image': 'fpd_expt/virtual_image/data'
        }
        
        for name, path in extra_datasets.items():
            if path in f:
                metadata[name] = f[path][:]
                metadata[f'{name}_shape'] = f[path].shape
                print(f"  {name}: {f[path].shape}")
        
        # HDF5 attributes (if any)
        for group_path in ['fpd_expt', 'microscope', 'sample', 'user']:
            if group_path in f:
                group_attrs = dict(f[group_path].attrs)
                if group_attrs:
                    metadata[f'{group_path}_attrs'] = group_attrs
        
        return datacube, metadata
        
# Force data into RAM at import time
def import_fpd_hdf5_fast(filepath, load_metadata=False):
    datacube = import_fpd_hdf5(filepath, load_metadata=False)
    
    # Force copy to contiguous RAM
    datacube.data = np.ascontiguousarray(datacube.data.copy())
    
    return datacube.data

def import_fpd_hdf5_fast_contiguous(hdf5_path, crop_N=None):
    """Import with proper memory layout"""
    raw_data = import_fpd_hdf5_fast(hdf5_path)
    
    if crop_N is not None:
        # Crop and immediately copy to contiguous memory
        cropped = raw_data[:, :, crop_N:-crop_N, crop_N:-crop_N]
        contiguous_data = np.ascontiguousarray(cropped.astype(np.uint16))
    else:
        # For full data, force contiguous copy
        contiguous_data = np.ascontiguousarray(raw_data.astype(np.uint16))
    
    return contiguous_data
