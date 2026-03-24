"""
Import functions for specific formats.
New formats can be added here.
"""

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
