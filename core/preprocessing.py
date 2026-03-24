"""
Preprocessing utilities for 4D-STEM denoising.

Functions:
    bin_datacube        — 2×2 spatial binning of DPs (sum, preserves Poisson stats). Can also handle larger binning
    offset_datacube     — +1 offset to eliminate binary sparsity artifacts. CAUTION: destroys raw signal from experiment
    remove_offset       — subtract offset + clip after inference
    correct_pnll_bias   — rescale denoised data to match original total counts
    detect_dead_pixels   — identify dead/hot pixel clusters from PACBED
    correct_dead_pixels     — inpainting of masked pixels
    radial_profile      — azimuthally-integrated radial profile
    compare_radial_profiles — multi-datacube radial profile comparison
    save_denoised_h5    — compressed HDF5 output (float16 + shuffle + gzip)
    upsample_denoised   — bicubic upsampling on GPU after binned inference
"""

import numpy as np
import torch
import torch.nn.functional as F
import os

# ---- Binning ----

def bin_datacube(data, bin_factor=2):
    """Bin DPs by summing (preserves Poisson statistics)."""
    from skimage.measure import block_reduce
    Rx, Ry, Qx, Qy = data.shape
    binned = np.zeros((Rx, Ry, Qx // bin_factor, Qy // bin_factor), dtype=np.float32)
    for rx in range(Rx):
        for ry in range(Ry):
            binned[rx, ry] = block_reduce(
                data[rx, ry], block_size=(bin_factor, bin_factor), func=np.sum
            )
    print(f"Binned {bin_factor}x: {data.shape} → {binned.shape}")
    return binned


# ---- Offset for sparse data ----

def offset_datacube(data, offset=1.0):
    """Add constant offset to eliminate binary sparsity.
    Required when data has many exact-zero pixels that cause
    checkerboard artifacts with stride-2 pooling."""
    data_offset = data.astype(np.float32) + offset
    print(f"Applied +{offset} offset: range [{data_offset.min():.1f}, {data_offset.max():.1f}]")
    return data_offset


def remove_offset(denoised, offset=1.0):
    """Subtract offset after inference and clip to non-negative."""
    return np.clip(denoised - offset, 0, None)


# ---- Poisson NLL bias correction ----

def correct_pnll_bias(original_data, denoised_data):
    """Rescale denoised data so total counts match original.
    Poisson NLL systematically over-predicts by Jensen's inequality."""
    original_mean = original_data.mean(axis=(0, 1))
    denoised_mean = denoised_data.mean(axis=(0, 1))
    scale = original_mean.sum() / denoised_mean.sum()
    print(f"Poisson NLL bias correction: {scale:.4f}")
    return denoised_data * scale


# ---- Dead pixel handling ----
try:
    import py4DSTEM
    HAS_PY4DSTEM = True
except ImportError:
    HAS_PY4DSTEM = False
def detect_dead_pixels(datacube, method='statistical', threshold_factor=3.0, 
                      min_dead_fraction=0.8, visualize=True):
    """
    Detect systematic dead pixels in a 4D-STEM datacube.
    
    Parameters
    ----------
    datacube : py4DSTEM.DataCube or numpy.ndarray
        4D data with shape (Rx, Ry, Qx, Qy)
    method : str, default 'statistical'
        Detection method:
        - 'statistical': Based on intensity statistics across scan positions
        - 'zero_count': Pixels that are zero/very low in most positions
        - 'combined': Both methods
    threshold_factor : float, default 3.0
        For 'statistical': pixels with mean < (global_mean - threshold_factor * global_std)
    min_dead_fraction : float, default 0.8
        For 'zero_count': pixel is dead if below threshold in >80% of positions
    visualize : bool, default True
        Show detected dead pixel mask
    
    Returns
    -------
    dead_pixel_mask : numpy.ndarray (Qx, Qy)
        Boolean mask: True = dead pixel, False = good pixel
    stats : dict
        Detection statistics and thresholds used
    """
    
    # Handle py4DSTEM DataCube
    if hasattr(datacube, 'data'):
        data = datacube.data
    else:
        data = datacube
    
    Rx, Ry, Qx, Qy = data.shape
    print(f"Detecting dead pixels in {Qx}×{Qy} detector array...")
    
    # Compute per-pixel statistics across all scan positions
    pixel_means = np.mean(data, axis=(0, 1))      # (Qx, Qy)
    pixel_stds = np.std(data, axis=(0, 1))        # (Qx, Qy)
    pixel_mins = np.min(data, axis=(0, 1))        # (Qx, Qy)
    pixel_maxs = np.max(data, axis=(0, 1))        # (Qx, Qy)
    
    # Global statistics
    global_mean = np.mean(pixel_means)
    global_std = np.std(pixel_means)
    
    dead_masks = {}
    
    # Method 1: Statistical outliers
    if method in ['statistical', 'combined']:
        threshold_low = global_mean - threshold_factor * global_std
        statistical_dead = pixel_means < threshold_low
        dead_masks['statistical'] = statistical_dead
        n_stat = statistical_dead.sum()
        print(f"  Statistical method: {n_stat} dead pixels (threshold < {threshold_low:.2f})")
    
    # Method 2: Zero/low count analysis
    if method in ['zero_count', 'combined']:
        # Count how often each pixel is "effectively zero"
        zero_threshold = np.percentile(data, 1)  # Very low threshold
        zero_counts = np.sum(data <= zero_threshold, axis=(0, 1))  # (Qx, Qy)
        zero_fraction = zero_counts / (Rx * Ry)
        
        zero_count_dead = zero_fraction > min_dead_fraction
        dead_masks['zero_count'] = zero_count_dead
        n_zero = zero_count_dead.sum()
        print(f"  Zero-count method: {n_zero} dead pixels (>{min_dead_fraction:.0%} positions below {zero_threshold:.2f})")
    
    # Combine methods
    if method == 'combined':
        final_mask = dead_masks['statistical'] | dead_masks['zero_count']
    else:
        final_mask = dead_masks[method]
    
    n_dead = final_mask.sum()
    dead_percentage = 100 * n_dead / (Qx * Qy)
    print(f"  Final: {n_dead} dead pixels ({dead_percentage:.2f}% of detector)")
    
    # Prepare statistics
    stats = {
        'method': method,
        'n_dead_pixels': n_dead,
        'dead_percentage': dead_percentage,
        'global_mean': global_mean,
        'global_std': global_std,
        'threshold_factor': threshold_factor,
        'min_dead_fraction': min_dead_fraction,
        'pixel_means': pixel_means,
        'pixel_stds': pixel_stds,
    }
    
    if method in ['zero_count', 'combined']:
        stats['zero_threshold'] = zero_threshold
        stats['zero_fraction_map'] = zero_fraction
    
    # Visualization
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Mean intensity map
        im1 = axes[0,0].imshow(pixel_means, cmap='viridis')
        axes[0,0].set_title('Mean Intensity per Pixel')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Standard deviation map
        im2 = axes[0,1].imshow(pixel_stds, cmap='plasma')
        axes[0,1].set_title('Std Dev per Pixel')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Dead pixel mask
        axes[0,2].imshow(final_mask, cmap='RdYlBu_r')
        axes[0,2].set_title(f'Dead Pixels ({n_dead} total)')
        
        # Log mean (to see low values better)
        im4 = axes[1,0].imshow(np.log10(pixel_means + 0.1), cmap='viridis')
        axes[1,0].set_title('Log Mean Intensity')
        plt.colorbar(im4, ax=axes[1,0])
        
        # Individual method masks (if combined)
        if method == 'combined':
            axes[1,1].imshow(dead_masks['statistical'], cmap='Reds', alpha=0.7)
            axes[1,1].set_title(f'Statistical Dead ({dead_masks["statistical"].sum()})')
            
            axes[1,2].imshow(dead_masks['zero_count'], cmap='Blues', alpha=0.7)
            axes[1,2].set_title(f'Zero-count Dead ({dead_masks["zero_count"].sum()})')
        else:
            # Zero fraction map if available
            if 'zero_fraction_map' in stats:
                im6 = axes[1,1].imshow(stats['zero_fraction_map'], cmap='Reds')
                axes[1,1].set_title('Fraction of Zero Values')
                plt.colorbar(im6, ax=axes[1,1])
            
            # Histogram of pixel means
            axes[1,2].hist(pixel_means.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[1,2].axvline(global_mean, color='red', linestyle='--', label='Global Mean')
            if 'statistical' in dead_masks:
                axes[1,2].axvline(threshold_low, color='orange', linestyle='--', label='Dead Threshold')
            axes[1,2].set_xlabel('Mean Intensity')
            axes[1,2].set_ylabel('Count')
            axes[1,2].set_title('Pixel Mean Distribution')
            axes[1,2].legend()
            axes[1,2].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    return final_mask, stats

def correct_dead_pixels(datacube, dead_pixel_mask, method='median_local', 
                       kernel_size=3, visualize_sample=True):
    """
    Correct dead pixels in a 4D-STEM datacube.
    
    Parameters
    ----------
    datacube : py4DSTEM.DataCube or numpy.ndarray
        4D data with shape (Rx, Ry, Qx, Qy)
    dead_pixel_mask : numpy.ndarray (Qx, Qy)
        Boolean mask: True = dead pixel to correct
    method : str, default 'median_local'
        Correction method:
        - 'median_local': Replace with local median in each diffraction pattern
        - 'median_global': Replace with global median value across detector
        - 'interpolate': Bilinear interpolation from neighbors
        - 'mean_local': Replace with local mean
    kernel_size : int, default 3
        Size of local neighborhood for median/mean (must be odd)
    visualize_sample : bool, default True
        Show before/after for a sample diffraction pattern
    
    Returns
    -------
    corrected_datacube : py4DSTEM.DataCube
        Datacube with dead pixels corrected
    """
    
    # Handle py4DSTEM DataCube
    if hasattr(datacube, 'data'):
        data = datacube.data.copy()
        is_datacube = True
    else:
        data = datacube.copy()
        is_datacube = False
    
    Rx, Ry, Qx, Qy = data.shape
    n_dead = dead_pixel_mask.sum()
    print(f"Correcting {n_dead} dead pixels using method '{method}'...")
    
    if n_dead == 0:
        print("No dead pixels to correct.")
        return py4DSTEM.DataCube(data) if is_datacube else data
    
    # Get dead pixel coordinates
    dead_coords = np.where(dead_pixel_mask)
    
    if method == 'median_global':
        # Simple: replace with global median across all detector pixels
        global_median = np.median(data)
        for rx in range(Rx):
            for ry in range(Ry):
                data[rx, ry][dead_pixel_mask] = global_median
    
    elif method in ['median_local', 'mean_local']:
        # Replace each dead pixel with local neighborhood median/mean
        func = np.median if method == 'median_local' else np.mean
        half_kernel = kernel_size // 2
        
        for rx in range(Rx):
            for ry in range(Ry):
                pattern = data[rx, ry]
                for i, j in zip(*dead_coords):
                    # Define local neighborhood
                    i_min, i_max = max(0, i-half_kernel), min(Qx, i+half_kernel+1)
                    j_min, j_max = max(0, j-half_kernel), min(Qy, j+half_kernel+1)
                    
                    # Extract neighborhood, excluding dead pixels
                    neighborhood = pattern[i_min:i_max, j_min:j_max]
                    local_mask = dead_pixel_mask[i_min:i_max, j_min:j_max]
                    good_neighbors = neighborhood[~local_mask]
                    
                    if len(good_neighbors) > 0:
                        pattern[i, j] = func(good_neighbors)
                    else:
                        # Fallback: use global median if no good neighbors
                        pattern[i, j] = np.median(pattern)
    
    elif method == 'interpolate':
        # Bilinear interpolation using scipy
        from scipy.interpolate import griddata
        
        for rx in range(Rx):
            for ry in range(Ry):
                pattern = data[rx, ry]
                
                if n_dead > 0:
                    # Get coordinates of good pixels
                    good_mask = ~dead_pixel_mask
                    good_coords = np.where(good_mask)
                    good_values = pattern[good_coords]
                    
                    # Interpolate to dead pixel positions
                    if len(good_values) > 3:  # Need minimum points for interpolation
                        good_points = np.column_stack(good_coords)
                        dead_points = np.column_stack(dead_coords)
                        
                        interpolated = griddata(
                            good_points, good_values, dead_points, 
                            method='linear', fill_value=np.median(pattern)
                        )
                        
                        pattern[dead_coords] = interpolated
    
    print(f"✓ Dead pixel correction complete.")
                           
    # Return appropriate type
    if is_datacube and HAS_PY4DSTEM:
        return py4DSTEM.DataCube(data)
    return data                          


# ---- Radial profiles ----

def radial_profile(pattern, center_qx, center_qy):
    """Azimuthally-integrated radial profile extending to corners."""
    Qx, Qy = pattern.shape
    y, x = np.mgrid[:Qx, :Qy]
    r = np.sqrt((x - center_qy)**2 + (y - center_qx)**2).astype(int)

    r_max = int(np.sqrt(max(center_qx, Qx - center_qx)**2 +
                        max(center_qy, Qy - center_qy)**2))

    profile_sum = np.zeros(r_max)
    profile_std = np.zeros(r_max)
    counts = np.zeros(r_max)

    for ri in range(r_max):
        mask = r == ri
        n = mask.sum()
        if n > 0:
            vals = pattern[mask]
            profile_sum[ri] = vals.sum()
            counts[ri] = n
            profile_std[ri] = vals.std() if n > 1 else 0

    radii = np.arange(r_max)
    profile_mean = np.where(counts > 0, profile_sum / counts, 0)

    return radii, profile_sum, profile_mean, profile_std


def compare_radial_profiles(datacubes, labels, center_qx, center_qy,
                            plot_type='both', log_scale=True):
    """Compare radial profiles from multiple datacubes."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if plot_type == 'both' else 1,
                             figsize=(14 if plot_type == 'both' else 7, 5))
    if plot_type != 'both':
        axes = [axes]

    for dc, label in zip(datacubes, labels):
        data = dc.data if hasattr(dc, 'data') else dc
        mean_dp = data.mean(axis=(0, 1))
        radii, profile_sum, profile_avg, profile_std = radial_profile(
            mean_dp, center_qx, center_qy
        )

        if plot_type in ['average', 'both']:
            ax = axes[0]
            line, = (ax.semilogy(radii, profile_avg, label=label) if log_scale
                     else ax.plot(radii, profile_avg, label=label))
            #ax.fill_between(radii,
            #               np.clip(profile_avg - profile_std, 1e-4, None),
            #               profile_avg + profile_std,
            #               alpha=0.15, color=line.get_color())

        if plot_type in ['sum', 'both']:
            ax = axes[-1]
            if log_scale:
                ax.semilogy(radii, profile_sum, label=label)
            else:
                ax.plot(radii, profile_sum, label=label)

    for ax in axes:
        ax.set_xlabel('Radius (pixels)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    if plot_type in ['average', 'both']:
        axes[0].set_ylabel('Average intensity')
        axes[0].set_title('Radial profile (azimuthal average)')
    if plot_type in ['sum', 'both']:
        axes[-1].set_ylabel('Integrated intensity')
        axes[-1].set_title('Radial profile (azimuthal sum)')

    plt.tight_layout()
    plt.show()


# ---- Output ----

def save_denoised_h5(filepath, data):
    """Save denoised data as compressed HDF5 (float16 + shuffle + gzip)."""
    import h5py
    Qx, Qy = data.shape[-2:]
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(
            'data',
            data=data.astype(np.float16),
            compression='gzip',
            compression_opts=6,
            shuffle=True,
            chunks=(1, 1, Qx, Qy),
        )
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"Saved {filepath} ({size_mb:.1f} MB)")


def upsample_denoised(denoised_tensor, scale_factor=2):
    """GPU-accelerated bicubic upsampling after binned inference."""
    return F.interpolate(
        denoised_tensor, scale_factor=scale_factor,
        mode='bicubic', align_corners=False
    )
