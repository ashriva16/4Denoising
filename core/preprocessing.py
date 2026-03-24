"""
Preprocessing utilities for 4D-STEM denoising.

Functions:
    bin_datacube        — 2×2 spatial binning of DPs (sum, preserves Poisson stats). Can also handle larger binning
    offset_datacube     — +1 offset to eliminate binary sparsity artifacts. CAUTION: destroys raw signal from experiment
    remove_offset       — subtract offset + clip after inference
    correct_pnll_bias   — rescale denoised data to match original total counts
    build_defect_mask   — identify dead/hot pixel clusters from PACBED
    inpaint_defects     — nearest-neighbor inpainting of masked pixels
    radial_profile      — azimuthally-integrated radial profile
    compare_radial_profiles — multi-datacube radial profile comparison
    save_denoised_h5    — compressed HDF5 output (float16 + shuffle + gzip)
    upsample_denoised   — bicubic upsampling on GPU after binned inference
"""

import numpy as np
import torch
import torch.nn.functional as F


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

def build_defect_mask(data, sigma_threshold=5, dilate_radius=1):
    """Identify dead/hot pixel clusters from PACBED."""
    from scipy.ndimage import median_filter, binary_dilation, generate_binary_structure

    pacbed = data.mean(axis=(0, 1))
    local_median = median_filter(pacbed, size=7)
    local_std = median_filter(np.abs(pacbed - local_median), size=7) * 1.4826
    local_std = np.maximum(local_std, 1e-6)

    deviation = np.abs(pacbed - local_median) / local_std
    mask = deviation > sigma_threshold

    if dilate_radius > 0:
        struct = generate_binary_structure(2, 1)
        for _ in range(dilate_radius):
            mask = binary_dilation(mask, structure=struct)

    print(f"Defect mask: {mask.sum()} pixels ({100 * mask.sum() / mask.size:.2f}%)")
    return mask


def inpaint_defects(data, mask):
    """Nearest-valid-pixel inpainting for dead pixel clusters."""
    from scipy.ndimage import distance_transform_edt

    cleaned = data.copy().astype(np.float32)
    if not mask.any():
        return cleaned

    _, nearest_indices = distance_transform_edt(mask, return_distances=True, return_indices=True)

    Rx, Ry = data.shape[:2]
    for rx in range(Rx):
        for ry in range(Ry):
            dp = data[rx, ry].astype(np.float32)
            fill_values = dp[nearest_indices[0], nearest_indices[1]]
            cleaned[rx, ry] = np.where(mask, fill_values, dp)

    print(f"Inpainted {mask.sum()} pixels per DP across {Rx * Ry} patterns")
    return cleaned


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
