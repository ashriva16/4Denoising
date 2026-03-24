"""
infer.py — Apply a trained UDVD-MF model to denoise a 4D-STEM dataset.

Usage:
    python infer.py --config config.yml --checkpoint checkpoints/exp01/best.pth
    python infer.py --config config.yml --checkpoint checkpoints/exp01/best.pth \
                    --neighbor-mode spatial --upsample 2 --output results/denoised.h5
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import core.models as models
from core.models import get_unet
from core.dataloader import DataSetFromArray
from core.preprocessing import (
    bin_datacube, offset_datacube, remove_offset,
    correct_pnll_bias, build_defect_mask, inpaint_defects,
    save_denoised_h5,
)
from utils.opts import get_configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(cfg, config_path):
    """Load and preprocess data (same pipeline as training)."""
    filepath = config_path.parent / cfg.dataset.data_dir / cfg.dataset.file
    data, metadata = load_4dstem(filepath, crop_N=cfg.dataset.get('crop_N', None))

    # Must match training preprocessing exactly
    if cfg.dataset.get('bin_factor', 1) > 1:
        data = bin_datacube(data, cfg.dataset.bin_factor)

    if cfg.dataset.get('defect_mask', False):
        mask = build_defect_mask(data, sigma_threshold=cfg.dataset.get('defect_sigma', 5))
        data = inpaint_defects(data, mask)

    if cfg.dataset.get('offset', 0) > 0:
        data = offset_datacube(data, cfg.dataset.offset)

    print(f"Data ready: {data.shape}, range [{data.min():.2f}, {data.max():.2f}]")
    return data


def load_model_from_checkpoint(checkpoint_path, cfg):
    """Load model architecture from config and weights from checkpoint."""
    unet_cls = get_unet(cfg.model.get('unet', 'original'))

    args = argparse.Namespace(
        model=cfg.model.name,
        channels=cfg.model.channels,
        out_channels=cfg.model.out_channels,
        bias=cfg.model.bias,
        normal=cfg.model.normal,
        blind_noise=cfg.model.blind_noise,
        unet_cls=unet_cls,
    )
    model = models.build_model(args).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('loss', '?')
    neighbor_mode = checkpoint.get('neighbor_mode', 'unknown')
    print(f"Loaded model: {cfg.model.name} (UNet: {cfg.model.get('unet', 'original')})")
    print(f"  Checkpoint: epoch {epoch}, loss={loss}")
    print(f"  Trained with: {neighbor_mode}")

    return model, checkpoint


def run_inference(model, data, cfg, upsample_factor=1):
    """
    Denoise all scan positions.

    Parameters
    ----------
    model : nn.Module
    data : np.ndarray (Rx, Ry, Qx, Qy) — preprocessed (binned, offset, etc.)
    cfg : config namespace
    upsample_factor : int — bicubic upsampling after denoising (1 = none)

    Returns
    -------
    denoised : np.ndarray
    """
    neighbor_mode = cfg.infer.get('neighbor_mode', 'spatial')
    inference_dataset = DataSetFromArray(data, neighbor_mode=neighbor_mode)

    Rx = inference_dataset.Rx()
    Ry = inference_dataset.Ry()
    Qx = inference_dataset.Qx()
    Qy = inference_dataset.Qy()

    Qx_out = Qx * upsample_factor
    Qy_out = Qy * upsample_factor
    denoised = np.zeros((Rx, Ry, Qx_out, Qy_out), dtype=np.float32)

    margin = 1  # double-arrow temporal reaches ±2, but information loss in minimal with margin=1
    total = (Rx - 2 * margin) * (Ry - 2 * margin)
    processed = 0
    t0 = time.time()

    print(f"\nDenoising {total} positions (mode={neighbor_mode}, "
          f"upsample={upsample_factor}x)...")

    with torch.no_grad():
        for rx in tqdm(range(margin, Rx - margin), desc="Denoising"):
            for ry in range(margin, Ry - margin):
                item_input, _ = inference_dataset.getitem([rx, ry])
                model_input = item_input.unsqueeze(0).float().to(device)

                output = model(model_input)

                if upsample_factor > 1:
                    output = F.interpolate(
                        output, scale_factor=upsample_factor,
                        mode='bicubic', align_corners=False,
                    )

                denoised[rx, ry] = output.squeeze().cpu().numpy()

                processed += 1
                if processed % 2000 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    eta = (total - processed) / rate
                    tqdm.write(f"  {processed}/{total} — {rate:.0f} pos/s, "
                               f"ETA {eta:.0f}s")

    # Fill edges by copying nearest valid row/column
    denoised[0, :]  = denoised[1, :]
    denoised[-1, :] = denoised[-2, :]
    denoised[:, 0]  = denoised[:, 1]
    denoised[:, -1] = denoised[:, -2]

    elapsed = time.time() - t0
    print(f"Inference complete: {processed} positions in {elapsed:.1f}s "
          f"({processed/elapsed:.0f} pos/s)")

    return denoised


def postprocess(denoised, original_data, cfg):
    """
    Apply post-processing: remove offset, correct PNLL bias.

    Parameters
    ----------
    denoised : np.ndarray — raw model output
    original_data : np.ndarray — data BEFORE offset (for bias correction)
    cfg : config namespace

    Returns
    -------
    corrected : np.ndarray
    """
    result = denoised.copy()

    # Remove +1 offset if it was applied during training
    offset = cfg.dataset.get('offset', 0)
    if offset > 0:
        result = remove_offset(result, offset)
        print(f"Removed +{offset} offset")

    # Correct Poisson NLL systematic bias
    loss_type = cfg.train.get('loss', 'mse')
    if loss_type in ['poisson_nll', 'weighted_poisson_nll']:
        # Need the original data WITHOUT offset for comparison
        result = correct_pnll_bias(original_data, result)

    result = np.clip(result, 0, None)  # ensure non-negative

    print(f"Post-processed: range [{result.min():.4f}, {result.max():.4f}], "
          f"mean={result.mean():.4f}")
    return result


def main(args):
    # Load config from training
    config_path = Path(args.config).resolve()
    cfg = get_configuration(config_path)

    # Override inference-specific settings from CLI
    if not hasattr(cfg, 'infer'):
        cfg.infer = argparse.Namespace()
    if args.neighbor_mode:
        cfg.infer.neighbor_mode = args.neighbor_mode
    elif not hasattr(cfg.infer, 'neighbor_mode'):
        cfg.infer.neighbor_mode = 'spatial'

    # Load and preprocess data (same pipeline as training)
    data = load_data(cfg, config_path)

    # Also load the data BEFORE offset for bias correction
    original_data = data.copy()
    offset = cfg.dataset.get('offset', 0)
    if offset > 0:
        original_data = original_data - offset  # undo offset for bias reference

    # Load model
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, cfg)

    # Run inference
    upsample = args.upsample or cfg.infer.get('upsample', 1)
    denoised_raw = run_inference(model, data, cfg, upsample_factor=upsample)

    # Post-process
    denoised = postprocess(denoised_raw, original_data, cfg)

    # Save
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if output_path.endswith(('.h5', '.hdf5')):
        save_denoised_h5(output_path, denoised)
    else:
        np.save(output_path, denoised)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"Saved {output_path} ({size_mb:.1f} MB)")

    # Summary
    print(f"\n{'='*60}")
    print(f"INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:          {cfg.model.name} (UNet: {cfg.model.get('unet', 'original')})")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Neighbor mode:  {cfg.infer.neighbor_mode}")
    print(f"  Loss (train):   {cfg.train.get('loss', 'mse')}")
    print(f"  Bin factor:     {cfg.dataset.get('bin_factor', 1)}")
    print(f"  Offset:         {cfg.dataset.get('offset', 0)}")
    print(f"  Upsample:       {upsample}x")
    print(f"  Output:         {output_path}")
    print(f"  Output shape:   {denoised.shape}")
    print(f"  Output range:   [{denoised.min():.4f}, {denoised.max():.4f}]")
    print(f"{'='*60}")


def get_args():
    p = argparse.ArgumentParser(description="UDVD-MF 4D-STEM inference")

    p.add_argument("--config", required=True,
                   help="Path to config.yml (same one used for training)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--output", default="results/denoised.h5",
                   help="Output path (.h5 for compressed HDF5, .npy for numpy)")
    p.add_argument("--neighbor-mode", default=None, choices=["spatial", "temporal"],
                   help="Override inference neighbor mode (default: from config)")
    p.add_argument("--upsample", type=int, default=None,
                   help="Bicubic upsample factor after denoising (e.g. 2 to undo 2×2 binning)")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
