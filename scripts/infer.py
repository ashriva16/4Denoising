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
    correct_pnll_bias, detect_dead_pixels, correct_dead_pixels,
    save_denoised_h5,
)
from utils.opts import get_configuration
from core.io import load_4dstem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(cfg, config_path):
    """Load and preprocess data (same pipeline as training)."""
    filepath = Path(cfg.dataset.data_dir) / cfg.dataset.file
    if not filepath.is_absolute():
        filepath = Path.cwd() / filepath  # relative to where you run the command 
    data, metadata = load_4dstem(filepath, crop_N=getattr(cfg.dataset,'crop_N', None))

    # Must match training preprocessing exactly
    if getattr(cfg.dataset,'bin_factor', 1) > 1:
        data = bin_datacube(data, cfg.dataset.bin_factor)

    if getattr(cfg.dataset,'defect_mask', False):
        mask, stats = detect_dead_pixels(
            data,
            method=getattr(cfg.dataset,'defect_method', 'combined'),
            threshold_factor=getattr(cfg.dataset,'defect_sigma', 5),
            min_dead_fraction=getattr(cfg.dataset,'defect_dead_fraction', 0.8),
            visualize=False,
        )
        data = correct_dead_pixels(
            data, mask,
            method=getattr(cfg.dataset,'defect_correction', 'median_local'),
            visualize_sample=False,
        )

    if getattr(cfg.dataset,'offset', 0) > 0:
        data = offset_datacube(data, cfg.dataset.offset)

    print(f"Data ready: {data.shape}, range [{data.min():.2f}, {data.max():.2f}]")
    return data


def load_model_from_checkpoint(checkpoint_path, cfg):
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and any('weight' in k for k in checkpoint):
        state_dict = checkpoint  # raw state_dict
    else:
        raise KeyError(f"Cannot find model weights in checkpoint. "
                       f"Keys: {list(checkpoint.keys())}")

    model.load_state_dict(state_dict)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('loss', checkpoint.get('val_loss', '?'))
    print(f"Loaded model: {cfg.model.name} (UNet: {cfg.model.get('unet', 'original')})")
    print(f"  Checkpoint: epoch {epoch}")

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
    neighbor_mode = getattr(cfg.infer, 'neighbor_mode', 'spatial')
    inference_dataset = DataSetFromArray(data, neighbor_mode=neighbor_mode)
    upsample_factor = cfg.dataset.get('upsample_factor', 1)

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
    offset = getattr(cfg.dataset,'offset', 0)
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


def main(cli_args, cfg, config_path):
    # Load and preprocess data (same pipeline as training)
    data = load_data(cfg, config_path)

    # Also load the data BEFORE offset for bias correction
    original_data = data.copy()
    offset = getattr(cfg.dataset,'offset', 0)
    if offset > 0:
        original_data = original_data - offset

    # Override inference settings from CLI
    if not hasattr(cfg, 'infer'):
        cfg.infer = argparse.Namespace()
    if cli_args.neighbor_mode:
        cfg.infer.neighbor_mode = cli_args.neighbor_mode
    elif not hasattr(cfg.infer, 'neighbor_mode'):
        cfg.infer.neighbor_mode = 'spatial'

    # Load model
    model, checkpoint = load_model_from_checkpoint(cli_args.checkpoint, cfg)

    # Run inference
    upsample = cli_args.upsample or getattr(cfg.infer, 'upsample_factor', 1) or 1
    denoised_raw = run_inference(model, data, cfg, upsample_factor=upsample)

    # Post-process
    denoised = postprocess(denoised_raw, original_data, cfg)

    # Save
    output_path = cli_args.output
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
    print(f"  Checkpoint:     {cli_args.checkpoint}")
    print(f"  Neighbor mode:  {cfg.infer.neighbor_mode}")
    print(f"  Loss (train):   {cfg.train.get('loss', 'mse')}")
    print(f"  Bin factor:     {getattr(cfg.dataset,'bin_factor', 1)}")
    print(f"  Offset:         {getattr(cfg.dataset,'offset', 0)}")
    print(f"  Upsample:       {upsample}x")
    print(f"  Output:         {output_path}")
    print(f"  Output shape:   {denoised.shape}")
    print(f"  Output range:   [{denoised.min():.4f}, {denoised.max():.4f}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/denoised.h5")
    parser.add_argument("--neighbor-mode", default=None, choices=["spatial", "temporal"])
    parser.add_argument("--upsample", type=int, default=None)
    cli_args, remaining = parser.parse_known_args()

    # Clear sys.argv so get_configuration doesn't choke
    sys.argv = [sys.argv[0]] + remaining

    config_path = Path(cli_args.config).resolve()
    cfg = get_configuration(config_path)

    # Pass CLI args into main
    cli_args.config = str(config_path)
    main(cli_args, cfg, config_path)
