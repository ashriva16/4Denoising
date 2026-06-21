import argparse
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

import core.models as models
from core.models import get_unet
from core.dataloader import build_training_dataset
from core.losses import build_loss_fn
from core.preprocessing import (
    bin_datacube, offset_datacube,
    detect_dead_pixels, correct_dead_pixels,
)
from utils.logger import log_results, save_checkpoint, setup_logging
from utils.opts import get_configuration
from core.io import load_4dstem
import logging

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_bf16 = (device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8)
dtype = torch.bfloat16 if use_bf16 else torch.float16


def load_data(cfg, config_path):
    filepath = Path(cfg.dataset.data_dir) / cfg.dataset.file
    if not filepath.is_absolute():
        filepath = Path.cwd() / filepath  # relative to where you run the command
    data, metadata = load_4dstem(filepath, crop_N=cfg.dataset.get('crop_N', None))

    # Preprocessing chain
    if cfg.dataset.get('bin_factor', 1) > 1:
        data = bin_datacube(data, cfg.dataset.bin_factor)

    if cfg.dataset.get('defect_mask', False):
        mask, stats = detect_dead_pixels(
            data,
            method=cfg.dataset.get('defect_method', 'combined'),
            threshold_factor=cfg.dataset.get('defect_sigma', 5),
            min_dead_fraction=cfg.dataset.get('defect_dead_fraction', 0.8),
            visualize=False,
        )
        data = correct_dead_pixels(
            data, mask,
            method=cfg.dataset.get('defect_correction', 'median_local'),
            visualize_sample=False,
        )

    if cfg.dataset.get('offset', 0) > 0:
        data = offset_datacube(data, cfg.dataset.offset)

    print(f"Data ready: {data.shape}, range [{data.min():.2f}, {data.max():.2f}]")
    return data


def load_model(cfg):
    """Build model with selected UNet backend."""
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    print(f"Model: {cfg.model.name}, UNet: {cfg.model.get('unet', 'original')}")
    return model, optimizer


def main(cfg, config_path: Path, cli_args):
    torch.manual_seed(cfg.train.seed)

    # DATA SETUP -------------------------------------------------------
    data = load_data(cfg, config_path)

    train_dataset = build_training_dataset(
        data,
        neighbor_mode=cfg.dataset.get('neighbor_mode', 'alternating_spatial'),
        image_size=cfg.dataset.get('image_size', None),
    )

    # Train/val split
    val_size = int(cfg.train.val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, valid_ds = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )
    n_workers = cfg.train.get('num_workers', 0)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0 and device.type == "cuda"),
        prefetch_factor=4 if n_workers > 0 else None,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # MODEL SETUP ------------------------------------------------------
    model, optimizer = load_model(cfg)
    #if device.type == "cuda":
    #    model = cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead"))
    model.name = cfg.model.name

    # LOSS FUNCTION ----------------------------------------------------
    loss_fn = build_loss_fn(
        cfg.train.get('loss', 'mse'),
        data=data, device=device,
    )

    # TRAINING SETUP ---------------------------------------------------
    scaler = GradScaler(enabled=device.type == "cuda")

    scheduler_type = cfg.train.get('scheduler', 'multistep')
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.num_epochs, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma
        )

    # LOGGING + RESUME -------------------------------------------------
    if cli_args.resume:
        resume_path = Path(cli_args.resume).parent.resolve()
        args_for_logger = argparse.Namespace(model=cfg.model.name, **cfg.train)
        args_for_logger.log_path = str(resume_path) + "/"

        # Load checkpoint FIRST
        ckpt = torch.load(cli_args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', float("inf"))

        # THEN set up logging
        handler = logging.FileHandler(str(resume_path / "train.log"), mode='a')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)-15s %(levelname)-8s %(name)s:%(lineno)d \n%(message)s\n"
        ))
        file_logger = logging.getLogger("4denoising.resume")
        file_logger.setLevel(logging.DEBUG)
        file_logger.addHandler(handler)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(resume_path / "tb"))

        logger = {"tb": writer, "file": file_logger}
        file_logger.info(f"=== Resumed from epoch {start_epoch} ===")

        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")
        print(f"Saving to: {args_for_logger.log_path}")
    else:
        # Fresh training — setup_logging creates new runN folder
        log_root = (Path.cwd() / getattr(cfg.output, 'save_dir', 'checkpoints')).resolve()
        log_root.mkdir(parents=True, exist_ok=True)
        args_for_logger = argparse.Namespace(model=cfg.model.name, **cfg.train)
        logger = setup_logging(args_for_logger, model, str(log_root) + "/")
        save_checkpoint(model, optimizer, scheduler, 0, args_for_logger.log_path, hparams=cfg)
        start_epoch = 0
        best_loss = float("inf")

    # BEGIN TRAINING ---------------------------------------------------
    for epoch in tqdm(range(start_epoch, cfg.train.num_epochs), desc="Epochs",
                      leave=True, dynamic_ncols=True):

        # Alternating neighbor mode
        if hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(epoch)

        model.train()
        train_loss_sum, train_count = 0, 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == "cuda"),
                          dtype=dtype):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_count += 1

        train_mean = train_loss_sum / max(train_count, 1)
        scheduler.step()

        # VALIDATION ---------------------------------------------------
        do_val = (((epoch + 1) % cfg.train.checkpoint == 0) or
                  ((epoch + 1) == cfg.train.num_epochs))

        if do_val:
            model.eval()
            val_loss_sum, val_count = 0, 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with autocast(device_type=device.type,
                                  enabled=(device.type == "cuda"), dtype=dtype):
                        outputs = model(inputs)
                    val_loss_sum += loss_fn(outputs, targets).item()
                    val_count += 1

            current_loss = val_loss_sum / max(val_count, 1)
            if current_loss < best_loss:
                best_loss = current_loss
                save_checkpoint(model, optimizer, scheduler, epoch + 1,
                                args_for_logger.log_path, best=True, hparams=cfg,
                                best_loss=best_loss)

            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            args_for_logger.log_path, hparams=cfg,
                            best_loss=best_loss)
            log_results(logger, {"train": train_mean, "validation": current_loss},
                        epoch + 1)
            logger["file"].info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    # Final save
    save_checkpoint(model, optimizer, scheduler, int(cfg.train.num_epochs),
                    args_for_logger.log_path, hparams=cfg)


if __name__ == "__main__":
    import argparse as ap
    import sys

    parser = ap.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    cli_args, remaining = parser.parse_known_args()

    # Remove --config from sys.argv so get_configuration doesn't choke
    sys.argv = [sys.argv[0]] + remaining

    config_path = Path(cli_args.config).resolve()
    config = get_configuration(config_path)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(config, config_path, cli_args)
