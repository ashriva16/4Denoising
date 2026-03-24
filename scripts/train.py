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
    build_defect_mask, inpaint_defects,
)
from utils.logger import log_results, save_checkpoint, setup_logging
from utils.opts import get_configuration

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_bf16 = (device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8)
dtype = torch.bfloat16 if use_bf16 else torch.float16


def load_data(cfg, config_path):
    filepath = config_path.parent / cfg.dataset.data_dir / cfg.dataset.file
    data, metadata = load_4dstem(filepath, crop_N=cfg.dataset.get('crop_N', None))

    # Preprocessing chain
    if cfg.dataset.get('bin_factor', 1) > 1:
        data = bin_datacube(data, cfg.dataset.bin_factor)

    if cfg.dataset.get('defect_mask', False):
        mask = build_defect_mask(data, sigma_threshold=cfg.dataset.get('defect_sigma', 5))
        data = inpaint_defects(data, mask)

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


def main(cfg, config_path: Path):
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

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.get('num_workers', 4),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
        prefetch_factor=4 if device.type == "cuda" else None,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # MODEL SETUP ------------------------------------------------------
    model, optimizer = load_model(cfg)
    if device.type == "cuda":
        model = cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead"))
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

    # LOGGING ----------------------------------------------------------
    log_root = (config_path.parent / cfg.output.save_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    args_for_logger = argparse.Namespace(model=cfg.model.name, **cfg.train)
    logger = setup_logging(args_for_logger, model, str(log_root) + "/")
    save_checkpoint(model, optimizer, scheduler, 0, args_for_logger.log_path, hparams=cfg)

    # BEGIN TRAINING ---------------------------------------------------
    best_loss = float("inf")

    for epoch in tqdm(range(cfg.train.num_epochs), desc="Epochs",
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
                                args_for_logger.log_path, best=True, hparams=cfg)

            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            args_for_logger.log_path, hparams=cfg)
            log_results(logger, {"train": train_mean, "validation": current_loss},
                        epoch + 1)
            logger["file"].info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    # Final save
    save_checkpoint(model, optimizer, scheduler, int(cfg.train.num_epochs),
                    args_for_logger.log_path, hparams=cfg)


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yml").resolve()
    config = get_configuration(config_path)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(config, config_path)
