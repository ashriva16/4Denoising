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
from core.data_setup import load_training_data, uses_sources
from core.dataloader import build_multi_dataset, build_scan_mask, build_training_dataset
from core.losses import build_loss_fn
from utils.logger import log_results, save_checkpoint, setup_logging
from utils.opts import get_configuration
import logging

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_bf16 = (device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8)
dtype = torch.bfloat16 if use_bf16 else torch.float16


def build_dataset(cfg, cubes):
    """Single legacy file without a scan mask: the original dataset, unchanged.
    Otherwise: all sources pooled, each with its own scan mask."""
    neighbor_mode = cfg.dataset.get('neighbor_mode', 'alternating_spatial')
    image_size = cfg.dataset.get('image_size', None)
    scan_mask_cfg = cfg.dataset.get('scan_mask', None)
    if not uses_sources(cfg) and scan_mask_cfg is None:
        return build_training_dataset(cubes[0], neighbor_mode=neighbor_mode,
                                      image_size=image_size)
    sm = dict(scan_mask_cfg or {})
    mode = sm.pop('mode', 'all')
    masks = [build_scan_mask(c.shape[0], c.shape[1], mode=mode, **sm) for c in cubes]
    for c, m in zip(cubes, masks):
        print(f"  scan mask '{mode}': {int(m.sum())}/{m.size} targets on {c.shape[:2]}")
    return build_multi_dataset(cubes, masks=masks, neighbor_mode=neighbor_mode,
                               image_size=image_size)


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
    cubes, labels, data_infos = load_training_data(cfg)
    train_dataset = build_dataset(cfg, cubes)

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
    # weighted_poisson_nll uses the mean pattern; pooled over all sources
    pacbed_data = (cubes[0] if len(cubes) == 1
                   else np.stack([c.mean(axis=(0, 1)) for c in cubes])[None])
    loss_fn = build_loss_fn(
        cfg.train.get('loss', 'mse'),
        data=pacbed_data, device=device,
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

        for batch in train_loader:              # (inputs, targets[, source_id])
            inputs, targets = batch[0], batch[1]
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
                for batch in valid_loader:
                    inputs, targets = batch[0], batch[1]
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
