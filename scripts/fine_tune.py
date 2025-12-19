import argparse
import warnings
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

import core.models as models
from core.data import STEMDataSet as DataSet
from utils.logger import log_results, save_checkpoint, setup_logging
from utils.opts import get_configuration

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_bf16 = (device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8)
dtype = torch.bfloat16 if use_bf16 else torch.float16


def load_model(model_path, device, parallel=True):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Build the exact architecture the checkpoint expects
    args = argparse.Namespace(**vars(ckpt["args"]))

    # Backward-compat patch (your earlier crash)
    if not hasattr(args, "blind_noise"):
        args.blind_noise = False

    model = models.build_model(args).to(device)

    # Load weights
    state_dict = ckpt["model"][0]
    if parallel and next(iter(state_dict)).startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return model


# ------------------------------
# fine-tune helpers
# ------------------------------
def freeze_all(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_by_substring(model: torch.nn.Module, patterns) -> None:
    for n, p in model.named_parameters():
        if any(s in n for s in patterns):
            p.requires_grad = True


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(cfg, config_path: Path):
    torch.manual_seed(cfg.train.seed)

    # ---------------- DATA ----------------
    filepath = config_path.parent / cfg.dataset.data_dir / cfg.dataset.file
    ds = DataSet(filepath, samplershape=cfg.dataset.samplershape)
    ds = torch.utils.data.Subset(ds, range(10))  # only for debugging

    val_size = int(cfg.train.val_split * len(ds))
    train_size = len(ds) - val_size
    train_set, valid_set = torch.utils.data.random_split(
        ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
        prefetch_factor=4,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ---------------- MODEL (fine-tune) ----------------
    codes_path = config_path.parent.parent.resolve() # returns location of codes directory
    model_path = codes_path / "pretrained" / "fluoro_micro.pt"
    print(f"Using pretrained: {model_path}")

    # parallel=True because fluoro_micro.pt was saved with module.* keys in your run_pretrain usage
    model = load_model(model_path, device=device, parallel=True)
    if device.type == "cuda":
        model = cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead"))
    model.name = cfg.model.name

    # Freeze policy (no architecture change; you can tune fewer params to speed up)
    # If you want full fine-tune, set patterns=None and do not freeze.
    finetune_patterns = getattr(cfg.train, "finetune_patterns", None)
    if finetune_patterns:
        freeze_all(model)
        unfreeze_by_substring(model, finetune_patterns)
        print("Trainable params:", count_trainable(model))
    else:
        print("Trainable params (full):", sum(p.numel() for p in model.parameters()))

    # Training Setup -----------------------------------------------------------
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                lr=cfg.train.lr)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma
    )

    # ---------------- LOGGING / CKPT ----------------
    log_root = (config_path.parent / cfg.output.save_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    args_for_logger = argparse.Namespace(model=cfg.model.name, **cfg.train)
    logger = setup_logging(args_for_logger, model, str(log_root) + "/")
    save_checkpoint(model, optimizer, scheduler, 0, args_for_logger.log_path, hparams=cfg)

    # ---------------- TRAIN ----------------
    best_loss = float("inf")
    for epoch in tqdm(range(cfg.train.num_epochs), desc="Epochs", leave=True, dynamic_ncols=True):

        model.train()
        train_loss_sum, n_train = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda"), dtype=dtype):
                outputs, _ = model(inputs)
                loss = F.mse_loss(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.detach().cpu())
            n_train += 1

        scheduler.step()
        train_loss = train_loss_sum / max(n_train, 1)

        # ---------------- VALIDATE (every checkpoint epochs) ----------------
        do_val = ((epoch + 1) % cfg.train.checkpoint == 0) or ((epoch + 1) == cfg.train.num_epochs)
        if do_val:

            model.eval()
            val_loss_sum, n_val = 0.0, 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    with autocast(device_type=device.type, enabled=(device.type == "cuda"),
                                dtype=dtype):
                        outputs, _ = model(inputs)
                        vloss = F.mse_loss(outputs, targets)

                    val_loss_sum += float(vloss.detach().cpu())
                    n_val += 1

            val_loss = val_loss_sum / max(n_val, 1)

            # checkpointing
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, args_for_logger.log_path,
                    best=True, hparams=cfg
                )

            save_checkpoint(model, optimizer, scheduler, epoch + 1, args_for_logger.log_path,
                            hparams=cfg)

            # logging
            log_results(logger, {"train": train_loss, "validation": val_loss}, epoch + 1)
            logger["file"].info(
                f"Epoch {epoch+1} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )
            logger["file"].info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        else:
            logger["file"].info(f"Epoch {epoch+1} | train_loss={train_loss:.6f}")

    # final save
    save_checkpoint(model, optimizer, scheduler, cfg.train.num_epochs, args_for_logger.log_path,
                    hparams=cfg)


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yml").resolve()
    cfg = get_configuration(config_path)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(cfg, config_path)
