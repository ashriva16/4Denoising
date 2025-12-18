import argparse
import warnings
from pathlib import Path

import core.metrics as metrics
import core.models as models
import torch
import torch.nn.functional as F
from core.data import STEMDataSet as DataSet
from utils.logger import log_results, save_checkpoint, setup_logging
from utils.opts import get_configuration

warnings.filterwarnings("ignore")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_model(cfg):
    args = argparse.Namespace(
        model=cfg.model.name,
        channels=cfg.model.channels,
        out_channels=cfg.model.out_channels,
        bias=cfg.model.bias,
        normal=cfg.model.normal,
        blind_noise=cfg.model.blind_noise,
    )
    model = models.build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    return model, optimizer


def main(cfg, config_path: Path):
    torch.manual_seed(cfg.train.seed)

    # DATA SETUP ------------------------------------------------------
    project_root = (config_path.parent / cfg.dataset.base_dir).resolve()
    filepath = project_root / cfg.dataset.file
    ds = DataSet(filepath, samplershape=cfg.dataset.samplershape)

    val_size = int(cfg.train.val_split * len(ds))
    train_size = len(ds) - val_size
    train, valid = torch.utils.data.random_split(
        ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0
    )


    # Model and Training Setup ---------------------------------------------
    print(device, cfg)
    model, optimizer = load_model(cfg)
    model.name = cfg.model.name  # enforce name from config for logging
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma
    )

    # Logging / checkpoint setup
    log_root = (config_path.parent / cfg.output.save_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    args_for_logger = argparse.Namespace(model=cfg.model.name, **cfg.train)
    logger = setup_logging(args_for_logger, model, str(log_root) + "/")
    save_checkpoint(model, optimizer, scheduler, 0, args_for_logger.log_path, hparams=cfg)

    # Begin training -------------------------------------------------------
    best_loss = float("inf")
    train_meters = {"train_loss": [], "train_psnr": [], "train_ssim": []}
    valid_meters = {"valid_psnr": [], "valid_ssim": []}

    for epoch in range(cfg.train.num_epochs):
        loss_avg, psnr_avg, ssim_avg, count = 0, 0, 0, 0
        for inputs, ground_truths in train_loader:
            model.train()
            inputs = inputs.to(device)
            ground_truths = ground_truths.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = F.mse_loss(outputs, ground_truths) / cfg.train.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_psnr = metrics.psnr(ground_truths, outputs, False)
            train_ssim = metrics.ssim(ground_truths, outputs, False)
            train_meters["train_loss"].append(loss.item())
            train_meters["train_psnr"].append(train_psnr.item())
            train_meters["train_ssim"].append(train_ssim.item())

            loss_avg += loss.item()
            psnr_avg += train_psnr.item()
            ssim_avg += train_ssim.item()
            count += 1

        scheduler.step()

        # ------- Validating at certain intervals -------------------------
        if (epoch + 1) % cfg.train.checkpoint != 0:
            continue

        model.eval()
        loss_avg, psnr_avg, ssim_avg, count = 0, 0, 0, 0
        for inputs, ground_truths in valid_loader:
            with torch.no_grad():
                sample = inputs.to(device)
                outputs = model(sample)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                valid_psnr = metrics.psnr(ground_truths, outputs, False)
                valid_ssim = metrics.ssim(ground_truths, outputs, False)
                valid_meters["valid_psnr"].append(valid_psnr.item())
                valid_meters["valid_ssim"].append(valid_ssim.item())

                loss_avg += F.mse_loss(outputs, ground_truths) / cfg.train.batch_size
                psnr_avg += valid_psnr.item()
                ssim_avg += valid_ssim.item()
                count += 1

        current_loss = loss_avg / max(count, 1)
        if current_loss < best_loss:
            best_loss = current_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, args_for_logger.log_path, best=True, hparams=cfg)

        # save checkpoint each validation
        save_checkpoint(model, optimizer, scheduler, epoch + 1, args_for_logger.log_path, hparams=cfg)
        log_results(logger, {"train": loss_avg, "validation": current_loss}, epoch + 1)
        logger["file"].info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    # Final save of best checkpoint path already handled; also save latest alias
    save_dir = (config_path.parent / cfg.output.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg}, save_dir / cfg.output.filename)


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yml").resolve()
    config = get_configuration(config_path)
    main(config, config_path)
