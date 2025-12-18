import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../experiment/codes

import core.models as models
from core.data import STEMDataSet as DataSet
from utils.opts import get_configuration

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")

def load_model(cfg, config_path: Path):
    args = argparse.Namespace(
        model=cfg.model.name,
        channels=cfg.model.channels,
        out_channels=cfg.model.out_channels,
        bias=cfg.model.bias,
        normal=cfg.model.normal,
        blind_noise=cfg.model.blind_noise,
    )
    model = models.build_model(args).to(device)
    model_path = (config_path.parent / cfg.pre_train_path).resolve()
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt['model'])
    return model

def plot_one(gt_img, pred_img, output_dir, name="run_pretrained.png"):
    # gt_img: (H,W) numpy
    # pred_img: (H,W) numpy

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axes[0].imshow(gt_img, cmap="magma")
    axes[0].set_title("Ground Truth")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_img, cmap="magma")
    axes[1].set_title("Prediction")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

def main(cfg, config_path: Path):

    filepath = config_path.parent / cfg.dataset.data_dir / cfg.dataset.file
    ds = DataSet(filepath, samplershape=cfg.dataset.samplershape)
    _, y0 = ds[0]
    _, _, H, W = y0.shape[0], y0.shape[1], y0.shape[-2], y0.shape[-1]

    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.validate.batch_size,
                                    shuffle=False, num_workers=0)
    # example = 0
    model = load_model(cfg, config_path)
    model = model.to(device).eval()

    total_eval = cfg.validate.batch_size*cfg.validate.max_batch
    preds = np.zeros((total_eval, H, W), dtype=np.float32)
    gts   = np.zeros((total_eval, H, W), dtype=np.float32)
    for batch_indx, (x, y) in enumerate(dl):

        if batch_indx >= cfg.validate.max_batch:
            break

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            if use_amp and device == "cuda":
                with autocast(device_type="cuda", dtype=torch.float16):
                    y_hat, _ = model(x)
            else:
                y_hat, _ = model(x)

        # y_hat shape is (B, 1, H, W)
        # y shape is (B, H, W)
        pred = y_hat.detach().cpu()[:, 0].float().numpy()
        gt = y.detach().cpu().float().numpy()

        b = pred.shape[0]
        preds[batch_indx*b:(batch_indx+1)*b] = pred
        gts[batch_indx*b:(batch_indx+1)*b]   = gt

    output_dir = config_path.parent / cfg.output.save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "preds.npy", preds)
    np.save(output_dir / "gts.npy", gts)

    print("Saved:", output_dir / "preds.npy")
    print("Saved:", output_dir / "gts.npy")

    print("Plotting data")
    for i in range(total_eval):

        rx, ry = ds.index_to_RxRy(i)

        plot_one(
            gts[i],
            preds[i],
            output_dir,
            name=f"Rx{rx:04d}_Ry{ry:04d}.png"
        )


if __name__ == "__main__":

    config_path = Path(__file__).with_name("config.yml").resolve()
    config = get_configuration(config_path)
    main(config, config_path)

