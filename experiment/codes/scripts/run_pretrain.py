import argparse
from pathlib import Path

import core.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from core.data import STEMDataSet as DataSet
from torch.amp import autocast
from utils.opts import get_configuration

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")

# definition for loading model from a pretrained network file
def load_model(model_path, Fast=False, parallel=False, pretrained=True, old=True, load_opt=False,
            mf2f=False):
    if not Fast:
        # Explicitly disable weights_only to allow loading checkpoints saved with argparse.Namespace
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**{**vars(ckpt["args"])})
        # ignore this
        if old:
            vars(args)['blind_noise'] = False

        model = models.build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        model = models.FastDVDnet(mf2f=mf2f)
        ckpt = None

    if load_opt and not Fast:
        for o, state in zip([optimizer], ckpt.get("optimizer", []), strict=False):
            o.load_state_dict(state)

    if pretrained:
        if Fast:
            state_dict = torch.load(model_path, weights_only=False, map_location="cpu")
        else:
            state_dict = ckpt["model"][0]
        own_state = model.state_dict()

        for name, param in state_dict.items():
            if parallel:
                name = name[7:]
            if Fast and not mf2f:
                name = name.split('.', 1)[1]
            if name not in own_state:
                print("not matching: ", name)
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    if not Fast:
        return model, optimizer, args
    else:
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

    codes_path = config_path.parent.parent.resolve() # returns location of codes directory

    filepath = config_path.parent / cfg.dataset.data_dir / cfg.dataset.file
    ds = DataSet(filepath, samplershape=cfg.dataset.samplershape)
    _, y0 = ds[0]
    _, _, H, W = y0.shape[0], y0.shape[1], y0.shape[-2], y0.shape[-1]

    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.validate.batch_size,
                                    shuffle=False, num_workers=0)
    # example = 0
    model_path = codes_path / "pretrained" / "fluoro_micro.pt"
    model, _, _ = load_model(model_path, parallel=True,
                                        pretrained=True, old=True, load_opt=False)
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

