import argparse

import torch
import torch.nn as nn

from experiment.codes.scripts.core import STEMDataset


# definition for loading model from a pretrained network file
def load_pretrained_udvd(PATH, models, parallel=False, old=True, device="cpu"):
    ckpt = torch.load(PATH, map_location="cpu")

    args = argparse.Namespace(**vars(ckpt["args"]))
    if old:
        args.blind_noise = False  # compatibility

    model = models.build_model(args).to(device).eval()

    state_dict = ckpt["model"][0]          # pretrained weights
    own_state = model.state_dict()

    for name, param in state_dict.items():
        if parallel and name.startswith("module."):
            name = name[7:]                # strip "module."
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)

    return model


if __name__ == "__main__":

    filepath = r"../../data/raw/4D-STEM_data_for_theophylline/20220711_182642_data_150kX_binned2.hdf5"
    ds = STEMDataset(filepath, T=5)
    model, optimizer, args = load_pretrained_udvd(PATH, parallel=parallel, pretrained=pretrained, old=old, load_opt=load_opt)
