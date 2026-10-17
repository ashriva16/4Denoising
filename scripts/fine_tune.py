"""
fine_tune.py - continue training from existing weights on new data.

    python -m scripts.fine_tune --config configs/config_sources.yml --weights path/to/checkpoint.pth

Same as ``python -m scripts.train ... --weights <path>``: the weights are loaded
(strictly, into the model the config describes), the optimizer and scheduler
start fresh, and the run's provenance records where the weights came from.
Before training starts, the weights' recorded preprocessing (detector/scan binning,
offset) is compared with the new data; a mismatch stops the run unless
``--domain-policy warn`` (or ``off``) is given, in which case the differences are
recorded in the new run's provenance.

To train only part of the network, list name fragments of the parameters to keep
trainable in the config, e.g.::

    train:
      trainable: [nin_A, nin_B, nin_C]     # everything else is frozen

(This file replaces an earlier fine-tuning script that depended on a data module
no longer in the repository.)
"""

import argparse
import sys
from pathlib import Path

import torch

from scripts.train import device, main
from utils.opts import get_configuration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml")
    parser.add_argument("--weights", type=str, required=True, help="checkpoint to start from")
    parser.add_argument("--domain-policy", choices=["strict", "warn", "off"], default=None)
    cli_args, remaining = parser.parse_known_args()
    cli_args.resume = None
    sys.argv = [sys.argv[0]] + remaining

    config_path = Path(cli_args.config).resolve()
    config = get_configuration(config_path)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(config, config_path, cli_args)
