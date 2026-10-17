"""
infer.py - denoise datasets with trained weights, block by block, in GPU batches.

    python -m scripts.infer --config cfg.yml --checkpoint ckpt.pth --output results/A.h5 --source A
    python -m scripts.infer --config cfg.yml --checkpoint ckpt.pth --output results/ --source all

Data are read in blocks of scan rows (core.stream) and denoised in batches
(core.inference), so memory is bounded by the block, not by the dataset.

Output
    .h5 / .hdf5   dataset 'data' (float16 by default) plus 'intensity_image' and
                  'mean_pattern'; full provenance in the 'provenance' attribute.
                  Resumable: an interrupted run continues where it stopped.
    .npy          memory-mapped float32, provenance in <output>.provenance.json
    a directory   with --source all: one <label>.h5 per dataset

Preprocessing is exactly the training one (core.data_setup); weights are checked
against the data (core.checkpoints, --domain-policy) and every difference is
recorded in the output's provenance.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

import core.models as models
from core.checkpoints import (check_compatible, extract_state_dict, file_digest,
                              resolve_policy, runtime_info)
from core.data_setup import (finish_info, load_training_data, mask_correction, source_specs,
                             uses_sources)
from core.inference import ArrayWriter, H5Writer, NpyWriter, denoise_source
from core.models import get_unet
from core.readers import ArraySource, open_source
from core.stream import dataset_info, plan_binned
from utils.opts import get_configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(checkpoint_path, cfg):
    args = argparse.Namespace(
        model=cfg.model.name, channels=cfg.model.channels,
        out_channels=cfg.model.out_channels, bias=cfg.model.bias,
        normal=cfg.model.normal, blind_noise=cfg.model.blind_noise,
        unet_cls=get_unet(cfg.model.get('unet', 'original')))
    model = models.build_model(args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(extract_state_dict(checkpoint))     # any checkpoint layout
    model.eval()
    print(f"Loaded model: {cfg.model.name} (UNet: {cfg.model.get('unet', 'original')}), "
          f"checkpoint epoch {checkpoint.get('epoch', '?')}")
    return model, checkpoint


# ---------------------------------------------------------------------------
# what to denoise
# ---------------------------------------------------------------------------
def _setting(cli_value, cfg, key, default):
    """CLI, then infer.<key>, then the default."""
    if cli_value is not None:
        return cli_value
    section = cfg.get('infer', None) or {}
    return section.get(key, default) if hasattr(section, 'get') else default


def _jobs(cfg, which):
    """List of jobs: dict(label, open=callable -> FrameSource, read=kwargs, info=dict)."""
    offset = cfg.dataset.get('offset', 0) or 0
    if uses_sources(cfg):
        correct_fn, mask_prov = mask_correction(cfg)
        jobs = []
        for spec in source_specs(cfg):
            read = dict(scan_shape=spec["scan_shape"], crop_y=spec["crop_y"],
                        crop_x=spec["crop_x"], q_bin=spec["q_bin"], r_bin=spec["r_bin"],
                        q_crop=spec["q_crop"], correct_fn=correct_fn)
            jobs.append({"label": spec["label"], "spec": spec, "read": read,
                         "mask_prov": mask_prov,
                         "open": (lambda s=spec: open_source(s["path"], fmt=s["fmt"],
                                                             **s["reader_options"]))})
    else:
        # legacy single file: preprocess in memory as before, then stream the array
        (cube,), (label,), (info,) = load_training_data(cfg)
        array = cube - offset if offset else cube              # engine adds the offset
        jobs = [{"label": label, "info": info, "read": {},
                 "open": (lambda a=array: ArraySource(a))}]
    labels = [j["label"] for j in jobs]
    if str(which) == "all":
        return jobs
    if isinstance(which, str) and not which.isdigit():
        if which not in labels:
            raise KeyError(f"source '{which}' not in {labels}")
        return [jobs[labels.index(which)]]
    return [jobs[int(which)]]


def _safe_name(label):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_") or "dataset"


def _writer(output, out_dtype):
    out = Path(output)
    if out.suffix.lower() in (".h5", ".hdf5"):
        return H5Writer(out, dtype=np.float16 if out_dtype == "auto" else out_dtype)
    if out.suffix.lower() == ".npy":
        return NpyWriter(out, dtype=np.float32 if out_dtype == "auto" else out_dtype)
    raise ValueError(f"output must end in .h5, .hdf5 or .npy (or be a directory with "
                     f"--source all), got {output}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(cli_args, cfg, config_path):
    get = lambda name, default=None: getattr(cli_args, name, default)   # noqa: E731
    neighbor_mode = _setting(get('neighbor_mode'), cfg, 'neighbor_mode', 'spatial')
    upsample = int(get('upsample') or _setting(None, cfg, 'upsample_factor', None)
                   or cfg.dataset.get('upsample_factor', 1) or 1)
    batch_size = int(_setting(get('batch_size'), cfg, 'batch_size', 16))
    block_rows = int(_setting(get('block_rows'), cfg, 'block_rows', 8))
    edges = _setting(get('edges'), cfg, 'edges', 'copy')
    out_dtype = _setting(get('out_dtype'), cfg, 'out_dtype', 'auto')
    amp = bool(get('amp', False))
    offset = cfg.dataset.get('offset', 0) or 0
    rescale = cfg.train.get('loss', 'mse') in ('poisson_nll', 'weighted_poisson_nll')
    policy = resolve_policy(get('domain_policy'), cfg)

    model, checkpoint = load_model_from_checkpoint(cli_args.checkpoint, cfg)
    ckpt_digest = file_digest(cli_args.checkpoint)
    jobs = _jobs(cfg, get('source', '0'))
    out_arg = Path(cli_args.output)
    if len(jobs) > 1 or out_arg.suffix == "":
        out_arg.mkdir(parents=True, exist_ok=True)
        outputs = [out_arg / f"{_safe_name(j['label'])}.h5" for j in jobs]
    else:
        outputs = [out_arg]

    summaries = []
    for job, output in zip(jobs, outputs):
        with job["open"]() as src:
            if "info" in job:
                info = job["info"]
            else:
                plan = plan_binned(src, **{k: v for k, v in job["read"].items()
                                           if k != "correct_fn"})
                info = finish_info(dataset_info(src, plan, job["read"]["q_crop"], np.float32,
                                                corrected=job["read"]["correct_fn"] is not None),
                                   job["spec"], job["mask_prov"])
            domain_shift = check_compatible(checkpoint.get('provenance'), [info],
                                            offset=offset, policy=policy)
            settings = {"neighbor_mode": neighbor_mode, "edges": edges, "upsample": upsample,
                        "offset": offset, "rescale": rescale, "out_dtype": str(out_dtype),
                        "amp": amp}
            run_key = hashlib.sha256(json.dumps(
                [ckpt_digest, info, settings], sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            record = {**runtime_info(), "output": str(output),
                      "checkpoint": str(cli_args.checkpoint), "checkpoint_digest": ckpt_digest,
                      "checkpoint_epoch": checkpoint.get('epoch'),
                      "weights_provenance": checkpoint.get('provenance'),
                      "data": info, "domain_policy": policy, "domain_shift": domain_shift,
                      "config": str(config_path), "run_key": run_key,
                      "inference": dict(settings)}
            print(f"\n{job['label']} -> {output}  ({info.get('out_shape')}, "
                  f"mode={neighbor_mode}, edges={edges}, batch={batch_size}, "
                  f"block_rows={block_rows})")
            summary = denoise_source(
                model, src, **{k: v for k, v in job["read"].items()}, offset=offset,
                neighbor_mode=neighbor_mode, edges=edges, upsample=upsample, rescale=rescale,
                batch_size=batch_size, block_rows=block_rows, writer=_writer(output, out_dtype),
                run_key=run_key, device=device, amp=amp, provenance=record)
        rate = summary["positions_per_second"]
        print(f"  done: {summary['positions_denoised']} positions in {summary['seconds']} s "
              f"({rate} pos/s), rescale x{summary['rescale_factor']:.4f}")
        if domain_shift:
            print(f"  NOTE: {len(domain_shift)} difference(s) from the training data "
                  f"recorded in the output's provenance")
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/denoised.h5",
                        help=".h5 / .npy file, or a directory (one .h5 per dataset)")
    parser.add_argument("--source", default="0",
                        help="dataset.sources entry to denoise: index, label, or 'all'")
    parser.add_argument("--neighbor-mode", default=None, choices=["spatial", "temporal"])
    parser.add_argument("--upsample", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="positions per forward pass (default 16; halved on GPU OOM)")
    parser.add_argument("--block-rows", type=int, default=None,
                        help="scan rows read and denoised per block (default 8)")
    parser.add_argument("--edges", choices=["copy", "denoise"], default=None,
                        help="outer scan rows/columns: copied from neighbours (default, as "
                             "before) or denoised with centre-frame substitution")
    parser.add_argument("--out-dtype", choices=["auto", "float16", "float32"], default=None,
                        help="auto = float16 for .h5, float32 for .npy")
    parser.add_argument("--amp", action="store_true",
                        help="mixed-precision inference (faster, not bit-identical)")
    parser.add_argument("--domain-policy", choices=["strict", "warn", "off"], default=None,
                        help="weights trained with other binning/offset: strict = stop "
                             "(default), warn/off = run and record the differences")
    cli_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    config_path = Path(cli_args.config).resolve()
    cfg = get_configuration(config_path)
    cli_args.config = str(config_path)
    main(cli_args, cfg, config_path)
