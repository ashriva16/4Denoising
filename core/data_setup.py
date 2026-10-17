"""
One place that turns a config into the arrays train.py and infer.py work on,
so training and inference can never preprocess differently.

Two config styles are accepted.

Legacy (unchanged behaviour; this is what the published configs use)::

    dataset:
      data_dir: data/TiO2
      file: TiO2_002.hdf5
      bin_factor: 2
      defect_mask: true
      ...

Sources (new): any number of datasets, read through core.readers / core.stream::

    dataset:
      sources:
        - path: data/scan_A_master.h5     # relative to the working directory
          scan_shape: [500, 1000]         # needed when the file does not store it
          crop_y: [0, 200]                # optional, per source
          crop_x: [0, 200]
          label: A                        # optional, used in logs and provenance
        - path: data/scan_B.npy
      fmt: auto                           # default reader for all sources
      q_bin: 2                            # detector binning (sum)
      r_bin: 1                            # scan binning (sum)
      defect_mask: true                   # same mask for every source (see below)
      ...
      scan_mask: {mode: all}              # which positions are training targets

Every source must give the same detector shape after binning; scan shapes may
differ. Per-source keys (``fmt``, ``reader_options``, ``q_crop``) override the
shared ones.

Defect handling in the sources style detects ONE mask (on the first source) and
applies it to every source, so all datasets are corrected identically.
"""

from pathlib import Path

import numpy as np

from core.preprocessing import (bin_datacube, correct_dead_pixels, detect_dead_pixels,
                                offset_datacube)

__all__ = ["load_training_data", "uses_sources"]


def _get(d, key, default=None):
    """dict/AttrDict-safe lookup that also tolerates a missing section."""
    if d is None:
        return default
    try:
        return d.get(key, default)
    except AttributeError:
        return getattr(d, key, default)


def uses_sources(cfg):
    return _get(cfg.dataset, "sources") is not None


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


def _defects(cubes, ds, verbose):
    """Detect one mask on the first cube, correct every cube with it."""
    if not _get(ds, "defect_mask", False):
        return cubes, None
    mask, _ = detect_dead_pixels(
        cubes[0], method=_get(ds, "defect_method", "combined"),
        threshold_factor=_get(ds, "defect_sigma", 5),
        min_dead_fraction=_get(ds, "defect_dead_fraction", 0.8), visualize=False)
    method = _get(ds, "defect_correction", "median_local")
    cubes = [correct_dead_pixels(c, mask, method=method, visualize_sample=False) for c in cubes]
    if verbose:
        print(f"defect mask: {int(mask.sum())} pixels, applied to {len(cubes)} source(s)")
    return cubes, mask


def _load_legacy(cfg, verbose):
    """The original single-file path, kept byte-for-byte equivalent."""
    from core.io import load_4dstem

    ds = cfg.dataset
    rel = Path(_get(ds, "data_dir")) / _get(ds, "file")
    data, _ = load_4dstem(_resolve(rel), crop_N=_get(ds, "crop_N"))
    if _get(ds, "bin_factor", 1) > 1:
        data = bin_datacube(data, _get(ds, "bin_factor"))
    if _get(ds, "defect_mask", False):
        mask, _ = detect_dead_pixels(
            data, method=_get(ds, "defect_method", "combined"),
            threshold_factor=_get(ds, "defect_sigma", 5),
            min_dead_fraction=_get(ds, "defect_dead_fraction", 0.8), visualize=False)
        data = correct_dead_pixels(data, mask, method=_get(ds, "defect_correction", "median_local"),
                                   visualize_sample=False)
    info = {"reader": "legacy core.io.load_4dstem", "path": str(rel),
            "q_bin": int(_get(ds, "bin_factor", 1)), "r_bin": 1}
    return [data], [rel.stem], [info]


def _load_sources(cfg, verbose):
    from core.stream import load_dataset

    ds = cfg.dataset
    cubes, labels, infos = [], [], []
    for i, src in enumerate(_get(ds, "sources")):
        path = _resolve(_get(src, "path"))
        data, info = load_dataset(
            path, fmt=_get(src, "fmt", _get(ds, "fmt", "auto")),
            scan_shape=_get(src, "scan_shape", _get(ds, "scan_shape")),
            crop_y=_get(src, "crop_y"), crop_x=_get(src, "crop_x"),
            q_bin=int(_get(ds, "q_bin", 1)), r_bin=int(_get(ds, "r_bin", 1)),
            q_crop=_get(src, "q_crop", _get(ds, "q_crop")),
            out_dtype=np.float32,
            reader_options=dict(_get(src, "reader_options", _get(ds, "reader_options", {})) or {}))
        label = str(_get(src, "label", f"{i}:{path.stem}"))
        info["label"] = label
        cubes.append(data)
        labels.append(label)
        infos.append(info)
        if verbose:
            print(f"source {label}: {info['reader']} {path.name} -> {data.shape}")

    q_shapes = {c.shape[2:] for c in cubes}
    if len(q_shapes) != 1:
        raise ValueError(f"all sources must have the same detector shape after binning, "
                         f"got {sorted(q_shapes)}")
    return cubes, labels, infos


def load_training_data(cfg, verbose=True):
    """Returns ``(cubes, labels, infos)``: lists of float32 ``(Rx, Ry, Qx, Qy)`` arrays,
    their names, and one provenance dict per source."""
    ds = cfg.dataset
    if uses_sources(cfg):
        cubes, labels, infos = _load_sources(cfg, verbose)
        cubes, mask = _defects(cubes, ds, verbose)
        for info in infos:
            info["defect_mask_px"] = None if mask is None else int(mask.sum())
    else:
        cubes, labels, infos = _load_legacy(cfg, verbose)

    offset = _get(ds, "offset", 0)
    if offset and offset > 0:
        cubes = [offset_datacube(c, offset) for c in cubes]
    cubes = [np.ascontiguousarray(c, dtype=np.float32) for c in cubes]
    if verbose:
        total = sum(c.nbytes for c in cubes) / 1e9
        print(f"data ready: {len(cubes)} source(s), {total:.2f} GB float32")
    return cubes, labels, infos
