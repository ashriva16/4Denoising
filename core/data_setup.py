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
      defect_mask_file: masks/det.h5      # preferred: scripts/make_defect_mask.py
      defect_mask: true                   # fallback: detect after binning (see below)
      ...
      scan_mask: {mode: all}              # which positions are training targets

A source that was already binned before it was written (e.g. a cache) declares
it with ``prebinned: {q_bin: 2, r_bin: 8}``; provenance and the weight checks then
use the total binning relative to the raw data.

Every source must give the same detector shape after binning; scan shapes may
differ. Per-source keys (``fmt``, ``reader_options``, ``q_crop``) override the
shared ones.

Defect handling in the sources style, in order of preference:
  ``defect_mask_file``  a mask built once at full detector size
                        (scripts/make_defect_mask.py); every frame is inpainted
                        while it is read, before any binning.
  ``defect_mask: true`` ONE mask detected on the first source after binning and
                        applied to every source (the original method).
"""

from pathlib import Path

import numpy as np

from core.preprocessing import (bin_datacube, correct_dead_pixels, detect_dead_pixels,
                                offset_datacube)

__all__ = ["load_training_data", "uses_sources", "source_specs", "finish_info",
           "mask_correction"]


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
    correct_fn, mask_prov = _mask_correction(ds, verbose)
    if correct_fn is not None:                    # full-size inpainting, before binning
        Rx, Ry = data.shape[:2]
        data = correct_fn(data.reshape(Rx * Ry, *data.shape[2:])).reshape(data.shape)
    if _get(ds, "bin_factor", 1) > 1:
        data = bin_datacube(data, _get(ds, "bin_factor"))
    if correct_fn is None and _get(ds, "defect_mask", False):
        mask, _ = detect_dead_pixels(
            data, method=_get(ds, "defect_method", "combined"),
            threshold_factor=_get(ds, "defect_sigma", 5),
            min_dead_fraction=_get(ds, "defect_dead_fraction", 0.8), visualize=False)
        data = correct_dead_pixels(data, mask, method=_get(ds, "defect_correction", "median_local"),
                                   visualize_sample=False)
    info = {"reader": "legacy core.io.load_4dstem", "path": str(rel), "label": rel.stem,
            "q_bin": int(_get(ds, "bin_factor", 1)), "r_bin": 1, **(mask_prov or {})}
    return [data], [rel.stem], [info]


def _mask_correction(ds, verbose):
    """correct_fn and provenance from dataset.defect_mask_file, or (None, None)."""
    path = _get(ds, "defect_mask_file")
    if not path:
        return None, None
    from core.defects import load_mask, make_inpainter

    mask, info = load_mask(_resolve(path))
    fn = make_inpainter(mask, **dict(info.get("inpaint", {}) or {}))
    prov = {"defect_mask_file": str(path), "defect_mask_digest": info.get("digest"),
            "defect_mask_px": int(mask.sum()), "defect_mask_shape": list(mask.shape)}
    if verbose:
        print(f"defect mask file {Path(path).name}: {int(mask.sum())} px, "
              f"digest {info.get('digest')}, applied at full detector size")
    if _get(ds, "defect_mask", False):
        print("note: defect_mask_file is set, so defect_mask (post-binning detection) is ignored")
    return fn, prov


def source_specs(cfg):
    """One dict per dataset.sources entry with everything needed to read it:
    path, fmt, scan_shape, crops, q_crop, reader_options, prebinned, label,
    plus the shared q_bin / r_bin. Nothing is opened."""
    ds = cfg.dataset
    specs = []
    for i, src in enumerate(_get(ds, "sources") or []):
        path = _resolve(_get(src, "path"))
        pre = _get(src, "prebinned", _get(ds, "prebinned"))
        specs.append({
            "label": str(_get(src, "label", f"{i}:{path.stem}")),
            "path": path,
            "fmt": _get(src, "fmt", _get(ds, "fmt", "auto")),
            "scan_shape": _get(src, "scan_shape", _get(ds, "scan_shape")),
            "crop_y": _get(src, "crop_y"), "crop_x": _get(src, "crop_x"),
            "q_crop": _get(src, "q_crop", _get(ds, "q_crop")),
            "q_bin": int(_get(ds, "q_bin", 1)), "r_bin": int(_get(ds, "r_bin", 1)),
            "reader_options": dict(_get(src, "reader_options",
                                        _get(ds, "reader_options", {})) or {}),
            "prebinned": None if not pre else {"q_bin": int(_get(pre, "q_bin", 1)),
                                               "r_bin": int(_get(pre, "r_bin", 1))},
        })
    return specs


def finish_info(info, spec, mask_prov=None):
    """Label, prebinned totals and defect-mask provenance on a dataset_info dict."""
    info["label"] = spec["label"]
    if spec.get("prebinned"):                  # binning applied before the file was written
        info["prebinned"] = spec["prebinned"]
        info["q_bin"] = info["q_bin"] * spec["prebinned"]["q_bin"]   # totals relative to
        info["r_bin"] = info["r_bin"] * spec["prebinned"]["r_bin"]   # the raw data
    if mask_prov:
        info.update(mask_prov)
    return info


def mask_correction(cfg_or_ds, verbose=True):
    """Public form of the defect-mask lookup: (correct_fn, provenance) or (None, None)."""
    ds = getattr(cfg_or_ds, "dataset", cfg_or_ds)
    return _mask_correction(ds, verbose)


def _load_sources(cfg, verbose):
    from core.stream import load_dataset

    ds = cfg.dataset
    correct_fn, mask_prov = _mask_correction(ds, verbose)
    cubes, labels, infos = [], [], []
    for spec in source_specs(cfg):
        data, info = load_dataset(
            spec["path"], fmt=spec["fmt"], scan_shape=spec["scan_shape"],
            crop_y=spec["crop_y"], crop_x=spec["crop_x"],
            q_bin=spec["q_bin"], r_bin=spec["r_bin"], q_crop=spec["q_crop"],
            out_dtype=np.float32, correct_fn=correct_fn,
            reader_options=spec["reader_options"])
        finish_info(info, spec, mask_prov)
        cubes.append(data)
        labels.append(spec["label"])
        infos.append(info)
        if verbose:
            print(f"source {spec['label']}: {info['reader']} {spec['path'].name} -> {data.shape}")

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
        if not _get(ds, "defect_mask_file"):
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
