"""
Detector defect masks: built once, stored as a file, applied while reading.

Defects belong to the detector, not to a dataset, so the mask is computed once
at FULL detector size from a reference region and every dataset -- training and
inference -- is corrected with the same file, before any binning:

    stats = reference_stats(source, scan_shape, crop_y, crop_x)   # streamed
    mask  = build_mask(stats, [{"type": "hot", "factor": 3.0}, ...])
    save_mask("detector_mask.h5", mask, info)
    ...
    mask, info = load_mask("detector_mask.h5")
    correct_fn = make_inpainter(mask, **info["inpaint"])          # for core.stream

Detector types (each returns a boolean mask; the final mask is their union)
    hot          pixel > factor x median of its 8 neighbours in the mean pattern
                 (and above min_mean)
    py4dstem_hot py4DSTEM's DataCube.filter_hot_pixels on the mean pattern
                 (thresh, ind_compare); needs py4DSTEM
    dead         mean below min_frac x median of the non-zero mean pattern
    low_mean     mean below global mean - sigma x std (core.preprocessing 'statistical')
    zero_count   at or below zero_level in more than min_dead_fraction of the frames
    lines        whole rows / columns (chip seams, central cross)
    pixels       explicit (qy, qx) coordinates
New types: add a function to DETECTORS.

The inpainter replaces each flagged pixel by the mean of the nearest good pixels
(window grown until ``min_donors`` are found, up to ``max_radius``), so clusters
of bad pixels never donate to each other.
"""

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

__all__ = ["reference_stats", "build_mask", "make_inpainter", "save_mask", "load_mask",
           "mask_digest", "DETECTORS"]


# ---------------------------------------------------------------------------
# Reference statistics, streamed (never more than one scan row in memory)
# ---------------------------------------------------------------------------
def reference_stats(source, scan_shape=None, crop_y=None, crop_x=None, zero_level=0):
    """Per-pixel mean, std, max and zero fraction over a scan region, at full size."""
    from core.stream import _bounds, _resolve_scan_shape

    Ny, Nx = _resolve_scan_shape(source, scan_shape)
    y0, y1 = _bounds(crop_y, Ny)
    x0, x1 = _bounds(crop_x, Nx)
    shape = tuple(source.frame_shape)
    s = np.zeros(shape, np.float64)
    s2 = np.zeros(shape, np.float64)
    mx = np.full(shape, -np.inf)
    zeros = np.zeros(shape, np.int64)
    n = 0
    for y in range(y0, y1):
        run = source.read(y * Nx + x0, y * Nx + x1).astype(np.float64)
        s += run.sum(axis=0)
        s2 += (run ** 2).sum(axis=0)
        mx = np.maximum(mx, run.max(axis=0))
        zeros += (run <= zero_level).sum(axis=0)
        n += run.shape[0]
    mean = s / n
    return {"mean": mean, "std": np.sqrt(np.maximum(s2 / n - mean ** 2, 0)), "max": mx,
            "zero_fraction": zeros / n, "n_frames": n,
            "region": {"scan_shape": [Ny, Nx], "crop_y": [y0, y1], "crop_x": [x0, x1]}}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def _hot(stats, factor=3.0, min_mean=0.0):
    m = stats["mean"]
    fp = np.ones((3, 3), bool)
    fp[1, 1] = False
    neigh = ndimage.median_filter(m, footprint=fp, mode="nearest")
    return (m > factor * neigh) & (m > min_mean)


def _py4dstem_hot(stats, thresh=0.3, ind_compare=5):
    import py4DSTEM
    dp = stats["mean"].astype(np.float32)
    dc = py4DSTEM.DataCube(dp[None, None])
    _, m = dc.filter_hot_pixels(thresh=thresh, ind_compare=ind_compare, return_mask=True)
    return np.asarray(m).reshape(dp.shape).astype(bool)


def _dead(stats, min_frac=0.02):
    m = stats["mean"]
    ref = np.median(m[m > 0]) if (m > 0).any() else 0.0
    return m < min_frac * ref


def _low_mean(stats, sigma=3.0):
    m = stats["mean"]
    return m < (m.mean() - sigma * m.std())


def _zero_count(stats, min_dead_fraction=0.8):
    return stats["zero_fraction"] > min_dead_fraction


def _lines(stats, rows=(), cols=()):
    m = np.zeros(stats["mean"].shape, bool)
    for r in rows:
        m[int(r), :] = True
    for c in cols:
        m[:, int(c)] = True
    return m


def _pixels(stats, coords=()):
    m = np.zeros(stats["mean"].shape, bool)
    for qy, qx in coords:
        m[int(qy), int(qx)] = True
    return m


DETECTORS = {"hot": _hot, "py4dstem_hot": _py4dstem_hot, "dead": _dead,
             "low_mean": _low_mean, "zero_count": _zero_count,
             "lines": _lines, "pixels": _pixels}


def build_mask(stats, detectors):
    """Union of the listed detectors. ``detectors``: list of dicts with a 'type'.
    Returns ``(mask, counts)`` where counts gives the pixels each detector flagged."""
    mask = np.zeros(stats["mean"].shape, bool)
    counts = []
    for spec in detectors:
        spec = dict(spec)
        kind = spec.pop("type")
        if kind not in DETECTORS:
            raise KeyError(f"unknown detector '{kind}'. Available: {sorted(DETECTORS)}")
        m = DETECTORS[kind](stats, **spec)
        counts.append({"type": kind, **spec, "flagged": int(m.sum())})
        mask |= m
    return mask, counts


# ---------------------------------------------------------------------------
# Inpainting (the correct_fn handed to core.stream)
# ---------------------------------------------------------------------------
def make_inpainter(mask, max_radius=4, min_donors=3):
    """``correct_fn(run) -> run`` for ``(n, Qy, Qx)`` frames at full detector size.

    The frame dtype is kept: integer frames get the donor mean rounded to the
    nearest count, float frames the exact mean."""
    mask = np.asarray(mask, bool)
    Qy, Qx = mask.shape
    good = ~mask
    bad = np.argwhere(mask)
    bad_flat, donors = [], []
    for qy, qx in bad:
        found, r = np.empty((0, 2), int), 1
        while len(found) < min_donors and r <= max_radius:
            y0, y1 = max(0, qy - r), min(Qy, qy + r + 1)
            x0, x1 = max(0, qx - r), min(Qx, qx + r + 1)
            ys, xs = np.nonzero(good[y0:y1, x0:x1])
            found = np.stack([ys + y0, xs + x0], axis=1)
            r += 1
        bad_flat.append(qy * Qx + qx)
        donors.append(found[:, 0] * Qx + found[:, 1] if len(found) else np.array([], int))
    n_orphans = sum(1 for d in donors if d.size == 0)

    def correct(run):
        if run.shape[1:] != (Qy, Qx):
            raise ValueError(f"defect mask is {(Qy, Qx)}, frames are {run.shape[1:]} "
                             f"(the mask must be built at full detector size)")
        out = np.array(run, copy=True)                 # sources may be read-only
        flat = out.reshape(out.shape[0], Qy * Qx)
        is_int = out.dtype.kind in "ui"
        for bp, dn in zip(bad_flat, donors):
            if dn.size:
                v = flat[:, dn].mean(axis=1)
                flat[:, bp] = np.rint(v) if is_int else v
        return out

    correct.n_masked = int(mask.sum())
    correct.n_orphans = n_orphans                      # bad pixels with no good donor
    return correct


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def mask_digest(mask):
    """Short content hash, recorded in provenance so a run names its exact mask."""
    m = np.ascontiguousarray(np.asarray(mask, bool))
    return hashlib.sha256(m.tobytes() + str(m.shape).encode()).hexdigest()[:16]


def save_mask(path, mask, info=None, mean_pattern=None):
    """HDF5 with the mask, optional reference mean pattern and a JSON 'info' attribute."""
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = dict(info or {})
    info.setdefault("created_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
    info.update(shape=list(mask.shape), n_masked=int(np.sum(mask)), digest=mask_digest(mask))
    with h5py.File(path, "w") as f:
        f.create_dataset("mask", data=np.asarray(mask, bool))
        if mean_pattern is not None:
            f.create_dataset("reference_mean", data=np.asarray(mean_pattern, np.float32))
        f.attrs["info"] = json.dumps(info, default=str)
    return info


def load_mask(path):
    """``(mask, info)`` from an .h5 written by save_mask, or a boolean .npy."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mask = np.load(path).astype(bool)
        return mask, {"shape": list(mask.shape), "n_masked": int(mask.sum()),
                      "digest": mask_digest(mask), "inpaint": {}}
    import h5py
    with h5py.File(path, "r") as f:
        mask = f["mask"][...].astype(bool)
        info = json.loads(f.attrs["info"]) if "info" in f.attrs else {}
    if info.get("digest") not in (None, mask_digest(mask)):
        raise ValueError(f"{path.name}: mask content does not match its recorded digest")
    info.setdefault("inpaint", {})
    return mask, info
