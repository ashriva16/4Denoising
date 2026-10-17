"""
Format-independent, memory-bounded loading of 4D-STEM data.

``stream_binned`` reads one scan row at a time from any ``core.readers`` frame
source and never holds more than one row of full-size frames:

    full-size frames -> correct_fn (artifact inpainting, at full detector size)
                     -> detector crop -> detector binning (q_bin)
                     -> scan binning (r_bin, accumulated over r_bin rows)

Binning SUMS, which keeps the Poisson statistics the denoiser's loss assumes.
Integer data are summed as integers (int64/uint64) and checked against the
output dtype before the cast, so an overflow raises instead of wrapping; float
data (e.g. detector-corrected exports) are summed in float64.

    data, info = load_dataset("scan_master.h5", scan_shape=(500, 1000), q_bin=2, r_bin=8)

``info`` records the reader, the file and every binning/crop setting, for the
provenance of whatever is computed from ``data``.
"""

import numpy as np

from core.readers import FrameSource, open_source

__all__ = ["plan_binned", "stream_binned", "dataset_info", "load_dataset"]


def _bounds(crop, n):
    """None | slice | (start, stop) -> (start, stop) within [0, n]. YAML gives lists."""
    if crop is None:
        return 0, n
    if isinstance(crop, slice):
        start, stop, step = crop.indices(n)
        if step != 1:
            raise ValueError("crops must be contiguous (step 1)")
        return start, stop
    start, stop = crop
    return slice(start, stop).indices(n)[:2]


def _resolve_scan_shape(source, scan_shape):
    if scan_shape is None:
        if source.scan_shape is None:
            raise ValueError(f"{source.name}: the file does not store the scan shape; "
                             f"pass scan_shape=(Ny, Nx)")
        return tuple(source.scan_shape)
    scan_shape = tuple(int(v) for v in scan_shape)
    if scan_shape[0] * scan_shape[1] != source.n_frames:
        raise ValueError(f"scan_shape {scan_shape} needs {scan_shape[0] * scan_shape[1]} "
                         f"frames, {source.name} has {source.n_frames} "
                         f"(dropped or flyback frames?)")
    if source.scan_shape is not None and tuple(source.scan_shape) != scan_shape:
        raise ValueError(f"scan_shape {scan_shape} contradicts the file's {source.scan_shape}")
    return scan_shape


def plan_binned(source, scan_shape=None, crop_y=None, crop_x=None, q_bin=1, r_bin=1,
                q_crop=None):
    """Geometry of a binned read, without reading anything.

    Returns a dict with the raw scan shape, the raw scan and detector bounds actually
    used (trimmed to multiples of the binning) and the output shape
    ``(ny, nx, Qy, Qx)``."""
    Ny, Nx = _resolve_scan_shape(source, scan_shape)
    y0, y1 = _bounds(crop_y, Ny)
    x0, x1 = _bounds(crop_x, Nx)
    ny_out, nx_out = (y1 - y0) // r_bin, (x1 - x0) // r_bin
    if ny_out == 0 or nx_out == 0:
        raise ValueError(f"scan crop {(y1 - y0, x1 - x0)} is smaller than r_bin={r_bin}")
    Qy0, Qx0 = source.frame_shape
    qc = (None, None) if q_crop is None else q_crop
    qy0, qy1 = _bounds(qc[0], Qy0)
    qx0, qx1 = _bounds(qc[1], Qx0)
    qy_len, qx_len = (qy1 - qy0) // q_bin, (qx1 - qx0) // q_bin
    return {"scan_shape": (Ny, Nx), "y0": y0, "x0": x0,
            "y1": y0 + ny_out * r_bin, "x1": x0 + nx_out * r_bin,
            "q": (qy0, qy0 + qy_len * q_bin, qx0, qx0 + qx_len * q_bin),
            "q_bin": int(q_bin), "r_bin": int(r_bin),
            "out_shape": (ny_out, nx_out, qy_len, qx_len)}


def stream_binned(source, scan_shape=None, crop_y=None, crop_x=None, q_bin=1, r_bin=1,
                  q_crop=None, out_dtype=np.float32, correct_fn=None, rows=None):
    """Read a frame source into ``(ny, nx, Qy, Qx)``, binned while reading.

    Parameters
    ----------
    source : FrameSource
    scan_shape : (Ny, Nx) or None
        Required when the file does not store it (ARINA, MIB, plain 3D stacks).
    crop_y, crop_x : None, slice or (start, stop)
        Scan region, trimmed at the far edge to a multiple of ``r_bin``.
    q_bin, r_bin : int
        Detector and scan binning factors (counts are summed).
    q_crop : None or ((qy0, qy1), (qx0, qx1))
        Detector region, applied after ``correct_fn`` and before ``q_bin``.
    out_dtype : numpy dtype
        float32 (default) holds integer counts exactly up to 2**24.
    correct_fn : callable or None
        ``(n, Qy, Qx) -> (n, Qy, Qx)`` at full detector size, e.g. dead-pixel inpainting.
    rows : None or (i0, i1)
        Only output rows ``[i0, i1)`` (in binned units), for block-wise processing.
    """
    plan = plan_binned(source, scan_shape, crop_y, crop_x, q_bin, r_bin, q_crop)
    Ny, Nx = plan["scan_shape"]
    ny_out, nx_out, qy_len, qx_len = plan["out_shape"]
    x0, x1 = plan["x0"], plan["x1"]
    qy0, qy1, qx0, qx1 = plan["q"]
    nx_crop = x1 - x0
    i0, i1 = (0, ny_out) if rows is None else (max(0, rows[0]), min(ny_out, rows[1]))
    y0 = plan["y0"] + i0 * r_bin
    y1 = plan["y0"] + i1 * r_bin

    in_dtype, out_dtype = np.dtype(source.dtype), np.dtype(out_dtype)
    is_float = in_dtype.kind == "f"
    if is_float and out_dtype.kind in "ui":
        raise ValueError(f"{in_dtype} input would be truncated by out_dtype={out_dtype}")
    acc_dtype = np.float64 if is_float else (np.int64 if in_dtype.kind == "i" else np.uint64)
    if out_dtype.kind in "ui":
        lo, hi = np.iinfo(out_dtype).min, np.iinfo(out_dtype).max
    elif not is_float:
        lo, hi = -(2 ** 24), 2 ** 24           # integers stay exact in float32 up to 2**24
    else:
        lo, hi = -np.finfo(out_dtype).max, np.finfo(out_dtype).max

    out = np.empty((i1 - i0, nx_out, qy_len, qx_len), dtype=out_dtype)
    acc = np.zeros((nx_out, qy_len, qx_len), dtype=acc_dtype)
    for iy, y in enumerate(range(y0, y1)):
        run = source.read(y * Nx + x0, y * Nx + x1)           # (nx_crop, Qy0, Qx0)
        if correct_fn is not None:
            run = correct_fn(run)
        run = run[:, qy0:qy1, qx0:qx1].astype(acc_dtype)
        if q_bin > 1:
            run = run.reshape(nx_crop, qy_len, q_bin, qx_len, q_bin).sum(axis=(2, 4))
        if r_bin > 1:
            run = run.reshape(nx_out, r_bin, qy_len, qx_len).sum(axis=1)
        acc += run
        if (iy + 1) % r_bin == 0:
            amin, amax = acc.min(), acc.max()
            if amax > hi or amin < lo:
                raise OverflowError(f"binned values span [{amin}, {amax}], outside "
                                    f"[{lo}, {hi}] for {out_dtype.name}; use a wider out_dtype")
            out[iy // r_bin] = acc.astype(out_dtype)
            acc[:] = 0
    return out


def dataset_info(source, plan, q_crop=None, out_dtype=np.float32, corrected=False):
    """Provenance dict for a planned read (the same keys load_dataset returns)."""
    Ny, Nx = plan["scan_shape"]
    return {**source.describe(),
            "scan_shape": [Ny, Nx],
            "crop_y": [plan["y0"], plan["y1"]], "crop_x": [plan["x0"], plan["x1"]],
            "q_crop": None if q_crop is None else [list(c) if c else None for c in q_crop],
            "q_bin": plan["q_bin"], "r_bin": plan["r_bin"], "bin_mode": "sum",
            "out_dtype": str(np.dtype(out_dtype)), "out_shape": list(plan["out_shape"]),
            "corrected": bool(corrected)}


def load_dataset(path_or_source, fmt="auto", scan_shape=None, crop_y=None, crop_x=None,
                 q_bin=1, r_bin=1, q_crop=None, out_dtype=np.float32, correct_fn=None,
                 reader_options=None):
    """Open, stream and bin in one call. Returns ``(data, info)``.

    ``reader_options`` go to the reader (e.g. ``{"key": ..., "index": ...}`` for h5,
    ``{"navigation_shape": ...}`` for some RosettaSciIO plugins).
    """
    own = not isinstance(path_or_source, FrameSource)
    source = open_source(path_or_source, fmt=fmt, **(reader_options or {})) if own \
        else path_or_source
    try:
        data = stream_binned(source, scan_shape, crop_y, crop_x, q_bin, r_bin, q_crop,
                             out_dtype, correct_fn)
        plan = plan_binned(source, scan_shape, crop_y, crop_x, q_bin, r_bin, q_crop)
        info = dataset_info(source, plan, q_crop, out_dtype, corrected=correct_fn is not None)
        return data, info
    finally:
        if own:
            source.close()
