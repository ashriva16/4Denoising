"""
Frame sources: one small interface in front of every file format.

A *frame source* only has to say how many diffraction patterns it holds, their
shape and dtype, and return patterns ``[a, b)`` in scan (raster) order. Everything
else -- artifact correction, cropping, binning -- lives in ``core.stream`` and is
the same for every format.

Built-in readers
    arina    DECTRIS ARINA ``*_master.h5`` (streamed; needs ``hdf5plugin``)
    h5       a 4D (or 5D, with ``index``) dataset in any HDF5 file
    npy      a ``.npy`` file, memory-mapped
    rsciio   any format RosettaSciIO reads (``fmt="rsciio:<plugin>"``, e.g.
             ``rsciio:quantumdetector`` for Merlin .mib). Lazy where the plugin
             supports it; otherwise the whole file is read into RAM, with a warning.
    array    an array already in memory (numpy, dask, memmap, h5py dataset)

Adding a format
    @register_reader("myformat", extensions=(".xyz",))
    def open_myformat(path, **options) -> FrameSource: ...
    or, if the data is already an array-like of shape (Ny, Nx, Qy, Qx):
        return ArraySource(array_like, name="myformat", path=path)

``open_source(path, fmt="auto")`` picks a reader from the file's extension and
layout: ARINA master, then a known HDF5 dataset, then .npy, then RosettaSciIO by
extension (if installed).
"""

import importlib
import warnings
from pathlib import Path

import numpy as np

__all__ = ["FrameSource", "ArraySource", "register_reader", "open_source",
           "available_readers", "READERS"]


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class FrameSource:
    """Base class. Subclasses set the attributes below and implement ``read``."""

    name = "base"
    n_frames: int
    frame_shape: tuple            # (Qy, Qx)
    dtype: np.dtype
    scan_shape = None             # (Ny, Nx) if the file stores it, else None
    path = None

    def read(self, a, b):
        """Frames ``[a, b)`` as a numpy array of shape ``(b - a, Qy, Qx)``."""
        raise NotImplementedError

    def describe(self):
        """Provenance: what was read, from where, with which reader."""
        return {"reader": self.name, "path": None if self.path is None else str(self.path),
                "n_frames": int(self.n_frames), "frame_shape": list(self.frame_shape),
                "dtype": str(np.dtype(self.dtype)),
                "scan_shape": None if self.scan_shape is None else list(self.scan_shape)}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class ArraySource(FrameSource):
    """Any array-like that supports basic slicing: numpy, np.memmap, dask, h5py.

    Accepts ``(Ny, Nx, Qy, Qx)`` (scan shape known) or ``(N, Qy, Qx)``.
    Only the requested frames are materialised (``.compute()`` for dask).
    """

    def __init__(self, array, name="array", path=None, closer=None, extra_info=None):
        if array.ndim not in (3, 4):
            raise ValueError(f"expected a 3D or 4D array, got shape {array.shape}")
        self._a = array
        self.name, self.path, self._closer = name, path, closer
        self._extra = dict(extra_info or {})
        self.frame_shape = tuple(array.shape[-2:])
        self.dtype = np.dtype(array.dtype)
        if array.ndim == 4:
            self.scan_shape = tuple(array.shape[:2])
            self.n_frames = self.scan_shape[0] * self.scan_shape[1]
        else:
            self.n_frames = array.shape[0]

    def describe(self):
        return {**super().describe(), **self._extra}

    @staticmethod
    def _materialise(x):
        x = x.compute() if hasattr(x, "compute") else x
        return np.asarray(x)

    def read(self, a, b):
        if not (0 <= a <= b <= self.n_frames):
            raise IndexError(f"frames [{a}, {b}) outside [0, {self.n_frames})")
        if self._a.ndim == 3:
            return self._materialise(self._a[a:b])
        nx = self.scan_shape[1]
        parts, i = [], a
        while i < b:                                   # one slice per scan row spanned
            y, x = divmod(i, nx)
            stop = min(b - i, nx - x)
            parts.append(self._materialise(self._a[y, x:x + stop]))
            i += stop
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def close(self):
        if self._closer is not None:
            self._closer()
            self._closer = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
READERS = {}


def register_reader(name, extensions=(), sniff=None):
    """Register ``fn(path, **options) -> FrameSource`` under ``name``.

    extensions : file suffixes this reader claims for ``fmt="auto"``
    sniff      : optional ``fn(path) -> bool`` that inspects the file; when given,
                 the reader is only chosen automatically if it returns True
    """
    def deco(fn):
        READERS[name] = {"open": fn, "extensions": tuple(e.lower() for e in extensions),
                         "sniff": sniff}
        return fn
    return deco


def available_readers():
    return sorted(READERS) + ["rsciio:<plugin>"]


def open_source(path_or_array, fmt="auto", **options):
    """Open a frame source. ``fmt``: 'auto', a registered name, or 'rsciio:<plugin>'."""
    if not isinstance(path_or_array, (str, Path)):
        return ArraySource(path_or_array)
    path = Path(path_or_array)
    if not path.exists():
        raise FileNotFoundError(path)

    if fmt.startswith("rsciio:"):
        return open_rsciio(path, plugin=fmt.split(":", 1)[1], **options)
    if fmt != "auto":
        if fmt not in READERS:
            raise KeyError(f"unknown format '{fmt}'. Available: {available_readers()}")
        return READERS[fmt]["open"](path, **options)

    suffix = path.suffix.lower()
    for name, r in READERS.items():                  # registration order = priority
        if suffix in r["extensions"] and (r["sniff"] is None or r["sniff"](path)):
            return r["open"](path, **options)
    plugin = _rsciio_plugin_for(path)
    if plugin is not None:
        return open_rsciio(path, plugin=plugin, **options)
    raise ValueError(f"could not pick a reader for {path.name}; pass fmt= one of "
                     f"{available_readers()}")


# ---------------------------------------------------------------------------
# DECTRIS ARINA
# ---------------------------------------------------------------------------
_WARNED_NO_PLUGIN = False


def _h5():
    global _WARNED_NO_PLUGIN
    import h5py
    try:
        import hdf5plugin  # noqa: F401  (bitshuffle/LZ4 filter used by ARINA)
    except ImportError:
        if not _WARNED_NO_PLUGIN:
            warnings.warn("hdf5plugin not installed: compressed ARINA/HDF5 data will not open")
            _WARNED_NO_PLUGIN = True
    return h5py


def _is_arina(path):
    h5py = _h5()
    try:
        with h5py.File(str(path), "r") as f:
            return "entry/data" in f and len(f["entry/data"]) > 0
    except OSError:
        return False


class ArinaSource(FrameSource):
    """``*_master.h5`` whose ``entry/data`` links to ``data_00000N`` frame stacks.

    The scan shape is not stored in the file; pass it to the pipeline.
    """

    name = "arina"

    def __init__(self, path):
        h5py = _h5()
        self.path = Path(path)
        self._f = h5py.File(str(self.path), "r")
        grp = self._f["entry/data"]
        self._index, offset = [], 0
        for key in sorted(grp.keys()):                # data_000001, ... (zero-padded)
            d = grp[key]
            self._index.append((d, offset, d.shape[0]))
            offset += d.shape[0]
        d0 = self._index[0][0]
        self.n_frames, self.frame_shape, self.dtype = offset, tuple(d0.shape[1:]), d0.dtype

    def read(self, a, b):
        parts = []
        for d, start, n in self._index:
            end = start + n
            if end <= a or start >= b:
                continue
            parts.append(d[max(a, start) - start: min(b, end) - start])
            if end >= b:
                break
        if not parts or a < 0 or b > self.n_frames:
            raise IndexError(f"frames [{a}, {b}) outside [0, {self.n_frames})")
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def describe(self):
        return {**super().describe(), "n_files": len(self._index)}

    def close(self):
        if getattr(self, "_f", None) is not None:
            self._f.close()
            self._f = None


@register_reader("arina", extensions=(".h5", ".hdf5"), sniff=_is_arina)
def open_arina(path, **_):
    return ArinaSource(path)


# ---------------------------------------------------------------------------
# Generic HDF5 dataset
# ---------------------------------------------------------------------------
# keys tried in order; the first three are the formats core.io already reads
H5_KNOWN_KEYS = ("fpd_expt/fpd_data/data", "Experiments/__unnamed__/data",
                 "datacube", "data")


def _find_h5_key(f):
    for k in H5_KNOWN_KEYS:
        if k in f and hasattr(f[k], "shape") and f[k].ndim in (3, 4):
            return k
    found = []
    f.visititems(lambda n, o: found.append(n) if hasattr(o, "ndim") and o.ndim == 4 else None)
    if len(found) == 1:
        return found[0]
    raise KeyError("no unique 4D dataset; pass key=. 4D datasets: "
                   f"{found or 'none'}")


def _is_generic_h5(path):
    h5py = _h5()
    try:
        with h5py.File(str(path), "r") as f:
            _find_h5_key(f)
            return True
    except (OSError, KeyError):
        return False


@register_reader("h5", extensions=(".h5", ".hdf5", ".emd"), sniff=_is_generic_h5)
def open_h5(path, key=None, index=None, **_):
    """``key``: dataset path (auto-detected if None). ``index``: position along the
    first axis of a 5D stack ``(P, Ny, Nx, Qy, Qx)``, e.g. one entry of a series."""
    h5py = _h5()
    f = h5py.File(str(path), "r")
    try:
        key = key or _find_h5_key(f)
        d = f[key]
        if d.ndim == 5:
            if index is None:
                raise ValueError(f"'{key}' is 5D {d.shape}; pass index= to pick one entry")
            arr = _Indexed(d, index)
        else:
            arr = d
        return ArraySource(arr, name="h5", path=path, closer=f.close,
                           extra_info={"key": key, "index": index})
    except Exception:
        f.close()
        raise


class _Indexed:
    """View of one entry of a 5D dataset without reading the others."""

    def __init__(self, d, i):
        self._d, self._i = d, int(i)
        self.shape, self.dtype, self.ndim = d.shape[1:], d.dtype, d.ndim - 1

    def __getitem__(self, k):
        k = k if isinstance(k, tuple) else (k,)
        return self._d[(self._i, *k)]


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------
@register_reader("npy", extensions=(".npy",))
def open_npy(path, **_):
    arr = np.load(str(path), mmap_mode="r")
    mm = getattr(arr, "_mmap", None)               # release the file handle on close
    return ArraySource(arr, name="npy", path=path,
                       closer=(mm.close if mm is not None else None))


# ---------------------------------------------------------------------------
# RosettaSciIO (optional dependency: pip install ".[formats]")
# ---------------------------------------------------------------------------
def _rsciio_plugin_for(path):
    """Plugin name for an extension, if RosettaSciIO is installed and unambiguous."""
    try:
        from rsciio import IO_PLUGINS
    except ImportError:
        return None
    ext = path.suffix.lower().lstrip(".")
    hits = [p for p in IO_PLUGINS
            if ext in [e.lower() for e in p.get("file_extensions", [])]]
    if len(hits) == 1:
        return hits[0]["api"].split(".")[-1]
    if len(hits) > 1:
        names = [h["api"].split(".")[-1] for h in hits]
        raise ValueError(f"'.{ext}' is claimed by several RosettaSciIO plugins {names}; "
                         f"pass fmt='rsciio:<plugin>'")
    return None


def open_rsciio(path, plugin, lazy=True, signal_index=None, **reader_kwargs):
    """Read with ``rsciio.<plugin>.file_reader``. Extra kwargs go to that reader
    (e.g. ``navigation_shape`` for formats that do not store the scan shape)."""
    try:
        mod = importlib.import_module(f"rsciio.{plugin}")
    except ImportError as e:
        raise ImportError(f"RosettaSciIO plugin '{plugin}' not available "
                          f"(pip install rosettasciio): {e}") from e
    try:
        signals = mod.file_reader(str(path), lazy=lazy, **reader_kwargs)
    except TypeError:                                  # plugin without a lazy option
        signals = mod.file_reader(str(path), **reader_kwargs)
    candidates = [i for i, s in enumerate(signals) if np.ndim(s["data"]) in (3, 4)]
    if signal_index is None:
        if not candidates:
            raise ValueError(f"no 3D/4D dataset in {path.name} "
                             f"(shapes: {[np.shape(s['data']) for s in signals]})")
        signal_index = candidates[0]
    sig = signals[signal_index]
    data = sig["data"]
    if lazy and not hasattr(data, "compute"):
        warnings.warn(f"rsciio.{plugin} read {path.name} fully into RAM "
                      f"({np.asarray(data).nbytes / 1e9:.1f} GB); this plugin is not lazy")
    return ArraySource(data, name=f"rsciio:{plugin}", path=path,
                       extra_info={"signal_index": signal_index,
                                   "rsciio_metadata": sig.get("metadata", {})})
