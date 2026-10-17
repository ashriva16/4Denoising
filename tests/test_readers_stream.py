"""Tests for core.readers and core.stream on synthetic data (no microscopy data, no GPU).

Groups
  numpy only   pipeline maths, ArraySource, registry           (always run)
  h5py         ARINA layout, generic HDF5, .npy round trips    (skip without h5py)
  rsciio       adapter around a RosettaSciIO-style plugin      (fake plugin, always run)
  hyperspy     real RosettaSciIO round trip via a .hspy file   (skip without hyperspy)
"""

import sys
import types

import numpy as np
import pytest

from core.readers import ArraySource, READERS, open_source, register_reader
from core.stream import load_dataset, stream_binned

NY, NX, Q = 16, 24, 12


def _cube(seed=0, lam=4.0, dtype=np.uint16):
    rng = np.random.default_rng(seed)
    return rng.poisson(lam, size=(NY, NX, Q, Q)).astype(dtype)


def _reference(cube, crop_y, crop_x, q_bin, r_bin):
    c = cube[slice(*crop_y), slice(*crop_x)].astype(np.float64)
    ny, nx = c.shape[0] // r_bin, c.shape[1] // r_bin
    qq = Q // q_bin
    c = c[:ny * r_bin, :nx * r_bin, :qq * q_bin, :qq * q_bin]
    return c.reshape(ny, r_bin, nx, r_bin, qq, q_bin, qq, q_bin).sum(axis=(1, 3, 5, 7))


# --------------------------------------------------------------------------
# numpy only
# --------------------------------------------------------------------------
@pytest.mark.parametrize("q_bin,r_bin,crop_y,crop_x", [
    (1, 1, (0, NY), (0, NX)),
    (2, 1, (3, 13), (5, 21)),
    (2, 8, (0, NY), (0, NX)),
    (3, 3, (1, 15), (2, 23)),    # crop not divisible by r_bin -> trimmed
])
def test_binning_matches_reference(q_bin, r_bin, crop_y, crop_x):
    cube = _cube()
    got = stream_binned(ArraySource(cube), crop_y=crop_y, crop_x=crop_x,
                        q_bin=q_bin, r_bin=r_bin)
    ref = _reference(cube, crop_y, crop_x, q_bin, r_bin)
    assert got.shape == ref.shape and got.dtype == np.float32
    np.testing.assert_array_equal(got, ref)


def test_3d_stack_needs_scan_shape_and_matches_4d():
    cube = _cube()
    stack = cube.reshape(NY * NX, Q, Q)
    with pytest.raises(ValueError):
        stream_binned(ArraySource(stack))                      # shape unknown
    with pytest.raises(ValueError):
        stream_binned(ArraySource(stack), scan_shape=(NY, NX + 1))
    a = stream_binned(ArraySource(stack), scan_shape=(NY, NX), q_bin=2, r_bin=4)
    b = stream_binned(ArraySource(cube), q_bin=2, r_bin=4)
    np.testing.assert_array_equal(a, b)


def test_array_source_reads_across_rows():
    cube = _cube()
    src = ArraySource(cube)
    flat = cube.reshape(-1, Q, Q)
    np.testing.assert_array_equal(src.read(20, 75), flat[20:75])
    with pytest.raises(IndexError):
        src.read(0, NY * NX + 1)


def test_counts_conserved_integer_output():
    cube = _cube()
    got = stream_binned(ArraySource(cube), q_bin=2, r_bin=4, out_dtype=np.uint32)
    assert int(got.sum()) == int(cube.astype(np.int64).sum())


def test_float_input_keeps_fractions():
    cube = _cube().astype(np.float32) * 1.37 + 0.25            # e.g. gain-corrected counts
    got = stream_binned(ArraySource(cube), q_bin=2, r_bin=2)
    np.testing.assert_allclose(got, _reference(cube, (0, NY), (0, NX), 2, 2), rtol=1e-6)
    with pytest.raises(ValueError):
        stream_binned(ArraySource(cube), out_dtype=np.uint16)   # would truncate


def test_correct_fn_runs_at_full_size_before_binning():
    cube = _cube()
    cube[:, :, 5, 7] = 60000
    seen = []

    def fix(run):
        seen.append(run.shape[1:])
        run = run.copy()
        run[:, 5, 7] = 0
        return run

    got = stream_binned(ArraySource(cube), q_bin=2, correct_fn=fix)
    assert set(seen) == {(Q, Q)}
    cube[:, :, 5, 7] = 0
    np.testing.assert_array_equal(got, _reference(cube, (0, NY), (0, NX), 2, 1))


def test_overflow_raises_instead_of_wrapping():
    cube = np.full((NY, NX, Q, Q), 60000, np.uint16)
    with pytest.raises(OverflowError):
        stream_binned(ArraySource(cube), q_bin=2, r_bin=2, out_dtype=np.uint16)


def test_crop_smaller_than_r_bin_raises():
    with pytest.raises(ValueError):
        stream_binned(ArraySource(_cube()), crop_y=(0, 4), crop_x=(0, 4), r_bin=8)


def test_load_dataset_info_records_settings():
    data, info = load_dataset(ArraySource(_cube()), q_bin=2, r_bin=4, crop_x=[0, 20])
    assert data.shape == (4, 5, 6, 6)
    for k, v in {"reader": "array", "q_bin": 2, "r_bin": 4, "bin_mode": "sum",
                 "scan_shape": [NY, NX], "crop_x": [0, 20], "out_shape": [4, 5, 6, 6]}.items():
        assert info[k] == v, k


def test_custom_reader_registration(tmp_path):
    """A new format is one function; the pipeline does not change."""
    cube = _cube()
    p = tmp_path / "scan.fake"
    p.write_bytes(b"x")

    @register_reader("fakefmt", extensions=(".fake",))
    def _open(path, **_):
        return ArraySource(cube, name="fakefmt", path=path)

    try:
        data, info = load_dataset(p, q_bin=2)                   # picked by extension
        assert info["reader"] == "fakefmt"
        np.testing.assert_array_equal(data, _reference(cube, (0, NY), (0, NX), 2, 1))
        with pytest.raises(KeyError):
            open_source(p, fmt="no-such-format")
    finally:
        READERS.pop("fakefmt", None)


# --------------------------------------------------------------------------
# RosettaSciIO adapter, with a fake plugin module (no install needed)
# --------------------------------------------------------------------------
def test_rsciio_adapter_with_fake_plugin(tmp_path, monkeypatch):
    cube = _cube()
    calls = {}

    def file_reader(filename, lazy=False, **kw):
        calls.update(lazy=lazy, **kw)
        return [{"data": np.zeros(5)},                          # e.g. a 1D signal first
                {"data": cube, "axes": [], "metadata": {"General": {"title": "t"}}}]

    pkg = types.ModuleType("rsciio")
    pkg.__path__ = []
    mod = types.ModuleType("rsciio.fakeplugin")
    mod.file_reader = file_reader
    monkeypatch.setitem(sys.modules, "rsciio", pkg)
    monkeypatch.setitem(sys.modules, "rsciio.fakeplugin", mod)

    p = tmp_path / "scan.whatever"
    p.write_bytes(b"x")
    with pytest.warns(UserWarning):                             # numpy, not lazy -> RAM warning
        data, info = load_dataset(p, fmt="rsciio:fakeplugin", q_bin=2,
                                  reader_options={"navigation_shape": (NX, NY)})
    assert calls == {"lazy": True, "navigation_shape": (NX, NY)}
    assert info["reader"] == "rsciio:fakeplugin" and info["signal_index"] == 1
    np.testing.assert_array_equal(data, _reference(cube, (0, NY), (0, NX), 2, 1))


# --------------------------------------------------------------------------
# h5py: real files
# --------------------------------------------------------------------------
def _write_arina(h5py, hdf5plugin, tmp_path, frames, per_file=150, compress=True):
    """Master file whose entry/data links to separate data files, like ARINA."""
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as m:
        grp = m.create_group("entry/data")
        for k, a in enumerate(range(0, len(frames), per_file), start=1):
            name = f"scan_data_{k:06d}.h5"
            with h5py.File(tmp_path / name, "w") as d:
                kw = hdf5plugin.Bitshuffle(cname="lz4") if compress else {}
                d.create_dataset("entry/data/data", data=frames[a:a + per_file],
                                 chunks=(1, Q, Q), **kw)
            grp[f"data_{k:06d}"] = h5py.ExternalLink(name, "/entry/data/data")
    return master


@pytest.mark.parametrize("compress", [False, True])
def test_arina_files(tmp_path, compress):
    h5py = pytest.importorskip("h5py")
    hdf5plugin = pytest.importorskip("hdf5plugin")
    cube = _cube(seed=1)
    master = _write_arina(h5py, hdf5plugin, tmp_path, cube.reshape(-1, Q, Q), compress=compress)

    with open_source(master) as src:                            # auto-detected
        assert src.name == "arina" and src.n_frames == NY * NX
        assert src.describe()["n_files"] == 3
    data, info = load_dataset(master, scan_shape=(NY, NX), crop_y=[2, 14], q_bin=2, r_bin=4)
    np.testing.assert_array_equal(data, _reference(cube, (2, 14), (0, NX), 2, 4))
    with pytest.raises(ValueError):
        load_dataset(master)                                    # ARINA has no scan shape
    with pytest.raises(ValueError):
        load_dataset(master, scan_shape=(NY, NX + 1))


def test_generic_h5_4d_and_5d(tmp_path):
    h5py = pytest.importorskip("h5py")
    cube = _cube(seed=2)
    p4, p5 = tmp_path / "c4.h5", tmp_path / "c5.h5"
    with h5py.File(p4, "w") as f:
        f.create_dataset("some/group/cube", data=cube)        # unknown key, unique 4D
    with h5py.File(p5, "w") as f:
        f.create_dataset("series/data", data=np.stack([cube, cube + 1, cube + 2]))

    data, info = load_dataset(p4, q_bin=2)                      # scan shape from the file
    assert info["reader"] == "h5" and info["key"] == "some/group/cube"
    np.testing.assert_array_equal(data, _reference(cube, (0, NY), (0, NX), 2, 1))

    data, info = load_dataset(p5, fmt="h5", reader_options={"key": "series/data", "index": 2})
    np.testing.assert_array_equal(data, cube + 2)
    assert info["index"] == 2
    with pytest.raises(ValueError):
        load_dataset(p5, fmt="h5", reader_options={"key": "series/data"})   # 5D needs index


def test_npy_memmap(tmp_path):
    cube = _cube(seed=3)
    p = tmp_path / "c.npy"
    np.save(p, cube)
    data, info = load_dataset(p, r_bin=2)
    assert info["reader"] == "npy"
    np.testing.assert_array_equal(data, _reference(cube, (0, NY), (0, NX), 1, 2))


# --------------------------------------------------------------------------
# hyperspy: real RosettaSciIO round trip (lazy .hspy read)
# --------------------------------------------------------------------------
def test_rsciio_hspy_roundtrip(tmp_path):
    hs = pytest.importorskip("hyperspy.api")
    pytest.importorskip("rsciio.hspy")
    cube = _cube(seed=4)
    p = tmp_path / "c.hspy"
    hs.signals.Signal2D(cube).save(p)
    data, info = load_dataset(p, fmt="rsciio:hspy", q_bin=2, r_bin=2)
    assert info["reader"] == "rsciio:hspy"
    np.testing.assert_array_equal(data, _reference(cube, (0, NY), (0, NX), 2, 2))
