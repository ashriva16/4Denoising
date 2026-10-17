"""Tests for core.defects, the defect-mask file and its use while reading."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core.defects import (build_mask, load_mask, make_inpainter, mask_digest,
                          reference_stats, save_mask)
from core.readers import ArraySource
from core.stream import stream_binned

NY, NX, Q = 10, 12, 16


def _cube(seed=0, lam=20.0):
    rng = np.random.default_rng(seed)
    c = rng.poisson(lam, (NY, NX, Q, Q)).astype(np.uint16)
    c[:, :, 3, 4] = 2000            # hot pixel
    c[:, :, 10, 11] = 0             # dead pixel
    c[:, :, 7, :] = 0               # dead row (e.g. a chip seam)
    return c


# --------------------------------------------------------------------------
# statistics and detectors
# --------------------------------------------------------------------------
def test_reference_stats_match_numpy():
    cube = _cube()
    st = reference_stats(ArraySource(cube), crop_y=(2, 8), crop_x=(1, 11))
    sub = cube[2:8, 1:11].reshape(-1, Q, Q).astype(np.float64)
    np.testing.assert_allclose(st["mean"], sub.mean(0))
    np.testing.assert_allclose(st["std"], sub.std(0), atol=1e-9)
    np.testing.assert_allclose(st["zero_fraction"], (sub <= 0).mean(0))
    assert st["n_frames"] == 60 and st["region"]["crop_y"] == [2, 8]


def test_detectors_find_planted_defects():
    st = reference_stats(ArraySource(_cube()))
    hot, _ = build_mask(st, [{"type": "hot", "factor": 3.0}])
    assert hot[3, 4] and hot.sum() == 1
    dead, _ = build_mask(st, [{"type": "dead", "min_frac": 0.1}])
    assert dead[10, 11] and dead[7].all()
    zero, _ = build_mask(st, [{"type": "zero_count", "min_dead_fraction": 0.8}])
    np.testing.assert_array_equal(zero, dead)
    lines, _ = build_mask(st, [{"type": "lines", "rows": [7], "cols": [0]}])
    assert lines[7].all() and lines[:, 0].all() and lines.sum() == 2 * Q - 1
    px, _ = build_mask(st, [{"type": "pixels", "coords": [[1, 2], [5, 6]]}])
    assert px.sum() == 2 and px[1, 2] and px[5, 6]
    union, counts = build_mask(st, [{"type": "hot", "factor": 3.0},
                                    {"type": "dead", "min_frac": 0.1}])
    assert union.sum() == 1 + 1 + Q
    assert [c["flagged"] for c in counts] == [1, 1 + Q]
    with pytest.raises(KeyError):
        build_mask(st, [{"type": "nonsense"}])


def test_py4dstem_hot_if_available():
    pytest.importorskip("py4DSTEM")
    st = reference_stats(ArraySource(_cube()))
    m, _ = build_mask(st, [{"type": "py4dstem_hot", "thresh": 0.3, "ind_compare": 5}])
    assert m[3, 4]


# --------------------------------------------------------------------------
# inpainting
# --------------------------------------------------------------------------
def test_inpainter_uses_only_good_donors():
    mask = np.zeros((Q, Q), bool)
    mask[5:8, 5:8] = True                                    # a 3x3 cluster
    frames = np.random.default_rng(1).poisson(10, (4, Q, Q)).astype(np.float32)
    frames[:, mask] = 1e6                                    # bad values must not leak
    fn = make_inpainter(mask, max_radius=4, min_donors=3)
    out = fn(frames)
    assert out[:, mask].max() < 100
    np.testing.assert_array_equal(out[:, ~mask], frames[:, ~mask])   # good pixels untouched
    assert fn.n_masked == 9 and fn.n_orphans == 0


def test_inpainter_integer_readonly_and_shape_check():
    mask = np.zeros((Q, Q), bool)
    mask[0, 0] = True
    frames = np.full((3, Q, Q), 7, np.uint16)
    frames[:, 0, 0] = 60000
    frames.setflags(write=False)                             # e.g. a memory-mapped source
    out = make_inpainter(mask)(frames)
    assert out.dtype == np.uint16 and (out[:, 0, 0] == 7).all()
    with pytest.raises(ValueError):
        make_inpainter(mask)(np.zeros((2, Q // 2, Q // 2), np.uint16))   # binned frames
    all_bad = make_inpainter(np.ones((4, 4), bool))
    assert all_bad.n_orphans == 16


def test_correction_happens_before_binning():
    cube = _cube()
    st = reference_stats(ArraySource(cube))
    mask, _ = build_mask(st, [{"type": "hot", "factor": 3.0}, {"type": "dead", "min_frac": 0.1}])
    fn = make_inpainter(mask)
    got = stream_binned(ArraySource(cube), q_bin=2, r_bin=2, correct_fn=fn)
    fixed = fn(cube.reshape(-1, Q, Q)).reshape(cube.shape).astype(np.float64)
    ref = fixed.reshape(NY // 2, 2, NX // 2, 2, Q // 2, 2, Q // 2, 2).sum(axis=(1, 3, 5, 7))
    np.testing.assert_array_equal(got, ref)


# --------------------------------------------------------------------------
# file round trip
# --------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    pytest.importorskip("h5py")
    mask = np.zeros((Q, Q), bool)
    mask[2, 3] = mask[9, 9] = True
    info = save_mask(tmp_path / "m.h5", mask, {"inpaint": {"max_radius": 3, "min_donors": 2}},
                     mean_pattern=np.ones((Q, Q)))
    m2, info2 = load_mask(tmp_path / "m.h5")
    np.testing.assert_array_equal(m2, mask)
    assert info2["digest"] == info["digest"] == mask_digest(mask)
    assert info2["inpaint"] == {"max_radius": 3, "min_donors": 2} and info2["n_masked"] == 2

    np.save(tmp_path / "m.npy", mask)                        # plain .npy masks also work
    m3, info3 = load_mask(tmp_path / "m.npy")
    np.testing.assert_array_equal(m3, mask)
    assert info3["inpaint"] == {}


def test_tampered_mask_is_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    mask = np.zeros((Q, Q), bool)
    mask[1, 1] = True
    save_mask(tmp_path / "m.h5", mask)
    with h5py.File(tmp_path / "m.h5", "a") as f:
        f["mask"][2, 2] = True
    with pytest.raises(ValueError):
        load_mask(tmp_path / "m.h5")


# --------------------------------------------------------------------------
# end to end: make_defect_mask script, then data_setup applies the file
# --------------------------------------------------------------------------
CONFIG = """
dataset:
  sources:
    - path: scan.npy
      label: A
  q_bin: 2
  r_bin: 1
  defect_mask_file: masks/det.h5
defects:
  output: masks/det.h5
  reference:
    source: A
    crop_y: [0, 6]
  detectors:
    - {type: hot, factor: 3.0}
    - {type: dead, min_frac: 0.1}
  inpaint: {max_radius: 4, min_donors: 3}
"""


def test_make_mask_script_and_data_setup(tmp_path, monkeypatch):
    for mod in ("h5py", "ruamel.yaml", "torch", "matplotlib"):
        pytest.importorskip(mod)
    monkeypatch.chdir(tmp_path)
    cube = _cube(seed=5)
    np.save(tmp_path / "scan.npy", cube)
    (tmp_path / "cfg.yml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["test"])

    from utils.opts import get_configuration
    import scripts.make_defect_mask as mdm
    from core.data_setup import load_training_data

    cfg = get_configuration(tmp_path / "cfg.yml")
    mask, info = mdm.main(cfg, plot=str(tmp_path / "mask.png"))
    assert (tmp_path / "masks" / "det.h5").exists() and (tmp_path / "mask.png").exists()
    assert mask[3, 4] and mask[10, 11] and mask[7].all()
    assert info["reference"]["crop_y"] == [0, 6]

    (data,), _, infos = load_training_data(cfg, verbose=False)
    fn = make_inpainter(mask, max_radius=4, min_donors=3)
    fixed = fn(cube.reshape(-1, Q, Q)).reshape(cube.shape).astype(np.float64)
    ref = fixed.reshape(NY, NX, Q // 2, 2, Q // 2, 2).sum(axis=(3, 5))
    np.testing.assert_array_equal(data, ref)
    assert infos[0]["defect_mask_digest"] == info["digest"]

    legacy = SimpleNamespace(dataset={"data_dir": ".", "file": "scan.npy", "bin_factor": 2,
                                      "defect_mask_file": "masks/det.h5"})
    (data_legacy,), _, _ = load_training_data(legacy, verbose=False)
    # the legacy loader works in float32, so inpainted values are exact donor means;
    # integer sources get rounded means (they stay integer counts)
    fixed_f = fn(cube.reshape(-1, Q, Q).astype(np.float32)).reshape(cube.shape).astype(np.float64)
    ref_f = fixed_f.reshape(NY, NX, Q // 2, 2, Q // 2, 2).sum(axis=(3, 5))
    np.testing.assert_allclose(data_legacy, ref_f, rtol=1e-6)
    assert np.abs(data_legacy - data).max() <= 2          # rounding, at most 0.5 per pixel
