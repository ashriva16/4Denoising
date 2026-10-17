"""Tests for core.inference (block-streamed, batched inference) and scripts.infer.

The reference is the original position-by-position loop over DataSetFromArray;
the engine must reproduce it for every block size, batch size and neighbour mode."""

import argparse
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from core.dataloader import DataSetFromArray  # noqa: E402
from core.inference import ArrayWriter, H5Writer, denoise_source  # noqa: E402
from core.readers import ArraySource  # noqa: E402

Q = 16


def _model(seed=0):
    import core.models as models
    from core.models import get_unet
    torch.manual_seed(seed)
    args = argparse.Namespace(model="blind-video-net-5d-cross", channels=1, out_channels=1,
                              bias=False, normal=False, blind_noise=False,
                              unet_cls=get_unet("original"))
    return models.build_model(args).eval()


def _cube(R=8, C=10, seed=1):
    return np.random.default_rng(seed).poisson(4.0, (R, C, Q, Q)).astype(np.float32)


def _reference(model, data, mode="spatial", upsample=1, offset=0.0, rescale=False):
    """The original infer.py, position by position."""
    ds = DataSetFromArray(data + offset, neighbor_mode=mode)
    R, C = data.shape[:2]
    out = np.zeros((R, C, Q * upsample, Q * upsample), np.float32)
    with torch.no_grad():
        for rx in range(1, R - 1):
            for ry in range(1, C - 1):
                x, _ = ds.getitem([rx, ry])
                y = model(x.unsqueeze(0).float())
                if upsample > 1:
                    y = F.interpolate(y, scale_factor=upsample, mode="bicubic",
                                      align_corners=False)
                out[rx, ry] = y.squeeze().numpy()
    out[0] = out[1]
    out[-1] = out[-2]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    out = np.clip(out - offset, 0, None)
    if rescale:
        out = out * (data.sum(dtype=np.float64) / out.sum(dtype=np.float64))
    return out


def _run(model, data, **kw):
    w = ArrayWriter()
    denoise_source(model, ArraySource(data), writer=w, progress=False, **kw)
    return w.data


# --------------------------------------------------------------------------
# equivalence with the original loop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["spatial", "temporal"])
@pytest.mark.parametrize("block_rows,batch_size", [(2, 1), (3, 7), (8, 64), (5, 16)])
def test_matches_position_by_position(mode, block_rows, batch_size):
    model, data = _model(), _cube()
    ref = _reference(model, data, mode)
    got = _run(model, data, neighbor_mode=mode, block_rows=block_rows,
               batch_size=batch_size, rescale=False)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_offset_and_upsample():
    model, data = _model(), _cube()
    ref = _reference(model, data, offset=1.0, upsample=2)
    got = _run(model, data, offset=1.0, rescale=False, upsample=2, block_rows=3)
    assert got.shape == (8, 10, 2 * Q, 2 * Q) and np.isfinite(got).all()
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_rescale_conserves_counts():
    model, data = _model(), _cube()
    ref = _reference(model, data, rescale=True)
    assert ref.sum() > 0                                     # non-degenerate output
    got = _run(model, data, rescale=True, block_rows=3)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got.sum(dtype=np.float64), data.sum(dtype=np.float64), rtol=1e-6)


def test_all_zero_output_never_gives_nan():
    """An output that is zero everywhere (e.g. an offset larger than the model's
    output) must not turn the count rescaling into 0 * inf = NaN."""
    model, data = _model(), _cube()
    with pytest.warns(UserWarning):
        got = _run(model, data, offset=1e6, rescale=True)
    assert np.isfinite(got).all() and (got == 0).all()


def test_edges_denoise_keeps_interior():
    model, data = _model(), _cube()
    a = _run(model, data, edges="copy", rescale=False)
    b = _run(model, data, edges="denoise", rescale=False)
    np.testing.assert_allclose(a[1:-1, 1:-1], b[1:-1, 1:-1], rtol=1e-5, atol=1e-6)
    assert np.isfinite(b).all() and not np.allclose(a[0], b[0])


def test_streaming_from_file_equals_binned_array(tmp_path):
    """Binning while reading, in blocks, gives the same result as binning first."""
    model = _model()
    raw = np.random.default_rng(2).poisson(1.0, (16, 20, 2 * Q, 2 * Q)).astype(np.uint16)
    np.save(tmp_path / "raw.npy", raw)
    from core.readers import open_source
    binned = raw.astype(np.float32).reshape(8, 2, 10, 2, Q, 2, Q, 2).sum(axis=(1, 3, 5, 7))
    ref = _run(model, binned, rescale=True)
    w = ArrayWriter()
    with open_source(tmp_path / "raw.npy") as src:
        denoise_source(model, src, q_bin=2, r_bin=2, writer=w, block_rows=3, progress=False)
    np.testing.assert_allclose(w.data, ref, rtol=1e-5, atol=1e-6)


def test_invalid_settings_are_refused():
    model, data = _model(), _cube()
    for bad in ({"amp": "tf32"}, {"edges": "crop"}, {"neighbor_mode": "diagonal"}):
        with pytest.raises(ValueError):
            denoise_source(model, ArraySource(data), progress=False, **bad)


def test_float16_overflow_is_refused():
    model, data = _model(), _cube() * 1e5        # counts conserved -> values ~ 4e5
    w = ArrayWriter(dtype=np.float16)
    with pytest.raises(OverflowError):
        denoise_source(model, ArraySource(data), writer=w, rescale=True, progress=False)


# --------------------------------------------------------------------------
# HDF5 output: provenance, resume after a crash, no double rescaling
# --------------------------------------------------------------------------
class _Crash(Exception):
    pass


def test_h5_output_and_resume(tmp_path):
    h5py = pytest.importorskip("h5py")
    model, data = _model(), _cube(R=11, C=9)
    full = _run(model, data, rescale=True, block_rows=3)

    class CrashingWriter(H5Writer):
        calls = 0

        def write(self, *a, **k):
            CrashingWriter.calls += 1
            if CrashingWriter.calls == 3:
                raise _Crash
            super().write(*a, **k)

    out = tmp_path / "den.h5"
    with pytest.raises(_Crash):
        denoise_source(model, ArraySource(data), writer=CrashingWriter(out), run_key="k1",
                       block_rows=3, progress=False, provenance={"note": "test"})
    assert (tmp_path / "den.h5.partial").exists() and not out.exists()

    summary = denoise_source(model, ArraySource(data), writer=H5Writer(out), run_key="k1",
                             block_rows=3, progress=False, provenance={"note": "test"})
    assert out.exists() and not (tmp_path / "den.h5.partial").exists()
    assert summary["positions_denoised"] < (11 - 2) * (9 - 2)        # resumed, not redone
    with h5py.File(out, "r") as f:
        # float16 storage, rounded before and after the count rescaling: ~2 float16 steps
        np.testing.assert_allclose(f["data"][...].astype(np.float32), full, rtol=4e-3, atol=4e-3)
        assert set(f) == {"data", "intensity_image", "mean_pattern"}
        prov = json.loads(f.attrs["provenance"])
    assert prov["note"] == "test" and prov["inference"]["block_rows"] == 3


def test_interrupted_rescale_never_scales_twice(tmp_path):
    h5py = pytest.importorskip("h5py")
    model, data = _model(), _cube(R=9, C=9)
    full = _run(model, data, rescale=True, block_rows=2)

    class CrashInRescale(H5Writer):
        n = 0

        def rewrite(self, r0, block):
            CrashInRescale.n += 1
            if CrashInRescale.n == 2:
                raise _Crash
            super().rewrite(r0, block)

    out = tmp_path / "den.h5"
    w = CrashInRescale(out, dtype=np.float32)
    with pytest.raises(_Crash):
        denoise_source(model, ArraySource(data), writer=w, run_key="k", block_rows=2,
                       progress=False)
    denoise_source(model, ArraySource(data), writer=H5Writer(out, dtype=np.float32),
                   run_key="k", block_rows=2, progress=False)
    with h5py.File(out, "r") as f:
        np.testing.assert_allclose(f["data"][...], full, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# scripts.infer: all sources into a directory of .h5 files
# --------------------------------------------------------------------------
CONFIG = """
dataset:
  sources:
    - path: a.npy
      label: A
    - path: b.npy
      label: B
  q_bin: 1
  r_bin: 1
  offset: 0
model:
  name: blind-video-net-5d-cross
  unet: original
  channels: 1
  out_channels: 1
  bias: false
  normal: false
  blind_noise: false
train:
  loss: poisson_nll
"""


def test_infer_all_sources_to_h5(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("ruamel.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["test"])
    np.save(tmp_path / "a.npy", _cube(R=6, C=7, seed=3))
    np.save(tmp_path / "b.npy", _cube(R=5, C=9, seed=4))
    model = _model()
    torch.save({"model": model.state_dict(), "epoch": 1,
                "provenance": {"preprocessing": {"q_bin": 1, "r_bin": 1, "offset": 0}}},
               tmp_path / "w.pth")
    (tmp_path / "cfg.yml").write_text(CONFIG, encoding="utf-8")

    from utils.opts import get_configuration
    import scripts.infer as infer
    cfg = get_configuration(tmp_path / "cfg.yml")
    cli = argparse.Namespace(checkpoint=str(tmp_path / "w.pth"), output=str(tmp_path / "out"),
                             source="all", batch_size=4, block_rows=2)
    infer.main(cli, cfg, tmp_path / "cfg.yml")
    for label, shape in (("A", (6, 7)), ("B", (5, 9))):
        with h5py.File(tmp_path / "out" / f"{label}.h5", "r") as f:
            assert f["data"].shape == (*shape, Q, Q) and f["data"].dtype == np.float16
            prov = json.loads(f.attrs["provenance"])
        assert prov["data"]["label"] == label and prov["domain_shift"] == []
        assert prov["inference"]["rescale_factor"] > 0 and "code" in prov
