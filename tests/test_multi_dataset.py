"""Tests for several-dataset training: scan masks, MultiCubeDataset, data_setup,
and a one-epoch train + infer smoke run on tiny synthetic data (CPU is fine)."""

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.dataloader import (MultiCubeDataset, build_scan_mask,  # noqa: E402
                             build_training_dataset)
from core.data_setup import load_training_data  # noqa: E402

Q = 16


def _cube(rx, ry, value=None, seed=0):
    if value is not None:
        return np.full((rx, ry, Q, Q), float(value), np.float32)
    return np.random.default_rng(seed).poisson(3.0, (rx, ry, Q, Q)).astype(np.float32)


# --------------------------------------------------------------------------
# scan masks
# --------------------------------------------------------------------------
def test_scan_masks_partition_and_shapes(tmp_path):
    for mode in ("diagonal", "checkerboard"):
        a = build_scan_mask(12, 20, mode=mode, half="A")
        b = build_scan_mask(12, 20, mode=mode, half="B")
        assert a.shape == (12, 20) and not (a & b).any() and (a | b).all()
    assert build_scan_mask(12, 20).all()
    assert build_scan_mask(12, 20, mode="corner", corner=4).sum() == 16
    p = tmp_path / "m.npy"
    np.save(p, np.eye(12, 20, dtype=bool))
    assert build_scan_mask(12, 20, mode="file", path=str(p)).sum() == 12
    with pytest.raises(ValueError):
        build_scan_mask(10, 20, mode="file", path=str(p))


# --------------------------------------------------------------------------
# MultiCubeDataset
# --------------------------------------------------------------------------
def test_neighbours_never_cross_datasets():
    cubes = [_cube(8, 10, value=1), _cube(6, 12, value=2)]   # different scan shapes
    ds = MultiCubeDataset(cubes, neighbor_mode="alternating_spatial")
    for epoch in (0, 1):                                     # spatial, then temporal
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            frames, target, ci = ds[i]
            assert frames.shape == (9, Q, Q) and target.shape == (1, Q, Q)
            assert torch.all(frames == ci + 1) and torch.all(target == ci + 1)


def test_positions_follow_masks_and_margin():
    cubes = [_cube(8, 10), _cube(6, 12)]
    masks = [build_scan_mask(8, 10, mode="checkerboard", block=2), None]
    ds = MultiCubeDataset(cubes, masks=masks, neighbor_mode="alternating_spatial")
    m0 = masks[0][2:-2, 2:-2].sum()                          # margin 2
    assert len(ds) == m0 + (6 - 4) * (12 - 4)
    for ci, rx, ry in ds.index:
        if ci == 0:
            assert masks[0][rx, ry]
        assert 2 <= rx < cubes[ci].shape[0] - 2 and 2 <= ry < cubes[ci].shape[1] - 2


def test_single_dataset_matches_original_dataset():
    """One dataset, all positions: same samples as build_training_dataset."""
    cube = _cube(8, 10, seed=1)
    old = build_training_dataset(cube, neighbor_mode="alternating_spatial", image_size=None)
    new = MultiCubeDataset([cube], neighbor_mode="alternating_spatial")
    assert len(old) == len(new)
    for epoch in (0, 1):
        old.set_epoch(epoch)
        new.set_epoch(epoch)
        for i in (0, 7, len(new) - 1):
            fo, to = old[i]
            fn, tn, _ = new[i]
            assert torch.equal(fo, fn) and torch.equal(to, tn)


def test_random_mode_is_reproducible():
    ds = MultiCubeDataset([_cube(8, 10)], neighbor_mode="random")
    ds.set_epoch(3)
    a = ds.sample_modes.copy()
    ds.set_epoch(4)
    ds.set_epoch(3)
    assert (a == ds.sample_modes).all()


# --------------------------------------------------------------------------
# data_setup: legacy and sources configs give the same data for the same file
# --------------------------------------------------------------------------
def test_legacy_and_sources_agree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = np.random.default_rng(2).poisson(4.0, (8, 10, 32, 32)).astype(np.uint16)
    np.save(tmp_path / "scan.npy", raw)

    legacy = SimpleNamespace(dataset={"data_dir": ".", "file": "scan.npy", "bin_factor": 2})
    sources = SimpleNamespace(dataset={"sources": [{"path": "scan.npy", "label": "s"}],
                                       "q_bin": 2})
    (a,), _, _ = load_training_data(legacy, verbose=False)
    (b,), labels, infos = load_training_data(sources, verbose=False)
    np.testing.assert_array_equal(a, b)
    assert labels == ["s"] and infos[0]["reader"] == "npy" and infos[0]["q_bin"] == 2


def test_sources_must_share_detector_shape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.save(tmp_path / "a.npy", np.zeros((6, 6, 32, 32), np.uint16))
    np.save(tmp_path / "b.npy", np.zeros((6, 6, 16, 16), np.uint16))
    cfg = SimpleNamespace(dataset={"sources": [{"path": "a.npy"}, {"path": "b.npy"}]})
    with pytest.raises(ValueError):
        load_training_data(cfg, verbose=False)


# --------------------------------------------------------------------------
# smoke: one epoch of scripts.train on two sources, then scripts.infer
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
  scan_mask:
    mode: checkerboard
    block: 2
  neighbor_mode: alternating_spatial
  offset: 0
  defect_mask: false
  image_size: null
model:
  name: blind-video-net-5d-cross
  unet: original
  channels: 1
  out_channels: 1
  bias: false
  normal: false
  blind_noise: false
train:
  lr: 1.0e-4
  loss: poisson_nll
  scheduler: multistep
  num_epochs: 1
  batch_size: 8
  val_split: 0.2
  seed: 314
  milestones: [1]
  gamma: 0.5
  checkpoint: 1
  num_workers: 0
output:
  save_dir: ckpt
"""


def test_train_and_infer_smoke(tmp_path, monkeypatch):
    for mod in ("ruamel.yaml", "torchinfo", "tensorboard"):
        pytest.importorskip(mod)
    monkeypatch.chdir(tmp_path)
    np.save(tmp_path / "a.npy", np.random.default_rng(3).poisson(3, (10, 10, Q, Q)).astype(np.uint16))
    np.save(tmp_path / "b.npy", np.random.default_rng(4).poisson(3, (8, 12, Q, Q)).astype(np.uint16))
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["test"])              # get_configuration reads argv

    from utils.opts import get_configuration
    import scripts.train as train
    import scripts.infer as infer

    train.main(get_configuration(cfg_path), cfg_path, argparse.Namespace(resume=None))
    ckpts = sorted((tmp_path / "ckpt").rglob("checkpoint_1.pth"))
    assert ckpts, "no checkpoint written"

    out = tmp_path / "den_B.npy"
    cli = argparse.Namespace(checkpoint=str(ckpts[0]), output=str(out), neighbor_mode=None,
                             upsample=None, source="B", config=str(cfg_path))
    infer.main(cli, get_configuration(cfg_path), cfg_path)
    den = np.load(out)
    assert den.shape == (8, 12, Q, Q) and np.isfinite(den).all() and (den >= 0).all()
