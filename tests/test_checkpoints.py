"""Tests for core.checkpoints: provenance, compatibility checks, weight loading,
and the scratch -> weights -> infer chain of scripts.train / scripts.infer."""

import argparse
import json
import sys
import warnings

import numpy as np
import pytest

from core.checkpoints import (IncompatibleWeights, build_provenance, check_compatible,
                              expected_preprocessing, extract_state_dict, resolve_policy)

CFG = {"dataset": {"offset": 0, "neighbor_mode": "alternating_spatial",
                   "scan_mask": {"mode": "checkerboard", "block": 2}},
       "model": {"name": "blind-video-net-5d-cross", "unet": "original"},
       "train": {"lr": 1e-4, "loss": "poisson_nll", "milestones": [1, 2]}}
INFOS = [{"label": "A", "reader": "arina", "path": "a.h5", "q_bin": 2, "r_bin": 1,
          "defect_mask_digest": "abc"},
         {"label": "B", "reader": "arina", "path": "b.h5", "q_bin": 2, "r_bin": 1,
          "defect_mask_digest": "abc"}]


# --------------------------------------------------------------------------
# provenance and compatibility (no torch needed)
# --------------------------------------------------------------------------
def test_provenance_records_data_and_init():
    prov = build_provenance(CFG, INFOS, init={"mode": "weights", "from": "w.pth"})
    pre = prov["preprocessing"]
    assert pre["q_bin"] == 2 and pre["r_bin"] == 1 and pre["offset"] == 0
    assert pre["defect_mask_digest"] == "abc" and pre["scan_mask"]["mode"] == "checkerboard"
    assert [d["label"] for d in prov["data"]] == ["A", "B"]
    assert prov["init"] == {"mode": "weights", "from": "w.pth"}
    assert prov["model"]["name"] == "blind-video-net-5d-cross"
    assert {"version", "commit", "dirty"} <= set(prov["code"])
    assert "python" in prov["software"] and "created_utc" in prov


def test_mixed_binning_is_recorded_not_hidden():
    infos = [dict(INFOS[0]), dict(INFOS[1], r_bin=2)]
    assert build_provenance(CFG, infos)["preprocessing"]["r_bin"] == [1, 2]


def test_compatible_data_passes():
    prov = build_provenance(CFG, INFOS)
    assert check_compatible(prov, [INFOS[0]], offset=0) == []


@pytest.mark.parametrize("key,value", [("q_bin", 1), ("r_bin", 8)])
def test_binning_mismatch_by_policy(key, value):
    prov = build_provenance(CFG, INFOS)
    data = [dict(INFOS[0], **{key: value})]
    with pytest.raises(IncompatibleWeights) as e:
        check_compatible(prov, data)                         # strict is the default
    assert e.value.differences[0]["key"] == key
    with pytest.warns(UserWarning):
        diffs = check_compatible(prov, data, policy="warn")
    assert diffs == [{"key": key, "source": "A", "trained": INFOS[0][key], "data": value,
                      "action": "refuse"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # 'off' must not warn
        assert check_compatible(prov, data, policy="off") == diffs


def test_offset_refused_mask_change_noted():
    prov = build_provenance(CFG, INFOS)
    with pytest.raises(IncompatibleWeights):
        check_compatible(prov, INFOS, offset=1)
    with pytest.warns(UserWarning):
        diffs = check_compatible(prov, [dict(INFOS[0], defect_mask_digest="other")])
    assert [(d["key"], d["action"]) for d in diffs] == [("defect_mask_digest", "note")]


def test_flat_and_missing_provenance():
    flat = {"r_bin": 8, "q_bin": 2}                      # older, flat provenance dicts
    assert expected_preprocessing(flat)["r_bin"] == 8
    with pytest.raises(IncompatibleWeights):
        check_compatible(flat, [dict(INFOS[0], r_bin=1)])
    with pytest.warns(UserWarning):
        diffs = check_compatible(None, INFOS)
    assert [(d["key"], d["action"]) for d in diffs] == [("provenance", "note")]


def test_policy_resolution():
    assert resolve_policy() == "strict"
    assert resolve_policy(None, {"dataset": {"domain_policy": "warn"}}) == "warn"
    assert resolve_policy("off", {"dataset": {"domain_policy": "warn"}}) == "off"
    with pytest.raises(ValueError):
        resolve_policy("lenient")


def test_extract_state_dict_layouts():
    w = {"a.weight": np.zeros(2), "b.weight": np.ones(3)}
    for ckpt in ({"model": w}, {"model_state_dict": w}, {"state_dict": w}, w):
        assert extract_state_dict(ckpt) is w or extract_state_dict(ckpt) == w
    dp = extract_state_dict({"model": {"module.a.weight": np.zeros(2)}})
    assert list(dp) == ["a.weight"]
    with pytest.raises(KeyError):
        extract_state_dict({"epoch": 3, "loss": 0.1})


# --------------------------------------------------------------------------
# with torch: loading and freezing
# --------------------------------------------------------------------------
def _model():
    torch = pytest.importorskip("torch")
    import core.models as models
    from core.models import get_unet
    args = argparse.Namespace(model="blind-video-net-5d-cross", channels=1, out_channels=1,
                              bias=False, normal=False, blind_noise=False,
                              unet_cls=get_unet("original"))
    return torch, models.build_model(args)


def test_load_weights_strict_and_freeze(tmp_path):
    torch, model = _model()
    from core.checkpoints import file_digest, freeze_parameters, load_weights
    torch.save({"model_state_dict": model.state_dict(), "epoch": 7,
                "provenance": {"q_bin": 2, "r_bin": 8}}, tmp_path / "w.pth")
    _, other = _model()
    prov, src = load_weights(other, tmp_path / "w.pth")
    for (k, a), (_, b) in zip(model.state_dict().items(), other.state_dict().items()):
        assert torch.equal(a, b), k
    assert prov == {"q_bin": 2, "r_bin": 8} and src["from_epoch"] == 7
    assert src["from_digest"] == file_digest(tmp_path / "w.pth")

    sd = model.state_dict()
    sd.pop(next(iter(sd)))
    torch.save({"model": sd}, tmp_path / "broken.pth")
    with pytest.raises(RuntimeError):
        load_weights(other, tmp_path / "broken.pth")

    n = freeze_parameters(other, ["nin_"])
    assert 0 < n < sum(p.numel() for p in other.parameters())
    assert all(p.requires_grad == ("nin_" in name) for name, p in other.named_parameters())
    with pytest.raises(ValueError):
        freeze_parameters(other, ["no-such-layer"])


# --------------------------------------------------------------------------
# chain: train from scratch -> train from those weights -> infer (and refuse a mismatch)
# --------------------------------------------------------------------------
CONFIG = """
dataset:
  sources:
    - path: a.npy
      label: A
    - path: b.npy
      label: B
  q_bin: 1
  r_bin: {r_bin}
  scan_mask:
    mode: all
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
  {trainable}
output:
  save_dir: {save_dir}
"""


def test_scratch_weights_infer_chain(tmp_path, monkeypatch):
    for mod in ("torch", "ruamel.yaml", "torchinfo", "tensorboard"):
        pytest.importorskip(mod)
    import torch
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["test"])
    Q = 16
    np.save(tmp_path / "a.npy", np.random.default_rng(3).poisson(3, (12, 12, Q, Q)).astype(np.uint16))
    np.save(tmp_path / "b.npy", np.random.default_rng(4).poisson(3, (12, 12, Q, Q)).astype(np.uint16))

    from utils.opts import get_configuration
    from core.checkpoints import file_digest
    import scripts.train as train
    import scripts.infer as infer

    def cfg(name, r_bin=1, trainable="", save_dir="ckpt"):
        p = tmp_path / name
        p.write_text(CONFIG.format(r_bin=r_bin, trainable=trainable, save_dir=save_dir),
                     encoding="utf-8")
        return get_configuration(p), p

    def run(c, p, **kw):
        ns = argparse.Namespace(resume=None, weights=None, domain_policy=None)
        vars(ns).update(kw)
        train.main(c, p, ns)

    c1, p1 = cfg("scratch.yml", save_dir="ck_scratch")
    run(c1, p1)
    ck1 = sorted((tmp_path / "ck_scratch").rglob("checkpoint_1.pth"))[0]
    prov1 = torch.load(ck1, weights_only=False)["provenance"]
    assert prov1["init"] == {"mode": "scratch"}
    assert [d["label"] for d in prov1["data"]] == ["A", "B"]

    c2, p2 = cfg("ft.yml", trainable="trainable: [nin_]", save_dir="ck_ft")
    run(c2, p2, weights=str(ck1))
    ck2 = sorted((tmp_path / "ck_ft").rglob("checkpoint_0.pth"))[0]   # before any update
    saved = torch.load(ck2, weights_only=False)
    assert saved["provenance"]["init"]["mode"] == "weights"
    assert saved["provenance"]["init"]["from_digest"] == file_digest(ck1)
    start = torch.load(ck1, weights_only=False)["model"]
    for k, v in saved["model"].items():
        assert torch.equal(v, start[k]), k                  # really started from ck1

    # inference with the same preprocessing works; with r_bin=2 it is refused
    out = tmp_path / "den.npy"
    cli = argparse.Namespace(checkpoint=str(ck1), output=str(out), neighbor_mode=None,
                             upsample=None, source="A", config=str(p1), domain_policy=None)
    infer.main(cli, c1, p1)
    assert np.load(out).shape == (12, 12, Q, Q)
    rec = json.loads((tmp_path / "den.npy.provenance.json").read_text(encoding="utf-8"))
    assert rec["domain_shift"] == [] and rec["checkpoint_digest"] == file_digest(ck1)

    c3, p3 = cfg("rbin2.yml", r_bin=2)
    cli.config = str(p3)
    with pytest.raises(IncompatibleWeights):
        infer.main(cli, c3, p3)
    cli.domain_policy = "warn"                              # a deliberate cross-bin run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        infer.main(cli, c3, p3)
    assert np.load(out).shape == (6, 6, Q, Q)
    rec = json.loads((tmp_path / "den.npy.provenance.json").read_text(encoding="utf-8"))
    assert rec["domain_policy"] == "warn"
    assert {(d["key"], d["trained"], d["data"]) for d in rec["domain_shift"]} == {("r_bin", 1, 2)}

    # fine-tuning across bins records the shift in the new checkpoint's provenance
    c4, p4 = cfg("ft_rbin2.yml", r_bin=2, save_dir="ck_ft2")
    with pytest.raises(IncompatibleWeights):
        run(c4, p4, weights=str(ck1))
    run(c4, p4, weights=str(ck1), domain_policy="warn")
    ck4 = sorted((tmp_path / "ck_ft2").rglob("checkpoint_1.pth"))[0]
    init = torch.load(ck4, weights_only=False)["provenance"]["init"]
    assert init["domain_policy"] == "warn" and init["domain_shift"][0]["key"] == "r_bin"


def test_prebinned_sources_report_total_binning(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from core.data_setup import load_training_data
    monkeypatch.chdir(tmp_path)
    np.save(tmp_path / "cache.npy", np.ones((6, 6, 8, 8), np.float32))
    cfg = SimpleNamespace(dataset={"sources": [{"path": "cache.npy", "label": "c",
                                                "prebinned": {"q_bin": 2, "r_bin": 8}}]})
    _, _, infos = load_training_data(cfg, verbose=False)
    assert infos[0]["q_bin"] == 2 and infos[0]["r_bin"] == 8
    prov = build_provenance(CFG, [dict(INFOS[0], q_bin=2, r_bin=8)])
    assert check_compatible(prov, infos) == []           # 2 x 8 binning matches
    with pytest.raises(IncompatibleWeights):                 # without prebinned it would not
        check_compatible(prov, [dict(infos[0], q_bin=1, r_bin=1)])
