"""
Checkpoint provenance, weight loading and compatibility checks.

Every checkpoint written by scripts/train.py carries a ``provenance`` dict:
what data it was trained on (readers, files, crops, binning, defect mask digest),
how training started (from scratch, from given weights, or resumed), the code
version and the software/hardware it ran on. Before weights are used on new
data, ``check_compatible`` compares the preprocessing the weights expect with
the preprocessing of the data. A different detector or scan binning or count offset
means the model is applied outside the regime it was trained in: refused under
the default 'strict' policy, run and recorded under 'warn' or 'off' (for
deliberate cross-domain experiments). The differences are returned as data, so
they can be stored with the result.

Older checkpoints without provenance still load, with a warning.
"""

import datetime
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

__all__ = ["build_provenance", "runtime_info", "extract_state_dict", "load_weights", "file_digest",
           "check_compatible", "resolve_policy", "expected_preprocessing",
           "freeze_parameters", "IncompatibleWeights", "POLICIES"]

REPO_ROOT = Path(__file__).resolve().parent.parent


class IncompatibleWeights(ValueError):
    """Weights were trained on differently preprocessed data."""


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _git():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=REPO_ROOT, capture_output=True, text=True,
                                    timeout=5).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _versions():
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "h5py", "hdf5plugin", "rsciio", "py4DSTEM"):
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = getattr(mod, "__version__", "?")
    return out


def _device():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        return {"type": "cuda", "name": torch.cuda.get_device_name(0)}
    return {"type": "cpu", "name": platform.processor() or platform.machine()}


def runtime_info():
    """Code version, git commit, software versions, host and device of this run."""
    try:
        version = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        version = None
    return {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "code": {"version": version, **_git()}, "software": _versions(),
            "host": platform.node(), "device": _device()}


def file_digest(path, chunk=1 << 20):
    """sha256 of a file (first 16 hex characters), e.g. of the starting weights."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def build_provenance(cfg, data_infos, init=None):
    """Plain-JSON provenance for a training run.

    cfg        : the run's config (AttrDict or dicts)
    data_infos : one dict per dataset from core.data_setup.load_training_data
    init       : {"mode": "scratch" | "weights" | "resume", "from": path, ...}
    """
    ds = cfg["dataset"]
    infos = _plain(list(data_infos))
    q_bins = {i.get("q_bin") for i in infos}
    r_bins = {i.get("r_bin") for i in infos}
    digests = {i.get("defect_mask_digest") for i in infos}
    try:
        version = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        version = None
    prov = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code": {"version": version, **_git()},
        "software": _versions(),
        "host": platform.node(),
        "device": _device(),
        "init": _plain(init or {"mode": "scratch"}),
        "model": _plain(dict(cfg["model"])),
        "training": {k: _plain(v) for k, v in dict(cfg["train"]).items()},
        "preprocessing": {
            "q_bin": q_bins.pop() if len(q_bins) == 1 else sorted(q_bins, key=str),
            "r_bin": r_bins.pop() if len(r_bins) == 1 else sorted(r_bins, key=str),
            "bin_mode": "sum",
            "offset": ds.get("offset", 0) or 0,
            "defect_mask_digest": digests.pop() if len(digests) == 1 else sorted(digests, key=str),
            "neighbor_mode": ds.get("neighbor_mode", "alternating_spatial"),
            "scan_mask": _plain(ds.get("scan_mask")),
        },
        "data": infos,
    }
    return _jsonable(prov)


def expected_preprocessing(provenance):
    """Preprocessing a checkpoint expects, from this repo's provenance or an older
    flat provenance dict (top-level q_bin / r_bin keys)."""
    if not provenance:
        return {}
    pre = dict(provenance.get("preprocessing", {}) or {})
    for key in ("q_bin", "r_bin", "offset", "defect_mask_digest"):
        if key not in pre and key in provenance:
            pre[key] = provenance[key]
    return pre


POLICIES = ("strict", "warn", "off")


def resolve_policy(cli_value=None, cfg=None):
    """CLI --domain-policy, else dataset.domain_policy, else 'strict'."""
    value = cli_value
    if value is None and cfg is not None:
        try:
            value = cfg["dataset"].get("domain_policy")
        except (KeyError, AttributeError, TypeError):
            value = None
    value = value or "strict"
    if value not in POLICIES:
        raise ValueError(f"domain policy must be one of {POLICIES}, got '{value}'")
    return value


def check_compatible(provenance, data_infos, offset=0, policy="strict"):
    """Compare the preprocessing the weights were trained with against the data.

    Returns a list of differences, each a dict::

        {"key": "r_bin", "source": "A", "trained": 8, "data": 1, "action": "refuse"}

    ``action`` is "refuse" for a different detector/scan binning or offset (the
    model is applied outside the regime it was trained in) and "note" for a
    different defect mask or missing provenance.

    policy
        strict  raise IncompatibleWeights if anything would be refused
        warn    warn for every difference and return them (record them!)
        off     return them without warning
    In every mode the returned list is meant to be stored in the run's
    provenance, so a cross-domain result is always labelled as such.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got '{policy}'")
    diffs = []
    if not provenance:
        diffs.append({"key": "provenance", "source": None, "trained": None, "data": None,
                      "action": "note",
                      "detail": "checkpoint has no provenance; binning and offset unchecked"})
    else:
        exp = expected_preprocessing(provenance)
        for info in data_infos:
            label = info.get("label", info.get("path", "data"))
            for key in ("q_bin", "r_bin"):
                if exp.get(key) is not None and info.get(key) is not None \
                        and exp[key] != info[key]:
                    diffs.append({"key": key, "source": label, "trained": exp[key],
                                  "data": info[key], "action": "refuse"})
            want, have = exp.get("defect_mask_digest"), info.get("defect_mask_digest")
            if want and have and want != have:
                diffs.append({"key": "defect_mask_digest", "source": label, "trained": want,
                              "data": have, "action": "note"})
        if exp.get("offset") is not None and (exp.get("offset") or 0) != (offset or 0):
            diffs.append({"key": "offset", "source": None, "trained": exp.get("offset"),
                          "data": offset, "action": "refuse"})

    def fmt(d):
        if d["key"] == "provenance":
            return d["detail"]
        where = f"{d['source']}: " if d["source"] else ""
        return f"{where}{d['key']} trained={d['trained']} data={d['data']} ({d['action']})"

    refused = [d for d in diffs if d["action"] == "refuse"]
    if policy == "strict" and refused:
        err = IncompatibleWeights(
            "weights used outside their training regime:\n  "
            + "\n  ".join(fmt(d) for d in refused)
            + "\n(--domain-policy warn runs anyway and records the differences)")
        err.differences = diffs
        raise err
    if policy != "off":
        for d in diffs:
            warnings.warn(fmt(d))
    return diffs


def extract_state_dict(checkpoint):
    """Model weights from any of the checkpoint layouts in use."""
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                sd = checkpoint[key]
                break
        else:
            if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
                sd = checkpoint                                # a raw state_dict
            else:
                raise KeyError(f"no model weights in checkpoint (keys: {list(checkpoint)})")
    else:
        raise TypeError(f"unexpected checkpoint type {type(checkpoint).__name__}")
    if sd and all(k.startswith("module.") for k in sd):      # saved from DataParallel
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def load_weights(model, path, map_location="cpu"):
    """Load weights into ``model`` (strict). Returns the checkpoint's provenance
    (or None) and a description of where the weights came from."""
    import torch

    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    sd = extract_state_dict(ckpt)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"{Path(path).name} does not match the configured model "
                           f"(check model.name / model.unet): {e}") from e
    prov = ckpt.get("provenance") if isinstance(ckpt, dict) else None
    source = {"from": str(path), "from_digest": file_digest(path),
              "from_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
              "from_created_utc": (prov or {}).get("created_utc")}
    return prov, source


def freeze_parameters(model, trainable_patterns):
    """Train only parameters whose name contains one of the patterns; returns the
    number of trainable parameters. Empty/None keeps everything trainable."""
    if not trainable_patterns:
        return sum(p.numel() for p in model.parameters())
    n = 0
    for name, p in model.named_parameters():
        p.requires_grad = any(s in name for s in trainable_patterns)
        n += p.numel() if p.requires_grad else 0
    if n == 0:
        raise ValueError(f"no parameter names contain any of {list(trainable_patterns)}")
    return n
