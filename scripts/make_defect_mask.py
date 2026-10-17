"""
make_defect_mask.py - build the detector defect mask once, at full detector size.

    python -m scripts.make_defect_mask --config configs/config_sources.yml --plot mask.png

Reads the ``defects`` section of the config:

    defects:
      output: masks/detector_mask.h5
      reference:                  # frames the mask is computed from (no binning)
        source: A                 # index or label in dataset.sources, or a file path
        crop_y: [150, 350]        # optional; defaults to that source's crop
        crop_x: [400, 600]
      detectors:                  # union of all; see core/defects.py for the types
        - {type: hot, factor: 3.0}
        - {type: dead, min_frac: 0.02}
      inpaint: {max_radius: 4, min_donors: 3}

Point ``dataset.defect_mask_file`` at the output and every dataset (training and
inference) is corrected with this mask while it is read, before any binning.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from core.defects import build_mask, make_inpainter, reference_stats, save_mask
from core.readers import open_source
from utils.opts import get_configuration


def _plain(x):
    """ruamel/AttrDict structures -> plain Python (for JSON provenance)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _reference_spec(cfg):
    ref = dict(cfg.defects.get("reference", {}) or {})
    src = ref.get("source", 0)
    sources = list(cfg.dataset.get("sources", []) or [])
    if isinstance(src, str) and not src.isdigit() and Path(src).suffix:
        base = {"path": src}                                  # a file path
    else:
        if not sources:
            raise ValueError("defects.reference.source needs dataset.sources (or give a path)")
        labels = [str(s.get("label", f"{i}:{Path(s['path']).stem}")) for i, s in enumerate(sources)]
        i = labels.index(src) if (isinstance(src, str) and not src.isdigit()) else int(src)
        base = dict(sources[i])
    spec = {
        "path": base["path"],
        "fmt": ref.get("fmt", base.get("fmt", cfg.dataset.get("fmt", "auto"))),
        "scan_shape": ref.get("scan_shape", base.get("scan_shape", cfg.dataset.get("scan_shape"))),
        "crop_y": ref.get("crop_y", base.get("crop_y")),
        "crop_x": ref.get("crop_x", base.get("crop_x")),
        "reader_options": dict(ref.get("reader_options", base.get("reader_options", {})) or {}),
    }
    return _plain(spec)


def main(cfg, plot=None):
    spec = _reference_spec(cfg)
    path = Path(spec["path"])
    path = path if path.is_absolute() else Path.cwd() / path
    detectors = _plain(list(cfg.defects.detectors))
    inpaint = _plain(dict(cfg.defects.get("inpaint", {}) or {}))
    zero_level = cfg.defects.get("zero_level", 0)

    with open_source(path, fmt=spec["fmt"], **spec["reader_options"]) as src:
        print(f"reference: {src.name} {path.name}, frames {src.n_frames} of {src.frame_shape}")
        stats = reference_stats(src, spec["scan_shape"], spec["crop_y"], spec["crop_x"],
                                zero_level=zero_level)
        reader = src.describe()

    mask, counts = build_mask(stats, detectors)
    fn = make_inpainter(mask, **inpaint)
    for c in counts:
        print(f"  {c['type']:13s} {c['flagged']:6d} px  {({k: v for k, v in c.items() if k not in ('type', 'flagged')})}")
    print(f"  union         {int(mask.sum()):6d} px of {mask.size} "
          f"({100 * mask.mean():.2f} %) from {stats['n_frames']} frames")
    if fn.n_orphans:
        print(f"  WARNING: {fn.n_orphans} masked pixels have no good neighbour within "
              f"max_radius={inpaint.get('max_radius', 4)} and will be left unchanged")

    info = {"reference": {**reader, **stats["region"], "n_frames": stats["n_frames"]},
            "detectors": counts, "inpaint": inpaint, "zero_level": zero_level}
    out = Path(cfg.defects.output)
    info = save_mask(out, mask, info, mean_pattern=stats["mean"])
    print(f"saved {out}  (digest {info['digest']})")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        logm = np.log10(stats["mean"] + 1e-3)
        ax[0].imshow(logm, cmap="gray")
        ax[0].set_title("reference mean pattern (log)")
        ax[1].imshow(logm, cmap="gray")
        ys, xs = np.nonzero(mask)
        ax[1].scatter(xs, ys, s=6, facecolors="none", edgecolors="#eb6834", linewidths=0.8)
        ax[1].set_title(f"masked: {int(mask.sum())} px")
        for a in ax:
            a.set_axis_off()
        fig.tight_layout()
        fig.savefig(plot, dpi=150)
        print(f"plot  {plot}")
    return mask, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_sources.yml")
    parser.add_argument("--plot", type=str, default=None, help="optional PNG of the mask")
    cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    main(get_configuration(Path(cli.config).resolve()), plot=cli.plot)
