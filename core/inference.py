"""
Block-streamed, batched inference.

The dataset is read in blocks of output scan rows (plus one halo row on each
side, which is all the 3x3 / double-arrow neighbourhoods need), denoised in GPU
batches with the neighbour frames gathered on the device, and written block by
block. Memory is bounded by the block, not the dataset, so full-size scans run on
a laptop; speed comes from the batching.

The result is identical to denoising position by position with
core.dataloader.DataSetFromArray (the reference the tests compare against):
same neighbour geometry, same substitution of the centre frame for neighbours
outside the scan, and -- with ``edges="copy"`` -- the same edge handling as the
original infer.py (outer rows/columns copied from their inner neighbours).

Post-processing, as before: remove the count offset, clip at zero and, for
Poisson-NLL models, rescale by one global factor so the total counts equal the
input's. The factor is only known at the end, so writers apply it in a second,
block-wise pass. With float16 output this rounds twice (before and after the
rescaling): the stored values are within ~2 float16 steps (about 0.2 %) of the
float32 result; use float32 output where that matters.

Writers
    H5Writer     resumable: writes to <output>.partial with a per-row "done" record
                 and renames on completion; provenance in the file attributes;
                 gzip by default, so any HDF5 reader opens it without plugins
    NpyWriter    memory-mapped .npy (+ provenance JSON next to it)
    ArrayWriter  in memory (tests, small data)
"""

import datetime
import json
import os
import time
from pathlib import Path

import numpy as np

from core.stream import plan_binned, stream_binned

__all__ = ["denoise_source", "H5Writer", "NpyWriter", "ArrayWriter", "NEIGHBOR_OFFSETS"]

# (d_row, d_col) per input channel; must match core.dataloader.DataSetFromArray
NEIGHBOR_OFFSETS = {
    "spatial": [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)],
    "temporal": [(-1, -1), (0, -2), (1, -1), (0, -1), (0, 0), (0, 1), (-1, 1), (0, 2), (1, 1)],
}
FLOAT16_MAX = 65504.0


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
class ArrayWriter:
    """Keeps the result in memory (``.data``)."""

    def __init__(self, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        self.data = None

    def open(self, shape, run_key, block_rows):
        self.data = np.zeros(shape, self.dtype)
        self.done = np.zeros(shape[0], bool)
        self.sums = np.zeros((shape[0], 2), np.float64)
        return self.done.copy(), self.sums.copy()

    def write(self, r0, block, done_rows, row_sums):
        self.data[r0:r0 + len(block)] = block
        self.done[r0:r0 + len(block)] = True
        self.sums[r0:r0 + len(block)] = row_sums

    def blocks(self, block_rows):
        for r0 in range(0, self.data.shape[0], block_rows):
            yield r0, self.data[r0:r0 + block_rows]

    def rewrite(self, r0, block):
        self.data[r0:r0 + len(block)] = block

    def rescale_state(self, factor):
        """(factor to use, first row not yet rescaled); in-memory runs start at 0."""
        return factor, 0

    def rescaled(self, next_row):
        pass

    def abort(self):
        pass

    def finalize(self, provenance, extras):
        self.provenance, self.extras = provenance, extras


class NpyWriter(ArrayWriter):
    """Memory-mapped .npy; provenance in ``<output>.provenance.json``. Not resumable."""

    def __init__(self, path, dtype=np.float32):
        super().__init__(dtype)
        self.path = Path(path)

    def open(self, shape, run_key, block_rows):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = np.lib.format.open_memmap(self.path, mode="w+", dtype=self.dtype,
                                              shape=tuple(shape))
        self.done = np.zeros(shape[0], bool)
        self.sums = np.zeros((shape[0], 2), np.float64)
        return self.done.copy(), self.sums.copy()

    def abort(self):
        if getattr(self, "data", None) is not None:
            self.data.flush()

    def finalize(self, provenance, extras):
        self.data.flush()
        del self.data
        with open(str(self.path) + ".provenance.json", "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=1, default=str)
        np.savez(str(self.path) + ".summary.npz", **extras)


class H5Writer:
    """HDF5 output: dataset ``data`` plus ``intensity_image`` and ``mean_pattern``;
    the provenance is the JSON attribute ``provenance``.

    Written to ``<output>.partial``. If a run stops, the next run with the same
    run key (same weights, data and settings) continues with the missing rows.
    """

    def __init__(self, path, dtype=np.float16, compression="gzip"):
        """compression: 'gzip' (default; readable by any HDF5 tool without plugins),
        'lz4' (bitshuffle-LZ4: faster, but readers need ``import hdf5plugin``) or 'none'."""
        if compression not in ("gzip", "lz4", "none"):
            raise ValueError("compression must be 'gzip', 'lz4' or 'none'")
        self.path = Path(path)
        self.partial = Path(str(self.path) + ".partial")
        self.dtype = np.dtype(dtype)
        self.compression = compression

    def _filters(self):
        if self.compression == "lz4":
            import hdf5plugin
            return dict(hdf5plugin.Bitshuffle(cname="lz4"))
        if self.compression == "gzip":
            return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        return {}

    def open(self, shape, run_key, block_rows):
        import h5py
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.partial.exists():
            f = h5py.File(self.partial, "a")
            if f.attrs.get("run_key") == run_key and tuple(f["data"].shape) == tuple(shape):
                self.f = f
                done = f["rows_done"][...]
                print(f"resuming {self.partial.name}: {int(done.sum())}/{len(done)} rows done")
                return done, f["row_sums"][...]
            f.close()
            print(f"{self.partial.name} is from a different run; starting over")
            self.partial.unlink()
        self.f = h5py.File(self.partial, "w")
        self.f.attrs["run_key"] = run_key
        chunks = (min(block_rows, shape[0]), min(16, shape[1]), shape[2], shape[3])
        self.f.create_dataset("data", shape=shape, dtype=self.dtype, chunks=chunks,
                              **self._filters())
        self.f.create_dataset("rows_done", data=np.zeros(shape[0], bool))
        self.f.create_dataset("row_sums", data=np.zeros((shape[0], 2), np.float64))
        return np.zeros(shape[0], bool), np.zeros((shape[0], 2), np.float64)

    def write(self, r0, block, done_rows, row_sums):
        n = len(block)
        self.f["data"][r0:r0 + n] = block
        self.f["row_sums"][r0:r0 + n] = row_sums
        self.f["rows_done"][r0:r0 + n] = True      # only after the data is written
        self.f.flush()

    def blocks(self, block_rows):
        d = self.f["data"]
        for r0 in range(0, d.shape[0], block_rows):
            yield r0, d[r0:r0 + block_rows]

    def rewrite(self, r0, block):
        self.f["data"][r0:r0 + len(block)] = block

    def rescale_state(self, factor):
        """A rescale pass that was interrupted continues with its own factor, so no
        row is ever scaled twice."""
        if "rescale_factor" in self.f.attrs:
            return float(self.f.attrs["rescale_factor"]), int(self.f.attrs["rescaled_rows"])
        self.f.attrs["rescale_factor"] = factor
        self.f.attrs["rescaled_rows"] = 0
        return factor, 0

    def rescaled(self, next_row):
        self.f.attrs["rescaled_rows"] = int(next_row)
        self.f.flush()

    def abort(self):
        """Close the partial file so a later run can resume it."""
        f = getattr(self, "f", None)
        if f is not None:
            try:
                f.flush()
                f.close()
            except Exception:                  # already closed
                pass
            self.f = None

    def finalize(self, provenance, extras):
        for k in ("rescale_factor", "rescaled_rows"):
            if k in self.f.attrs:
                del self.f.attrs[k]
        for k, v in extras.items():
            if k in self.f:
                del self.f[k]
            self.f.create_dataset(k, data=v)
        for k in ("rows_done", "row_sums"):
            del self.f[k]
        del self.f.attrs["run_key"]
        self.f.attrs["provenance"] = json.dumps(provenance, default=str)
        self.f.close()
        os.replace(self.partial, self.path)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def _block_ranges(n_rows, block_rows):
    """[r0, r1) blocks; the last block is merged if it would have a single row, so
    row n-1 and its copy source n-2 are always processed together."""
    edges = list(range(0, n_rows, block_rows)) + [n_rows]
    if len(edges) > 2 and edges[-1] - edges[-2] == 1:
        edges.pop(-2)
    return list(zip(edges[:-1], edges[1:]))


def denoise_source(model, source, *, scan_shape=None, crop_y=None, crop_x=None, q_bin=1,
                   r_bin=1, q_crop=None, correct_fn=None, offset=0.0,
                   neighbor_mode="spatial", edges="copy", upsample=1, rescale=True,
                   batch_size=16, block_rows=8, writer=None, run_key="", device=None,
                   amp=False, progress=True, provenance=None):
    """Denoise one dataset block by block. Returns a summary dict.

    model       : trained network, (B, 9, Q, Q) -> (B, 1, Q, Q)
    source      : core.readers FrameSource (raw or already binned)
    edges       : "copy" (as the original infer.py) or "denoise" (every position,
                  outside neighbours replaced by the centre frame, as in training)
    rescale     : global count-conserving factor (Poisson-NLL models)
    batch_size  : positions per forward pass; halved automatically on GPU out-of-memory
    block_rows  : output scan rows read and denoised per block (>= 2)
    writer      : H5Writer / NpyWriter / ArrayWriter (default: ArrayWriter)
    amp         : mixed precision: False (full precision), "auto" or True (bfloat16 on
                  GPUs that support it, else float16), "bf16" or "fp16". Faster on recent
                  GPUs, not bit-identical; the dtype used is recorded in the summary.
    """
    import torch
    import torch.nn.functional as F

    if neighbor_mode not in NEIGHBOR_OFFSETS:
        raise ValueError(f"neighbor_mode must be one of {list(NEIGHBOR_OFFSETS)}")
    if edges not in ("copy", "denoise"):
        raise ValueError("edges must be 'copy' or 'denoise'")
    if amp not in (False, None, True, "auto", "bf16", "fp16"):
        raise ValueError("amp must be False, 'auto', 'bf16' or 'fp16'")
    block_rows = max(2, int(block_rows))
    device = device or next(model.parameters()).device
    writer = writer or ArrayWriter()
    model.eval()

    plan = plan_binned(source, scan_shape, crop_y, crop_x, q_bin, r_bin, q_crop)
    R, C, Qy, Qx = plan["out_shape"]
    if R < 3 or C < 3:
        raise ValueError(f"scan too small for inference: {(R, C)}")
    out_shape = (R, C, Qy * upsample, Qx * upsample)
    done, row_sums = writer.open(out_shape, run_key, block_rows)
    try:
        return _run_blocks(model, source, writer, done, row_sums, plan, out_shape,
                           scan_shape, crop_y, crop_x, q_bin, r_bin, q_crop, correct_fn,
                           offset, neighbor_mode, edges, upsample, rescale, batch_size,
                           block_rows, device, amp, progress, provenance)
    except BaseException:
        writer.abort()                         # keep the partial output resumable
        raise


def _run_blocks(model, source, writer, done, row_sums, plan, out_shape, scan_shape, crop_y,
                crop_x, q_bin, r_bin, q_crop, correct_fn, offset, neighbor_mode, edges,
                upsample, rescale, batch_size, block_rows, device, amp, progress, provenance):
    import torch
    import torch.nn.functional as F

    R, C, Qy, Qx = plan["out_shape"]
    offsets = NEIGHBOR_OFFSETS[neighbor_mode]
    dr = torch.tensor([o[0] for o in offsets], device=device)
    dc = torch.tensor([o[1] for o in offsets], device=device)
    use_amp = bool(amp) and device.type == "cuda"
    if amp == "fp16":
        amp_dtype = torch.float16
    elif amp == "bf16":
        amp_dtype = torch.bfloat16
    else:                                       # auto: bfloat16 where supported (Ampere+)
        amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.get_device_capability(device)[0] >= 8
                     else torch.float16)

    blocks = _block_ranges(R, block_rows)
    todo = [(r0, r1) for r0, r1 in blocks if not done[r0:r1].all()]

    def n_model_positions(r0, r1):
        if edges == "denoise":
            return (r1 - r0) * C
        return len([r for r in range(r0, r1) if 0 < r < R - 1]) * (C - 2)

    n_pos = sum(n_model_positions(r0, r1) for r0, r1 in todo)
    t0, n_done, bs = time.time(), 0, int(batch_size)
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=n_pos, desc="denoising", unit="pos", dynamic_ncols=True)

    for r0, r1 in todo:
        h0, h1 = max(0, r0 - 1), min(R, r1 + 1)               # one halo row each side
        raw = stream_binned(source, scan_shape, crop_y, crop_x, q_bin, r_bin, q_crop,
                            np.float32, correct_fn, rows=(h0, h1))
        in_sums = raw[r0 - h0:r1 - h0].sum(axis=(1, 2, 3), dtype=np.float64)
        blk = torch.from_numpy(raw + offset if offset else raw).to(device)
        flat = blk.reshape(-1, Qy, Qx)

        rr, cc = torch.meshgrid(torch.arange(r0, r1, device=device),
                                torch.arange(C, device=device), indexing="ij")
        rr, cc = rr.reshape(-1), cc.reshape(-1)
        if edges == "copy":                                    # outer frame filled below
            keep = (rr > 0) & (rr < R - 1) & (cc > 0) & (cc < C - 1)
            rr, cc = rr[keep], cc[keep]
        nr, nc = rr[:, None] + dr[None], cc[:, None] + dc[None]
        inside = (nr >= 0) & (nr < R) & (nc >= 0) & (nc < C)
        nr = torch.where(inside, nr, rr[:, None])              # outside -> centre frame
        nc = torch.where(inside, nc, cc[:, None])
        idx = (nr - h0) * C + nc                               # (P, 9) into the block

        res = torch.zeros((r1 - r0) * C, Qy * upsample, Qx * upsample, device=device)
        pos = (rr - r0) * C + cc
        i = 0
        with torch.inference_mode():
            while i < len(idx):
                j = min(len(idx), i + bs)
                try:
                    x = flat[idx[i:j]]                         # (b, 9, Qy, Qx)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                        enabled=use_amp):
                        y = model(x)
                    y = y.float()
                    if upsample > 1:
                        y = F.interpolate(y, scale_factor=upsample, mode="bicubic",
                                          align_corners=False)
                    res[pos[i:j]] = y[:, 0]
                except torch.cuda.OutOfMemoryError:
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    torch.cuda.empty_cache()
                    print(f"\nGPU out of memory: batch size reduced to {bs}")
                    continue
                if bar is not None:
                    bar.update(j - i)
                n_done += j - i
                i = j

        out = res.reshape(r1 - r0, C, Qy * upsample, Qx * upsample)
        if edges == "copy":                                    # same order as before
            if r0 == 0:
                out[0] = out[1]
            if r1 == R:
                out[-1] = out[-2]
            out[:, 0] = out[:, 1]
            out[:, -1] = out[:, -2]
        if offset:
            out = out - offset
        out = out.clamp_min(0).cpu().numpy()
        if writer.dtype == np.float16 and out.max() > FLOAT16_MAX:
            raise OverflowError("values exceed float16; use --out-dtype float32")
        sums = np.stack([in_sums, out.sum(axis=(1, 2, 3), dtype=np.float64)], axis=1)
        writer.write(r0, out.astype(writer.dtype), True, sums)
        row_sums[r0:r1] = sums
        done[r0:r1] = True
        del blk, flat, res, out

    if bar is not None:
        bar.close()
    elapsed = time.time() - t0

    # global count-conserving factor, then the summary images
    scale = 1.0
    if rescale:
        total_in, total_out = row_sums[:, 0].sum(), row_sums[:, 1].sum()
        if total_out > 0 and np.isfinite(total_out):
            scale = float(total_in / total_out)
        else:
            import warnings
            warnings.warn(f"denoised output sums to {total_out}; count rescaling skipped")
    scale, next_row = writer.rescale_state(scale)
    intensity = np.zeros((R, C), np.float64)
    mean_pattern = np.zeros(out_shape[2:], np.float64)
    for r0, block in writer.blocks(block_rows):
        b = block[...].astype(np.float32)
        if scale != 1.0 and r0 >= next_row:
            b = b * scale
            if writer.dtype == np.float16 and b.max() > FLOAT16_MAX:
                raise OverflowError("values exceed float16 after rescaling; "
                                    "use --out-dtype float32")
            writer.rewrite(r0, b.astype(writer.dtype))
            writer.rescaled(r0 + len(b))
        intensity[r0:r0 + len(b)] = b.sum(axis=(2, 3))
        mean_pattern += b.sum(axis=(0, 1))
    mean_pattern /= R * C

    summary = {"out_shape": list(out_shape), "dtype": str(writer.dtype),
               "rescale_factor": scale, "positions_denoised": int(n_done),
               "seconds": round(elapsed, 2),
               "positions_per_second": round(n_done / elapsed, 1) if elapsed > 0 else None,
               "final_batch_size": bs, "block_rows": block_rows, "edges": edges,
               "neighbor_mode": neighbor_mode, "upsample": int(upsample),
               "offset": offset, "amp": use_amp,
               "amp_dtype": ({torch.bfloat16: "bfloat16", torch.float16: "float16"}[amp_dtype]
                             if use_amp else None),
               "compression": getattr(writer, "compression", None),
               "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    record = dict(provenance or {})
    record["inference"] = {**record.get("inference", {}), **summary}
    writer.finalize(record, {"intensity_image": intensity.astype(np.float32),
                             "mean_pattern": mean_pattern.astype(np.float32)})
    return summary
