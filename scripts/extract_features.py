#!/usr/bin/env python3
"""Tile H&E WSIs and embed tiles with a public pathology foundation model (Phikon).

Tiles are 112 um x 112 um (224 px at 0.5 um/px), matching the tile size in
Boehm et al. 2025. Tissue is found with Otsu thresholding on the saturation channel
of a low-resolution thumbnail. Runs in --watch mode alongside the downloader.
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
import torch
from transformers import ViTModel

TARGET_MPP = 0.5
TILE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def slide_mpp(slide: openslide.OpenSlide) -> float:
    for key in (openslide.PROPERTY_NAME_MPP_X, "aperio.MPP"):
        if key in slide.properties:
            return float(slide.properties[key])
    mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
    return 10.0 / mag  # 40x ~ 0.25 um/px


def tissue_tiles(slide: openslide.OpenSlide, tile0: int, min_tissue: float = 0.5):
    """Return level-0 coordinates of tiles with enough tissue, plus an RGB thumbnail."""
    w0, h0 = slide.dimensions
    px_per_tile = 8
    scale = tile0 / px_per_tile
    thumb = np.asarray(slide.get_thumbnail((int(w0 / scale), int(h0 / scale))).convert("RGB"))
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask &= ((hsv[..., 2] > 40).astype(np.uint8) * 255)  # drop black pen / scanner borders
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    coords = []
    for ty in range(h0 // tile0):
        for tx in range(w0 // tile0):
            patch = mask[ty * px_per_tile:(ty + 1) * px_per_tile, tx * px_per_tile:(tx + 1) * px_per_tile]
            if patch.size and (patch > 0).mean() >= min_tissue:
                coords.append((tx * tile0, ty * tile0))
    return np.array(coords, dtype=np.int64).reshape(-1, 2), thumb


class TileReader:
    """Thread-pooled tile reader; each thread keeps its own OpenSlide handle."""

    def __init__(self, path: Path, tile0: int, workers: int = 8):
        self.path, self.tile0 = str(path), tile0
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(workers)

    def _read(self, xy):
        if not hasattr(self.local, "slide"):
            slide = openslide.OpenSlide(self.path)
            level = slide.get_best_level_for_downsample(self.tile0 / TILE + 1e-3)
            self.local.slide, self.local.level = slide, level
            self.local.size = int(round(self.tile0 / slide.level_downsamples[level]))
        img = self.local.slide.read_region((int(xy[0]), int(xy[1])), self.local.level, (self.local.size,) * 2)
        arr = np.asarray(img.convert("RGB").resize((TILE, TILE)), dtype=np.float32) / 255.0
        return ((arr - MEAN) / STD).transpose(2, 0, 1)

    def batches(self, coords: np.ndarray, batch_size: int):
        chunks = [coords[i:i + batch_size] for i in range(0, len(coords), batch_size)]
        pending = self.pool.submit(lambda c: np.stack(list(self.pool.map(self._read, c))), chunks[0]) if chunks else None
        for nxt in chunks[1:] + [None]:
            batch = pending.result()
            # prefetch the next batch while the model runs on this one
            pending = self.pool.submit(lambda c: np.stack(list(self.pool.map(self._read, c))), nxt) if nxt is not None else None
            yield torch.from_numpy(batch)

    def close(self):
        self.pool.shutdown()


@torch.inference_mode()
def embed(model, device, path: Path, out: Path, thumbs: Path, max_tiles: int, seed: int):
    slide = openslide.OpenSlide(str(path))
    mpp = slide_mpp(slide)
    tile0 = int(round(TILE * TARGET_MPP / mpp))
    coords, thumb = tissue_tiles(slide, tile0)
    if len(coords) == 0:
        print(f"  no tissue found in {path.name}")
        return 0
    n_tissue = len(coords)
    if len(coords) > max_tiles:
        rng = np.random.default_rng(seed)
        coords = coords[np.sort(rng.choice(len(coords), max_tiles, replace=False))]
    reader = TileReader(path, tile0)
    feats = []
    for batch in reader.batches(coords, batch_size=128):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type != "cpu"):
            f = model(pixel_values=batch.to(device)).last_hidden_state[:, 0]
        feats.append(f.float().cpu().numpy())
    reader.close()
    feats = np.concatenate(feats).astype(np.float16)
    tmp = out.with_suffix(".tmp")
    with h5py.File(tmp, "w") as h:
        h["features"] = feats
        h["coords"] = coords
        h.attrs.update(mpp=mpp, tile0=tile0, n_tissue_tiles=n_tissue, dims=slide.dimensions)
    tmp.rename(out)
    cv2.imwrite(str(thumbs / f"{path.stem[:12]}.jpg"), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
    return len(coords)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", type=Path, default=Path("data/raw/slides"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--max-tiles", type=int, default=2000)
    ap.add_argument("--watch", action="store_true", help="keep polling for newly downloaded slides")
    ap.add_argument("--idle-exit", type=int, default=900, help="seconds without new slides before exiting")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: process every n-th slide (Slurm array jobs)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    thumbs = args.out.parent / "thumbs"
    thumbs.mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(device).eval()

    done, last_new = 0, time.time()
    while True:
        todo = [p for p in sorted(args.slides.glob("*.svs"), key=lambda p: p.stat().st_size)
                if not (args.out / f"{p.stem[:12]}.h5").exists() and not (args.out / f"{p.stem[:12]}.failed").exists()]
        shard_i, shard_n = map(int, args.shard.split("/"))
        todo = [p for p in todo if int(p.stem[8:12], 36) % shard_n == shard_i]
        for p in todo:
            t = time.time()
            try:
                n = embed(model, device, p, args.out / f"{p.stem[:12]}.h5", thumbs, args.max_tiles, seed=0)
            except Exception as exc:  # corrupt slide etc. — log and move on
                print(f"FAILED {p.name}: {exc}", flush=True)
                (args.out / f"{p.stem[:12]}.failed").touch()
                continue
            done += 1
            print(f"[{done}] {p.stem[:12]} {n} tiles in {time.time() - t:.1f}s", flush=True)
            last_new = time.time()
            if args.limit and done >= args.limit:
                return
        if not args.watch or time.time() - last_new > args.idle_exit:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
