#!/usr/bin/env python3
"""Attention heatmaps + highest-attention tiles for the most confident true positives (cf. Fig. 4f)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

RES = Path("results")
FEATS = Path("data/processed/features")
THUMBS = Path("data/processed/thumbs")
SLIDES = Path(os.environ.get("SLIDES_DIR", "data/raw/slides"))


def heatmap(task: str, patient: str, ax_map, tile_axes, att: np.ndarray, prob: float):
    with h5py.File(FEATS / f"{patient}.h5") as h:
        coords, tile0 = h["coords"][:], int(h.attrs["tile0"])
        w0, h0 = h.attrs["dims"]
    thumb = cv2.cvtColor(cv2.imread(str(THUMBS / f"{patient}.jpg")), cv2.COLOR_BGR2RGB)
    sx, sy = thumb.shape[1] / w0, thumb.shape[0] / h0
    heat = np.full(thumb.shape[:2], np.nan)
    r = (att - att.min()) / (np.ptp(att) + 1e-9)
    for (x, y), v in zip(coords, r):
        heat[int(y * sy):int((y + tile0) * sy), int(x * sx):int((x + tile0) * sx)] = v
    ax_map.imshow(thumb)
    ax_map.imshow(np.ma.masked_invalid(heat), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    ax_map.set_axis_off()
    ax_map.set_title(f"{patient} · P={prob:.2f}", fontsize=8, loc="left")

    slide = openslide.OpenSlide(str(next(SLIDES.glob(f"{patient}*.svs"))))
    level = slide.get_best_level_for_downsample(tile0 / 224 + 1e-3)
    size = int(round(tile0 / slide.level_downsamples[level]))
    for ax, i in zip(tile_axes, np.argsort(att)[::-1]):
        x, y = coords[i]
        ax.imshow(slide.read_region((int(x), int(y)), level, (size, size)).convert("RGB").resize((224, 224)))
        ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="LUAD_STK11,LUAD_EGFR,LUAD_NRF2_pathway")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks.split(","):
        f = RES / "attention" / f"{task}.npz"
        if not f.exists():
            continue
        att = np.load(f)
        p = pd.read_csv(RES / f"pred_{task}_random.csv")
        p = p[(p.label == 1) & p.patient.isin(att.files)]
        p = p[[any(SLIDES.glob(f"{x}*.svs")) for x in p.patient]].sort_values("prob_abmil", ascending=False).head(args.n)
        fig = plt.figure(figsize=(7.4, 2.7 * len(p)))
        gs = fig.add_gridspec(len(p) * 2, 6)
        for k, row in enumerate(p.itertuples()):
            ax_map = fig.add_subplot(gs[2 * k:2 * k + 2, 0:3])
            tiles = [fig.add_subplot(gs[2 * k + j // 3, 3 + j % 3]) for j in range(6)]
            heatmap(task, row.patient, ax_map, tiles, att[row.patient], row.prob_abmil)
        fig.suptitle(f"{task.replace('_', ' ')}: attention (left) and top-6 attended 112-µm tiles (right)", fontsize=9, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"fig_attention_{task}.png", dpi=150)
        plt.close(fig)
        print("wrote", task)


if __name__ == "__main__":
    main()
