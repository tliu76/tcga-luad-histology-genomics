#!/usr/bin/env python3
"""Schematics for the slides: pipeline overview and DNA- vs function-supervision."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, BLUE, ORANGE, RED, GREEN = "#1f2430", "#6b7280", "#2f6db5", "#d9822b", "#b5452f", "#3a9a6b"
OUT = Path("reports/figures")


def box(ax, x, y, w, h, text, fc, ec=None, fs=9, color=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12, color=color, lw=1.4, ls=ls))


def pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11), ax.set_ylim(0, 3.6), ax.axis("off")
    top = [("TCGA diagnostic\nH&E slides\n(open access)", "#eef2f7"),
           ("Otsu tissue mask\n112 µm tiles\n(224 px @ 0.5 µm/px)", "#eef2f7"),
           ("Phikon ViT-B\ntile embeddings\n(768-d)", "#eef2f7"),
           ("Gated attention-MIL\npatient-level CV\nrandom + site-grouped", "#e3ecf8")]
    for i, (t, c) in enumerate(top):
        box(ax, 0.2 + i * 2.25, 2.1, 1.95, 1.2, t, c, ec="#c9d3e0")
        if i:
            arrow(ax, 0.2 + i * 2.25 - 0.28, 2.7, 0.2 + i * 2.25 - 0.02, 2.7)
    box(ax, 9.3, 2.1, 1.55, 1.2, "Predicted\ngenotype /\npathway state", "#fdf1e4", ec="#efc99a", bold=True)
    arrow(ax, 9.02, 2.7, 9.28, 2.7)
    bottom = [("cBioPortal genomics\nmutations · GISTIC CNA (labels)", "#f3f4f6", 0.2),
              ("RNA-seq\npathway programs (teacher)", "#eaf5ef", 3.8),
              ("RPPA protein · overall survival\n(held-out judges)", "#fbecea", 7.4)]
    for t, c, x in bottom:
        box(ax, x, 0.3, 3.45, 0.85, t, c, ec="#d6d9de", fs=8.5)
    arrow(ax, 5.5, 1.17, 7.5, 2.08, color=MUTED)
    ax.text(0.2, 1.5, "Labels, never seen by the image encoder", fontsize=8, color=MUTED, style="italic")
    fig.savefig(OUT / "fig_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def supervision():
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(0, 11), ax.set_ylim(0, 3.9), ax.axis("off")
    ax.text(0.2, 3.6, "Paladin (DNA-supervised)", fontsize=11, fontweight="bold", color=INK)
    box(ax, 0.2, 2.2, 1.6, 1.0, "H&E slide", "#eef2f7", ec="#c9d3e0")
    arrow(ax, 1.85, 2.7, 2.35, 2.7)
    box(ax, 2.4, 2.2, 1.6, 1.0, "MIL model", "#e3ecf8", ec="#c9d3e0")
    arrow(ax, 4.05, 2.7, 4.55, 2.7)
    box(ax, 4.6, 2.2, 1.35, 1.0, "DNA label\n0 / 1", "#f3f4f6", ec="#d6d9de", bold=True)
    ax.text(0.2, 1.75, "✗ missense VUS: dropped or called WT\n✗ epigenetic / occult loss: labelled WT\n→ phenocopies found post hoc",
            fontsize=8.5, color=RED, va="top", linespacing=1.5)
    ax.plot([6.3, 6.3], [0.2, 3.7], color="#d6d9de", lw=1)
    ax.text(6.6, 3.6, "Proposed (function-supervised)", fontsize=11, fontweight="bold", color=BLUE)
    box(ax, 6.6, 2.2, 1.3, 1.0, "H&E slide", "#eef2f7", ec="#c9d3e0")
    arrow(ax, 7.95, 2.7, 8.3, 2.7)
    box(ax, 8.35, 2.2, 1.2, 1.0, "MIL\nstudent", "#e3ecf8", ec="#c9d3e0")
    arrow(ax, 9.6, 2.7, 9.9, 2.7)
    box(ax, 9.95, 2.2, 1.0, 1.0, "pathway\nactivity\n(cont.)", "#fdf1e4", ec="#efc99a", bold=True, fs=8.5)
    box(ax, 9.6, 0.45, 1.35, 0.95, "RNA teacher\n(training only)", "#eaf5ef", ec="#b9dcc7", fs=8.5)
    arrow(ax, 10.3, 1.42, 10.3, 2.17, color=GREEN)
    ax.text(6.6, 1.75, "✓ VUS carriers contribute\n✓ phenocopies labelled by function\n✓ inference is H&E only",
            fontsize=8.5, color=GREEN, va="top", linespacing=1.5)
    fig.savefig(OUT / "fig_supervision_schematic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    supervision()
    print("ok")
