#!/usr/bin/env python3
"""Label-efficiency curves: DNA- vs function-supervised H&E models (from train_functional.py)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GREY, BLUE, INK = "#8a8f98", "#2f6db5", "#1f2430"


def main():
    rep = json.loads(Path("results/functional/report.json").read_text())
    rows = [r for pw in rep.values() for r in pw.get("learning_curve", [])]
    lc = pd.DataFrame(rows)
    pws = [p for p in ["STK11", "NRF2"] if p in set(lc.pathway)]
    fig, axes = plt.subplots(1, 2 * len(pws), figsize=(3.0 * 2 * len(pws), 2.9))
    for j, pw in enumerate(pws):
        for k, (metric, ylab) in enumerate([("auroc_dna_label", "AUROC vs DNA label"),
                                            ("rho_heldout_rna", "ρ vs held-out RNA program")]):
            ax = axes[2 * j + k]
            for arm, col, name in [("dna", GREY, "DNA-supervised"), ("func", BLUE, "Function-supervised")]:
                g = lc[(lc.pathway == pw) & (lc.arm == arm)].groupby("frac").agg(n=("n_train", "mean"), m=(metric, "mean"), s=(metric, "std")).reset_index()
                ax.errorbar(g.n, g.m, yerr=g.s, color=col, marker="o", ms=4, lw=1.6, capsize=2, label=name)
            ax.axhline(0.5 if "auroc" in metric else 0, color=GREY, ls=":", lw=0.8)
            ax.set_xlabel("Training patients")
            ax.set_ylabel(ylab)
            ax.set_title(f"{'STK11 / LKB1' if pw == 'STK11' else 'NRF2 pathway'}", loc="left", fontsize=9.5, color=INK)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0 and k == 0:
                ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig("reports/figures/fig_learning_curve.png", dpi=200, bbox_inches="tight")
    summary = lc.groupby(["pathway", "arm", "frac"])[["n_train", "auroc_dna_label", "rho_heldout_rna"]].mean().round(3)
    print(summary)
    summary.reset_index().to_csv("results/functional/learning_curve_summary.csv", index=False)


if __name__ == "__main__":
    main()
