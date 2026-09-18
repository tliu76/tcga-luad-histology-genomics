#!/usr/bin/env python3
"""Figure: DNA labels vs transcriptomic pathway function (omics only, all LUAD with RNA)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_functional import load_expression, stk11_teacher

C = {"ink": "#1f2430", "muted": "#8a8f98", "wt": "#9aa3ad", "vus": "#d9822b", "onc": "#b5452f"}


def program_scores(z, lab, gene):
    d = lab[lab[gene].notna()]
    y = d[gene].astype(int).values
    zz = z.drop(columns=[gene], errors="ignore")
    score = pd.Series(np.nan, index=lab.index)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(d, y):
        s, _ = stk11_teacher(zz, d.index[tr], y[tr])
        score[d.index[te]] = s[d.index[te]]
    full, _ = stk11_teacher(zz, d.index, y)
    return score.fillna(full)


def main():
    z = load_expression()
    lab = pd.read_csv("data/processed/labels_luad.csv").set_index("patient")
    lab = lab[lab.index.isin(z.index)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1, 1]})
    rng = np.random.default_rng(0)
    for ax, gene, title in [(axes[0], "STK11", "LKB1-loss program"), (axes[1], "KEAP1", "KEAP1-loss / NRF2 program")]:
        s = program_scores(z, lab, gene)
        cls = lab[f"{gene}_class"]
        groups = [("WT", C["wt"]), ("VUS-like", C["vus"]), ("oncogenic", C["onc"])]
        for i, (g, col) in enumerate(groups):
            v = s[cls == g].values
            ax.boxplot([v], positions=[i], widths=0.55, showfliers=False, medianprops={"color": C["ink"], "lw": 1.5},
                       boxprops={"color": C["muted"]}, whiskerprops={"color": C["muted"]}, capprops={"color": C["muted"]})
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=7, color=col, alpha=0.55, lw=0)
        p = mannwhitneyu(s[cls == "VUS-like"], s[cls == "WT"], alternative="greater").pvalue
        ax.set_xticks([0, 1, 2], [f"WT\n(n={(cls == 'WT').sum()})", f"Missense\n(n={(cls == 'VUS-like').sum()})",
                                  f"Trunc/del\n(n={(cls == 'oncogenic').sum()})"])
        ax.set_ylabel("Program score (out-of-fold)")
        ax.set_title(f"{gene}: {title}\nmissense > WT, p = {p:.0e}", loc="left", fontsize=9.5)
    # LKB1 protein in WT: program-high vs low
    ax = axes[2]
    s = program_scores(z, lab, "STK11")
    cls = lab.STK11_class
    thr = np.quantile(s[cls == "oncogenic"], 0.25)
    ok = lab.RPPA_LKB1.notna()
    wt = ok & (cls == "WT")
    hi, lo = wt & (s > thr), wt & (s <= thr)
    for i, (m, col, name) in enumerate([(lo, C["wt"], "WT, program-low"), (hi, C["vus"], "WT, program-high"),
                                        (ok & (cls == "oncogenic"), C["onc"], "Truncating/del")]):
        v = lab.RPPA_LKB1[m].values
        ax.boxplot([v], positions=[i], widths=0.55, showfliers=False, medianprops={"color": C["ink"], "lw": 1.5},
                   boxprops={"color": C["muted"]}, whiskerprops={"color": C["muted"]}, capprops={"color": C["muted"]})
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=7, color=col, alpha=0.55, lw=0)
    p = mannwhitneyu(lab.RPPA_LKB1[hi], lab.RPPA_LKB1[lo], alternative="less").pvalue
    ax.set_xticks([0, 1, 2], [f"WT, low\n(n={lo.sum()})", f"WT, high\n(n={hi.sum()})",
                              f"Trunc/del\n(n={(ok & (cls == 'oncogenic')).sum()})"])
    ax.set_ylabel("LKB1 protein (RPPA)")
    ax.set_title(f"Phenocopies: WT with LKB1-loss program\nhave lower LKB1 protein, p = {p:.3f}", loc="left", fontsize=9.5)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("reports/figures/fig_teacher_omics.png", dpi=200, bbox_inches="tight")
    print("wrote reports/figures/fig_teacher_omics.png")


if __name__ == "__main__":
    main()
