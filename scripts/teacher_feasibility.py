#!/usr/bin/env python3
"""Omics-only sanity check for the transcriptomic teacher (no slides needed, runs in seconds).

Is a pathway-activity score a better training target than the DNA label? Checks, on all
TCGA-LUAD patients with RNA-seq:
  * the in-fold LKB1-loss / KEAP1-loss programs recover DNA status out-of-fold
  * missense-only (VUS-like) cases score like truncating cases -> DNA labels drop real positives
  * WT cases with a high program score have lower LKB1 protein (RPPA) -> phenocopies exist
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_functional import load_expression, stk11_teacher


def main():
    z = load_expression()
    lab = pd.read_csv("data/processed/labels_luad.csv").set_index("patient")
    lab = lab[lab.index.isin(z.index)]
    report = {}
    for gene in ["STK11", "KEAP1"]:
        d = lab[lab[gene].notna()]
        y = d[gene].astype(int).values
        score = pd.Series(np.nan, index=lab.index)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(d, y):
            s, _ = stk11_teacher(z.drop(columns=[gene], errors="ignore"), d.index[tr], y[tr])
            score[d.index[te]] = s[d.index[te]]
        full, top = stk11_teacher(z.drop(columns=[gene], errors="ignore"), d.index, y)
        score = score.fillna(full)  # VUS-like: scored by the full-data program (never labelled)
        cls = lab[f"{gene}_class"]
        r = {"n": int(len(lab)), "oof_auroc_vs_dna": roc_auc_score(y, score[d.index]), "top_up_genes": top,
             "median_score": {c: float(score[cls == c].median()) for c in ["WT", "VUS-like", "oncogenic"]},
             "n_class": {c: int((cls == c).sum()) for c in ["WT", "VUS-like", "oncogenic"]},
             "p_vus_gt_wt": float(mannwhitneyu(score[cls == "VUS-like"], score[cls == "WT"], alternative="greater").pvalue)}
        if gene == "STK11":
            ok = lab.RPPA_LKB1.notna()
            wt = ok & (cls == "WT")
            hi = wt & (score > np.quantile(score[cls == "oncogenic"], 0.25))
            r["lkb1_protein"] = {
                "rho_score": spearmanr(score[ok], lab.RPPA_LKB1[ok]).statistic,
                "rho_dna_label": spearmanr((cls[ok] == "oncogenic").astype(int), lab.RPPA_LKB1[ok]).statistic,
                "n_wt_program_high": int(hi.sum()),
                "median_wt_high": float(lab.RPPA_LKB1[hi].median()), "median_wt_low": float(lab.RPPA_LKB1[wt & ~hi].median()),
                "p_wt_high_lt_low": float(mannwhitneyu(lab.RPPA_LKB1[hi], lab.RPPA_LKB1[wt & ~hi], alternative="less").pvalue),
            }
        else:
            r["rho_with_literature_nrf2_score"] = spearmanr(score, lab.NRF2_score, nan_policy="omit").statistic
        report[gene] = r
    Path("results").mkdir(exist_ok=True)
    Path("results/teacher_feasibility.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
