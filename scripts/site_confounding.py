#!/usr/bin/env python3
"""How much of "H&E predicts genotype" in TCGA could be "H&E predicts the hospital"?

Mosaic uses TCGA as its external test set but does not examine tissue-source-site (TSS)
effects. Three measurements:
  1. Can slide embeddings identify the contributing site? (patient-level CV, multinomial LR)
  2. Does genotype prevalence differ by site? (chi-square per gene, LUAD)
  3. How well does *site alone* predict genotype? (one-hot site -> LR, CV AUROC) — the
     shortcut a model could exploit without learning any tumour morphology.
Also reports the random-CV vs site-grouped-CV AUROC drop from train_mil.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

FEATS = Path("data/processed/features")
RES = Path("results")
FIG = Path("reports/figures")
GENES = ["EGFR", "KRAS", "TP53", "STK11", "NRF2_pathway"]
C = {"ink": "#1f2430", "muted": "#8a8f98", "a": "#2f6db5", "b": "#d9822b", "d": "#b5452f"}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    luad = pd.read_csv("data/processed/labels_luad.csv").assign(cohort="LUAD")
    lusc = pd.read_csv("data/processed/labels_lusc.csv").assign(cohort="LUSC")
    lab = pd.concat([luad, lusc]).set_index("patient")
    lab["site"] = lab.index.str[5:7]
    pts = [p for p in lab.index if (FEATS / f"{p}.h5").exists()]
    X = np.stack([h5py.File(FEATS / f"{p}.h5")["features"][:].astype(np.float32).mean(0) for p in pts])
    d = lab.loc[pts]
    out = {}

    # 1. site identification from H&E embeddings (sites with >= 8 patients)
    counts = d.site.value_counts()
    keep = counts[counts >= 8].index
    m = d.site.isin(keep).values
    ys = d.site[m].values
    clf = make_pipeline(StandardScaler(), PCA(0.9, svd_solver="full"), LogisticRegression(C=0.1, max_iter=3000))
    pred = cross_val_predict(clf, X[m], ys, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    acc = float((pred == ys).mean())
    chance = float(pd.Series(ys).value_counts(normalize=True).iloc[0])
    out["site_id"] = {"n_patients": int(m.sum()), "n_sites": int(len(keep)), "accuracy": acc,
                      "majority_baseline": chance, "balanced_accuracy": float(balanced_accuracy_score(ys, pred)),
                      "balanced_chance": 1 / len(keep)}
    # same within LUAD only (removes the LUAD/LUSC difference)
    ml = m & (d.cohort == "LUAD").values
    cl = d.site[ml].value_counts()
    ml = ml & d.site.isin(cl[cl >= 8].index).values
    yl = d.site[ml].values
    pl = cross_val_predict(clf, X[ml], yl, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    out["site_id_luad"] = {"n_patients": int(ml.sum()), "n_sites": int(len(set(yl))),
                           "balanced_accuracy": float(balanced_accuracy_score(yl, pl)), "balanced_chance": 1 / len(set(yl))}

    # 2 + 3. genotype prevalence by site and site-only genotype prediction (LUAD, all labelled patients)
    L = lab[lab.cohort == "LUAD"]
    sc = L.site.value_counts()
    L = L[L.site.isin(sc[sc >= 8].index)]
    rows = []
    for g in GENES:
        s = L[L[g].notna()]
        y = s[g].astype(int).values
        tab = pd.crosstab(s.site, y)
        p_chi = chi2_contingency(tab)[1] if tab.shape[1] == 2 else np.nan
        enc = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(C=1.0, max_iter=2000))
        prob = cross_val_predict(enc, s[["site"]], y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        prev = s.groupby("site")[g].mean()
        rows.append({"gene": g, "n": int(len(s)), "n_pos": int(y.sum()), "chi2_p": float(p_chi),
                     "site_only_auroc": float(roc_auc_score(y, prob)), "prev_min": float(prev.min()), "prev_max": float(prev.max())})
    gt = pd.DataFrame(rows)
    out["genotype_by_site"] = gt.round(4).to_dict("records")

    # random vs site-grouped CV drop, if train_mil has run
    mf = RES / "metrics.csv"
    if mf.exists():
        mm = pd.read_csv(mf)
        piv = mm.pivot_table(index="task", columns="scheme", values="auroc_abmil")
        if {"random", "site"} <= set(piv.columns):
            piv["drop"] = piv["random"] - piv["site"]
            out["cv_drop"] = piv.round(3).reset_index().to_dict("records")

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), gridspec_kw={"width_ratios": [1.0, 1.2, 1.1]})
    ax = axes[0]
    vals = [out["site_id"]["balanced_accuracy"], out["site_id_luad"]["balanced_accuracy"]]
    chn = [out["site_id"]["balanced_chance"], out["site_id_luad"]["balanced_chance"]]
    ax.bar([0, 1], vals, color=C["a"], width=0.6)
    for i, c in enumerate(chn):
        ax.plot([i - 0.3, i + 0.3], [c, c], color=C["d"], ls="--", lw=1.2)
    ax.set_xticks([0, 1], [f"All\n{out['site_id']['n_sites']} sites", f"LUAD\n{out['site_id_luad']['n_sites']} sites"])
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("A  H&E embedding → hospital", loc="left", fontsize=9)
    ax.plot([], [], color=C["d"], ls="--", label="chance")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for i, r in gt.iterrows():
        ax.plot([r.prev_min, r.prev_max], [i, i], color=C["muted"], lw=2)
        ax.scatter([r.prev_min, r.prev_max], [i, i], color=[C["a"], C["d"]], s=18, zorder=3)
        ax.text(1.02, i, f"χ² p={r.chi2_p:.1g}", va="center", fontsize=7, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(gt)), [g.replace("_pathway", "") for g in gt.gene])
    ax.set_xlabel("Mutation prevalence across sites (min–max)")
    ax.set_title("B  Genotype prevalence varies by site", loc="left", fontsize=9)
    ax = axes[2]
    ax.barh(range(len(gt)), gt.site_only_auroc, color=C["b"], height=0.6)
    ax.axvline(0.5, color=C["muted"], ls=":", lw=0.8)
    ax.set_yticks(range(len(gt)), [g.replace("_pathway", "") for g in gt.gene])
    ax.set_xlim(0.4, 0.8)
    ax.set_xlabel("AUROC using site label only")
    ax.set_title("C  Shortcut: site alone → genotype", loc="left", fontsize=9)
    for sp in ("top", "right"):
        for a in axes:
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_site_confounding.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    RES.mkdir(exist_ok=True)
    (RES / "site_confounding.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
