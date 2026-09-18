#!/usr/bin/env python3
"""Downstream analyses mirroring Boehm et al. 2025 (Mosaic) on public TCGA data.

  fig_performance   – per-target AUROC (ABMIL vs linear; random vs site-grouped CV),
                      with the paper's internal-test AUROCs for reference (Table 1)
  fig_coarse        – coarse (NSCLC) vs granular (LUAD / LUSC) TP53 inference (Ext. Fig. 8)
  fig_stk11         – STK11: VUS-like scoring, phenocopy in WT via mRNA + RPPA LKB1 (Fig. 5d-h)
  fig_nrf2          – KEAP1/NRF2: VUS-like scoring validated by NRF2 target-gene expression
  fig_survival      – OS by inferred STK11 status (Fig. 5g)
  fig_clusters      – Leiden clusters of slide embeddings, mutation enrichment (Fig. 4a / ED Fig. 6)
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
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

RES = Path("results")
FIG = Path("reports/figures")
FEATS = Path("data/processed/features")
PAPER = {  # Boehm et al. 2025, Table 1 / text: internal test AUROC, primary LUAD (MSK-IMPACT, n~880)
    "LUAD_EGFR": 0.84, "LUAD_STK11": 0.92, "LUAD_TP53": 0.83,
}
C = {"ink": "#1f2430", "muted": "#8a8f98", "a": "#2f6db5", "b": "#d9822b", "c": "#3a9a6b", "d": "#b5452f", "grid": "#e6e8eb"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["muted"],
                     "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"], "figure.dpi": 150,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
SUMMARY: dict = {}


def labels():
    luad = pd.read_csv("data/processed/labels_luad.csv").assign(cohort="LUAD")
    lusc = pd.read_csv("data/processed/labels_lusc.csv").assign(cohort="LUSC")
    return pd.concat([luad, lusc]).set_index("patient")


def pred(task, scheme="random"):
    return pd.read_csv(RES / f"pred_{task}_{scheme}.csv").set_index("patient")


def fig_performance(m):
    tasks = [t for t in ["LUAD_EGFR", "LUAD_KRAS", "LUAD_TP53", "LUAD_STK11", "LUAD_KEAP1", "LUAD_NRF2_pathway", "NSCLC_subtype_LUSC"] if t in set(m.task)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(tasks))
    specs = [("random", "abmil", C["a"], "ABMIL · random CV"), ("site", "abmil", C["b"], "ABMIL · site-grouped CV"),
             ("random", "linear", C["muted"], "Mean-pool LR · random CV")]
    w = 0.26
    for j, (sch, mod, col, lab) in enumerate(specs):
        r = m[(m.scheme == sch)].set_index("task").reindex(tasks)
        v, lo, hi = r[f"auroc_{mod}"], r[f"auroc_{mod}_lo"], r[f"auroc_{mod}_hi"]
        ax.bar(x + (j - 1) * w, v, w * 0.92, color=col, label=lab)
        ax.errorbar(x + (j - 1) * w, v, yerr=[v - lo, hi - v], fmt="none", ecolor=C["ink"], lw=0.7, capsize=1.5)
    for i, t in enumerate(tasks):
        if t in PAPER:
            ax.plot([i - 1.5 * w, i + 1.5 * w], [PAPER[t]] * 2, color=C["d"], lw=1.4, ls="--")
    ax.plot([], [], color=C["d"], ls="--", label="Paladin (paper, MSK n≈880)")
    ax.axhline(0.5, color=C["muted"], lw=0.6, ls=":")
    ax.set_xticks(x, [t.replace("LUAD_", "").replace("NSCLC_subtype_LUSC", "LUAD/LUSC").replace("_pathway", "") for t in tasks])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Patient-level AUROC (5-fold OOF)")
    ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper left")
    ax.set_title("H&E → genomic target inference in TCGA-LUAD", loc="left", fontsize=10, color=C["ink"])
    fig.savefig(FIG / "fig_performance.png")
    plt.close(fig)


def fig_coarse(m):
    rows = m[(m.scheme == "random") & m.task.isin(["NSCLC_TP53", "LUAD_TP53", "LUSC_TP53"])].set_index("task")
    if len(rows) < 2:
        return
    order = [t for t in ["NSCLC_TP53", "LUAD_TP53", "LUSC_TP53"] if t in rows.index]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    v = rows.loc[order, "auroc_abmil"]
    ax.barh(range(len(order)), v, color=[C["d"], C["a"], C["c"]][:len(order)], height=0.6)
    ax.errorbar(v, range(len(order)), xerr=[v - rows.loc[order, "auroc_abmil_lo"], rows.loc[order, "auroc_abmil_hi"] - v],
                fmt="none", ecolor=C["ink"], lw=0.7, capsize=2)
    ax.set_yticks(range(len(order)), [f"{t.split('_')[0]}  (n={rows.loc[t, 'n']}, {rows.loc[t, 'n_pos'] / rows.loc[t, 'n']:.0%} mut)" for t in order])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1)
    ax.axvline(0.5, color=C["muted"], lw=0.6, ls=":")
    ax.set_xlabel("TP53 AUROC")
    ax.set_title("Coarse vs granular subtype", loc="left", fontsize=10)
    # what the coarse model learned: compare with a subtype-only predictor
    ax = axes[1]
    if (RES / "pred_NSCLC_subtype_LUSC_random.csv").exists():
        from sklearn.metrics import roc_auc_score
        a, b = pred("NSCLC_TP53"), pred("NSCLC_subtype_LUSC")
        j = a.join(b[["prob_abmil"]], rsuffix="_lusc").dropna(subset=["label"])
        vals = [roc_auc_score(j.label, j.prob_abmil), roc_auc_score(j.label, j.prob_abmil_lusc)]
        ax.barh([0, 1], vals, color=[C["d"], C["muted"]], height=0.55)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8.5)
        ax.set_yticks([0, 1], ["Pooled H&E model\n(TP53 target)", "Subtype score only\n(P(LUSC), no TP53 info)"])
        ax.invert_yaxis()
        ax.set_xlim(0.4, 1)
        ax.axvline(0.5, color=C["muted"], lw=0.6, ls=":")
        ax.set_xlabel("TP53 AUROC in pooled NSCLC")
        ax.set_title("Pooled signal ≈ subtype signal", loc="left", fontsize=10)
        SUMMARY["coarse_subtype_only_auroc"] = float(vals[1])
    fig.tight_layout()
    fig.savefig(FIG / "fig_coarse_vs_granular.png")
    plt.close(fig)
    SUMMARY["coarse"] = rows.loc[order, ["n", "n_pos", "auroc_abmil", "auroc_abmil_lo", "auroc_abmil_hi"]].round(3).to_dict("index")


def mw(a, b, alternative="two-sided"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return mannwhitneyu(a, b, alternative=alternative).pvalue


def strip(ax, groups, names, colors, ylabel):
    rng = np.random.default_rng(0)
    for i, (g, col) in enumerate(zip(groups, colors)):
        g = np.asarray(g, float)
        g = g[~np.isnan(g)]
        ax.boxplot([g], positions=[i], widths=0.5, showfliers=False, medianprops={"color": C["ink"]},
                   boxprops={"color": C["muted"]}, whiskerprops={"color": C["muted"]}, capprops={"color": C["muted"]})
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=6, color=col, alpha=0.6, lw=0)
    ax.set_xticks(range(len(groups)), [f"{n}\n(n={np.isfinite(np.asarray(g, float)).sum()})" for n, g in zip(names, groups)])
    ax.set_ylabel(ylabel)


def fig_stk11():
    if not (RES / "pred_LUAD_STK11_random.csv").exists():
        return
    lab = labels()
    p = pred("LUAD_STK11").join(lab[["STK11_class", "STK11_missense", "mRNA_STK11", "RPPA_LKB1", "KEAP1_class", "SMARCA4"]])
    wt = p[p.STK11_class == "WT"]
    thr = np.quantile(p[p.STK11_class == "oncogenic"].prob_abmil, 0.25)  # "looks like" an STK11-mutant slide
    wt_hi, wt_lo = wt[wt.prob_abmil >= thr], wt[wt.prob_abmil < thr]
    onc, vus = p[p.STK11_class == "oncogenic"], p[p.STK11_class == "VUS-like"]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    strip(axes[0], [wt.prob_abmil, vus.prob_abmil, onc.prob_abmil], ["WT", "Missense\n(VUS-like)", "Trunc/\ndeep del"],
          [C["muted"], C["b"], C["d"]], "P(STK11 oncogenic) from H&E")
    axes[0].set_title("A  Scoring missense variants", loc="left", fontsize=9)
    groups = [wt_lo, wt_hi, onc]
    names = ["WT\nH&E-low", "WT\nH&E-high", "Trunc/\ndeep del"]
    strip(axes[1], [g.mRNA_STK11 for g in groups], names, [C["muted"], C["b"], C["d"]], "STK11 mRNA (log2 RSEM)")
    p1 = mw(wt_hi.mRNA_STK11, wt_lo.mRNA_STK11, "less")
    axes[1].set_title(f"B  Phenocopy: mRNA  (p={p1:.1e})", loc="left", fontsize=9)
    strip(axes[2], [g.RPPA_LKB1 for g in groups], names, [C["muted"], C["b"], C["d"]], "LKB1 protein (RPPA)")
    p2 = mw(wt_hi.RPPA_LKB1, wt_lo.RPPA_LKB1, "less")
    axes[2].set_title(f"C  Phenocopy: protein  (p={p2:.1e})", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_stk11_phenocopy.png")
    plt.close(fig)

    SUMMARY["stk11"] = {
        "threshold": float(thr), "n_wt_high": int(len(wt_hi)), "n_wt_low": int(len(wt_lo)),
        "p_mrna_wt_high_lt_low": float(p1), "p_rppa_wt_high_lt_low": float(p2),
        "p_vus_gt_wt": float(mw(vus.prob_abmil, wt.prob_abmil, "greater")),
        "median_mrna": {k: float(np.nanmedian(g.mRNA_STK11)) for k, g in zip(["wt_low", "wt_high", "onc"], groups)},
        "keap1_or_smarca4_in_wt_high": float(((wt_hi.KEAP1_class != "WT") | (wt_hi.SMARCA4 == 1)).mean()),
        "keap1_or_smarca4_in_wt_low": float(((wt_lo.KEAP1_class != "WT") | (wt_lo.SMARCA4 == 1)).mean()),
        "top_vus": vus.sort_values("prob_abmil", ascending=False)[["STK11_missense", "prob_abmil", "mRNA_STK11", "RPPA_LKB1"]]
        .head(8).round(3).reset_index().to_dict("records"),
    }


def fig_nrf2():
    if not (RES / "pred_LUAD_NRF2_pathway_random.csv").exists():
        return
    lab = labels()
    p = pred("LUAD_NRF2_pathway").join(lab[["KEAP1_class", "KEAP1_missense", "NRF2_score"]])
    p = p[p.NRF2_score.notna()]
    wt = p[p.label == 0]
    alt = p[p.label == 1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.scatter(wt.prob_abmil, wt.NRF2_score, s=6, color=C["muted"], alpha=0.6, lw=0, label="NRF2-pathway WT")
    ax.scatter(alt.prob_abmil, alt.NRF2_score, s=8, color=C["d"], alpha=0.7, lw=0, label="KEAP1/NFE2L2/CUL3 altered")
    r_all = spearmanr(p.prob_abmil, p.NRF2_score)
    r_wt = spearmanr(wt.prob_abmil, wt.NRF2_score)
    ax.set_xlabel("P(NRF2 pathway altered) from H&E")
    ax.set_ylabel("NRF2 target-gene score (mRNA z)")
    ax.set_title(f"A  H&E score vs NRF2 activity\nρ_all={r_all.statistic:.2f}, ρ_WT={r_wt.statistic:.2f} (p={r_wt.pvalue:.1e})", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, markerscale=2)
    # KEAP1 missense: does the NRF2 target-gene program agree with 'altered'?
    ax = axes[1]
    k = p.KEAP1_class
    strip(ax, [p[k == "WT"].NRF2_score, p[k == "VUS-like"].NRF2_score, p[k == "oncogenic"].NRF2_score],
          ["KEAP1 WT", "KEAP1\nmissense", "KEAP1\ntrunc/del"], [C["muted"], C["b"], C["d"]], "NRF2 target-gene score")
    pv = mw(p[k == "VUS-like"].NRF2_score, p[k == "WT"].NRF2_score, "greater")
    ax.set_title(f"B  KEAP1 missense are functional  (p={pv:.1e})", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_nrf2.png")
    plt.close(fig)
    SUMMARY["nrf2"] = {"rho_all": float(r_all.statistic), "p_all": float(r_all.pvalue), "rho_wt": float(r_wt.statistic),
                       "p_wt": float(r_wt.pvalue), "p_keap1_missense_gt_wt": float(pv)}
    # KEAP1 missense scored by the KEAP1 (truncating) model, never trained on missense
    if (RES / "pred_LUAD_KEAP1_random.csv").exists():
        q = pred("LUAD_KEAP1").join(lab[["KEAP1_class", "KEAP1_missense", "NRF2_score"]])
        v = q[q.KEAP1_class == "VUS-like"].dropna(subset=["NRF2_score"])
        if len(v) > 5:
            r = spearmanr(v.prob_abmil, v.NRF2_score)
            SUMMARY["keap1_vus_prob_vs_nrf2"] = {"n": int(len(v)), "rho": float(r.statistic), "p": float(r.pvalue)}


def fig_survival():
    if not (RES / "pred_LUAD_STK11_random.csv").exists():
        return
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    lab = labels()
    p = pred("LUAD_STK11").join(lab[["STK11_class", "OS_MONTHS", "OS_STATUS", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE"]])
    p = p.dropna(subset=["OS_MONTHS", "OS_STATUS"])
    p["event"] = p.OS_STATUS.astype(str).str.startswith("1").astype(int)
    thr = SUMMARY.get("stk11", {}).get("threshold", 0.5)
    p["group"] = np.select(
        [(p.STK11_class == "oncogenic"), (p.STK11_class == "WT") & (p.prob_abmil >= thr), (p.STK11_class == "WT")],
        ["STK11 altered", "WT, H&E-high (phenocopy)", "WT, H&E-low"], "other")
    p = p[p.group != "other"]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for g, col in [("WT, H&E-low", C["a"]), ("WT, H&E-high (phenocopy)", C["b"]), ("STK11 altered", C["d"])]:
        s = p[p.group == g]
        KaplanMeierFitter().fit(s.OS_MONTHS, s.event, label=f"{g} (n={len(s)})").plot_survival_function(ax=ax, ci_show=False, color=col, lw=1.4)
    lr = multivariate_logrank_test(p.OS_MONTHS, p.group, p.event)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Months")
    ax.set_ylabel("Overall survival")
    ax.set_title(f"OS by inferred STK11 status (log-rank p={lr.p_value:.2g})", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig_stk11_survival.png")
    plt.close(fig)
    # Cox: H&E score as continuous covariate, adjusted for stage
    d = p.copy()
    d["stage34"] = d.AJCC_PATHOLOGIC_TUMOR_STAGE.astype(str).str.contains("III|IV").astype(int)
    d["he_score_z"] = (d.prob_abmil - d.prob_abmil.mean()) / d.prob_abmil.std()
    d["age"] = pd.to_numeric(d.AGE, errors="coerce")
    d = d.dropna(subset=["age"])
    cph = CoxPHFitter().fit(d[["OS_MONTHS", "event", "he_score_z", "stage34", "age"]], "OS_MONTHS", "event")
    s = cph.summary.loc["he_score_z"]
    SUMMARY["survival"] = {"logrank_p": float(lr.p_value), "cox_hr_per_sd": float(s["exp(coef)"]),
                           "cox_hr_lo": float(s["exp(coef) lower 95%"]), "cox_hr_hi": float(s["exp(coef) upper 95%"]), "cox_p": float(s["p"])}


def fig_clusters():
    import igraph as ig
    import leidenalg
    import umap
    from sklearn.neighbors import kneighbors_graph

    lab = labels()
    pts = [p for p in lab.index if (FEATS / f"{p}.h5").exists()]
    X = np.stack([h5py.File(FEATS / f"{p}.h5")["features"][:].astype(np.float32).mean(0) for p in pts])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    from sklearn.decomposition import PCA
    Z = PCA(30, random_state=0).fit_transform(X)
    emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(Z)
    A = kneighbors_graph(Z, 15, include_self=False)
    g = ig.Graph(edges=list(zip(*A.nonzero())), directed=False).simplify()
    cl = np.array(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1.0, seed=0).membership)
    d = lab.loc[pts].assign(u1=emb[:, 0], u2=emb[:, 1], cluster=cl)
    d.reset_index()[["patient", "cohort", "u1", "u2", "cluster"]].to_csv(RES / "slide_umap.csv", index=False)

    # Fisher's exact test per (gene, cluster) within LUAD, BH-FDR
    luad = d[d.cohort == "LUAD"]
    tests = []
    for gene in ["EGFR", "KRAS", "TP53", "STK11", "KEAP1", "NRF2_pathway", "SMARCA4"]:
        yv = luad[gene]
        ok = yv.notna()
        for c in sorted(luad.cluster.unique()):
            inc = (luad.cluster == c) & ok
            if inc.sum() < 10:
                continue
            t = [[int((yv[inc] == 1).sum()), int((yv[inc] == 0).sum())], [int((yv[ok & ~inc] == 1).sum()), int((yv[ok & ~inc] == 0).sum())]]
            orr, pv = fisher_exact(t)
            tests.append({"gene": gene, "cluster": int(c), "n_cluster": int(inc.sum()), "frac_in": t[0][0] / max(sum(t[0]), 1),
                          "frac_out": t[1][0] / max(sum(t[1]), 1), "odds_ratio": orr, "p": pv})
    tt = pd.DataFrame(tests)
    tt["q"] = multipletests(tt.p, method="fdr_bh")[1]
    tt.sort_values("p").to_csv(RES / "cluster_enrichment.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel("UMAP 1"), ax.set_ylabel("UMAP 2")
    ax = axes[0]
    for c, col in [("LUAD", C["a"]), ("LUSC", C["c"])]:
        s = d[d.cohort == c]
        ax.scatter(s.u1, s.u2, s=5, color=col, alpha=0.6, lw=0, label=c)
    ax.legend(frameon=False, markerscale=2, fontsize=7)
    ax.set_title("A  Slide embeddings by subtype", loc="left", fontsize=9)
    top = tt.sort_values("p").iloc[:2]
    for ax, (_, r), letter in zip(axes[1:], top.iterrows(), "BC"):
        ax.scatter(d.u1, d.u2, s=4, color=C["grid"], lw=0)
        s = luad[luad[r.gene].notna()]
        ax.scatter(s.u1, s.u2, s=5, c=np.where(s[r.gene] == 1, C["d"], C["muted"]), alpha=0.7, lw=0)
        m = d[d.cluster == r.cluster]
        ax.scatter(m.u1, m.u2, s=14, facecolors="none", edgecolors=C["ink"], lw=0.3)
        ax.set_title(f"{letter}  {r.gene} in LUAD; cluster {int(r.cluster)}\n{r.frac_in:.0%} vs {r.frac_out:.0%}, q={r.q:.1e}", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_clusters.png")
    plt.close(fig)
    SUMMARY["clusters"] = {"n_clusters": int(len(set(cl))), "n_sig_q05": int((tt.q < 0.05).sum()),
                           "top": tt.sort_values("p").head(6).round(4).to_dict("records")}


def fig_functional():
    """DNA-supervised vs function-supervised H&E models, judged on data neither saw."""
    rep_f = RES / "functional" / "report.json"
    if not rep_f.exists():
        return
    rep = json.loads(rep_f.read_text())
    panels = [(pw, rep[pw]) for pw in ["STK11", "NRF2"] if pw in rep]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.3] * len(panels) + [1]})
    for ax, (pw, r) in zip(axes, panels):
        metrics = [("auroc_dna_label", "DNA\nAUROC", 0.5), ("rho_heldout_rna_program", "ρ held-out\nRNA program", 0),
                   ("vus_auc_vs_wt", "VUS vs WT\nAUROC", 0.5)]
        if pw == "STK11":
            metrics.insert(2, ("rho_lkb1_protein", "−ρ LKB1\nprotein", 0))
        x = np.arange(len(metrics))
        for j, (arm, col, name) in enumerate([("dna", C["muted"], "DNA-supervised"), ("func", C["a"], "Function-supervised")]):
            v = [(-1 if k == "rho_lkb1_protein" else 1) * (r[arm][k] or np.nan) for k, _, _ in metrics]
            ax.bar(x + (j - 0.5) * 0.38, v, 0.36, color=col, label=name)
        for i, (_, _, base) in enumerate(metrics):
            ax.plot([i - 0.4, i + 0.4], [base, base], color=C["ink"], lw=0.6, ls=":")
        ax.set_xticks(x, [m[1] for m in metrics], fontsize=7.5)
        ax.set_title(f"{'STK11 / LKB1' if pw == 'STK11' else 'NRF2 pathway'}  (n={r['n_patients']})", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    # scatter: function-supervised H&E score vs held-out RNA program (STK11)
    ax = axes[-1]
    f = RES / "functional" / "oof_STK11.csv"
    if f.exists():
        d = pd.read_csv(f)
        col = d.STK11_class.map({"WT": C["muted"], "VUS-like": C["b"], "oncogenic": C["d"]})
        ax.scatter(d.score_func, d.teacher_heldout, s=6, c=col, alpha=0.7, lw=0)
        for k, c in [("WT", C["muted"]), ("VUS-like", C["b"]), ("oncogenic", C["d"])]:
            ax.scatter([], [], s=10, color=c, label=k)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_xlabel("H&E score (function-supervised, OOF rank)")
        ax.set_ylabel("LKB1-loss RNA program (held-out)")
        ax.set_title("STK11: H&E reads out pathway state", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_functional_supervision.png")
    plt.close(fig)
    SUMMARY["functional"] = rep


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(RES / "metrics.csv")
    SUMMARY["metrics"] = m.round(3).to_dict("records")
    for f in (fig_performance, fig_coarse):
        f(m)
    for f in (fig_stk11, fig_nrf2, fig_survival, fig_clusters, fig_functional):
        try:
            f()
        except Exception as exc:  # keep going; small cohorts can make one analysis degenerate
            print(f"{f.__name__} failed: {exc!r}")
    (RES / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))
    print(json.dumps({k: v for k, v in SUMMARY.items() if k != "metrics"}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
