#!/usr/bin/env python3
"""Proposal: supervise H&E models with pathway *function*, not DNA genotype.

Paladin-style models learn   H&E -> DNA alteration (binary, noisy: epigenetic loss and
functional VUS are mislabelled or dropped; phenocopies are found post hoc).
Here a transcriptomic "teacher" turns genotype into a continuous pathway-activity score,
and the H&E "student" regresses that score. At inference only H&E is needed.

Arms (identical patients, folds and seeds):
  DNA   – gated ABMIL, BCE on DNA label (VUS-like excluded from training)
  FUNC  – gated ABMIL, MSE on teacher score (all patients with RNA, incl. VUS-like)

Teachers:
  STK11 – LKB1-loss program: top-50 up / top-50 down genes (Welch t, truncating vs WT),
          derived INSIDE each training fold only; test-fold RNA is never used for training.
  NRF2  – literature NRF2 target-gene score (NQO1, AKR1C1-3, GCLM, TXNRD1, SRXN1, ...),
          fixed a priori (no label information).

Judges (never used for training): DNA AUROC, held-out RNA program, LKB1 protein (RPPA),
overall survival (Cox, stage/age-adjusted), and VUS-like scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from train_mil import GatedABMIL, load_bags, predict

RES = Path("results/functional")
EXPR = Path("data/raw/cbio/luad_mrna_rsem.txt")


def load_expression() -> pd.DataFrame:
    e = pd.read_csv(EXPR, sep="\t").dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol")
    e = e.set_index("Hugo_Symbol").drop(columns="Entrez_Gene_Id")
    e.columns = [c[:12] for c in e.columns]
    x = np.log2(e.T.groupby(level=0).mean().clip(lower=0) + 1)
    x = x.loc[:, (x.mean() > 3) & (x.std() > 0.5)]
    return (x - x.mean()) / x.std()


def stk11_teacher(z: pd.DataFrame, train_ids, y_train, n=50) -> tuple[pd.Series, list]:
    a = z.loc[train_ids]
    t = pd.Series(ttest_ind(a[y_train == 1], a[y_train == 0]).statistic, index=z.columns).drop("STK11", errors="ignore")
    up, dn = t.nlargest(n).index, t.nsmallest(n).index
    return z[up].mean(1) - z[dn].mean(1), list(up[:10])


def train(bags, target, mode, epochs, seed, max_tiles=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedABMIL()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(bags))
    if mode == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(target) - target.sum()) / max(target.sum(), 1)))
    else:
        mu, sd = target.mean(), target.std()
        target = (target - mu) / sd
        loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for i in rng.permutation(len(bags)):
            x = bags[i]
            if len(x) > max_tiles:
                x = x[rng.choice(len(x), max_tiles, replace=False)]
            out, _ = model(x)
            loss = loss_fn(out, torch.tensor(float(target[i])))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def raw_scores(model, bags):
    # predict() returns sigmoid(logit); both arms are compared by rank, so use the logit
    p, _ = predict(model, bags, torch.device("cpu"))
    return np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))


def run(pathway: str, lab: pd.DataFrame, z: pd.DataFrame, bags_all: dict, args) -> pd.DataFrame:
    dna_col = {"STK11": "STK11", "NRF2": "NRF2_pathway"}[pathway]
    d = lab[lab.index.isin(bags_all) & lab.index.isin(z.index)].copy()
    strat = d[dna_col].fillna(2).astype(int)  # stratify on WT / altered / VUS-like
    bags = {p: bags_all[p] for p in d.index}
    out = []
    for seed in range(args.seeds):
        folds = StratifiedKFold(5, shuffle=True, random_state=100 + seed).split(d, strat)
        for k, (tr, te) in enumerate(folds):
            tr_ids, te_ids = d.index[tr], d.index[te]
            if pathway == "STK11":
                lab_tr = d.loc[tr_ids, dna_col].dropna()
                teacher, top = stk11_teacher(z, lab_tr.index, lab_tr.values.astype(int))
            else:
                teacher, top = d["NRF2_score"], []
            # DNA arm: labelled training patients only
            dna_tr = d.loc[tr_ids, dna_col].dropna()
            m_dna = train([bags[p] for p in dna_tr.index], dna_tr.values.astype(float), "bce", args.epochs, seed * 10 + k)
            # FUNC arm: every training patient with RNA (VUS-like included)
            m_fun = train([bags[p] for p in tr_ids], teacher.loc[tr_ids].values.astype(float), "mse", args.epochs, seed * 10 + k)
            te_bags = [bags[p] for p in te_ids]
            out.append(pd.DataFrame({
                "patient": te_ids, "seed": seed, "fold": k,
                "score_dna": raw_scores(m_dna, te_bags),
                "score_func": [m_fun(b)[0].item() for b in te_bags],
                "teacher_heldout": teacher.loc[te_ids].values,  # computed from training-fold signature
            }))
            print(f"{pathway} seed {seed} fold {k} done  (teacher genes: {top[:5]})", flush=True)
    res = pd.concat(out)
    # ensemble over seeds: average ranks so the two arms are on comparable scales
    for c in ["score_dna", "score_func"]:
        res[c] = res.groupby("seed")[c].rank(pct=True)
    agg = res.groupby("patient")[["score_dna", "score_func", "teacher_heldout"]].mean()
    agg = agg.join(d[[dna_col, f"{'STK11' if pathway == 'STK11' else 'KEAP1'}_class", "RPPA_LKB1", "NRF2_score",
                      "OS_MONTHS", "OS_STATUS", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE"]])
    agg.to_csv(RES / f"oof_{pathway}.csv")
    return agg


def paired_boot(fn, df, n=2000, seed=0):
    """Bootstrap the FUNC - DNA difference of a metric over patients."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        try:
            diffs.append(fn(s, "score_func") - fn(s, "score_dna"))
        except ValueError:
            continue
    diffs = np.array(diffs)
    diffs = diffs[np.isfinite(diffs)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs <= 0).mean())


def cox_hr(df, col):
    from lifelines import CoxPHFitter

    d = df.dropna(subset=["OS_MONTHS", "OS_STATUS"]).copy()
    d["event"] = d.OS_STATUS.astype(str).str.startswith("1").astype(int)
    d["stage34"] = d.AJCC_PATHOLOGIC_TUMOR_STAGE.astype(str).str.contains("III|IV").astype(int)
    d["age"] = pd.to_numeric(d.AGE, errors="coerce")
    d["s"] = (d[col] - d[col].mean()) / d[col].std()
    d = d.dropna(subset=["age"])
    c = CoxPHFitter().fit(d[["OS_MONTHS", "event", "s", "stage34", "age"]], "OS_MONTHS", "event")
    r = c.summary.loc["s"]
    return {"hr_per_sd": float(r["exp(coef)"]), "lo": float(r["exp(coef) lower 95%"]), "hi": float(r["exp(coef) upper 95%"]), "p": float(r["p"])}


def evaluate(pathway: str, agg: pd.DataFrame) -> dict:
    dna_col = "STK11" if pathway == "STK11" else "NRF2_pathway"
    cls_col = "STK11_class" if pathway == "STK11" else "KEAP1_class"
    lab = agg[agg[dna_col].notna()]
    rep = {"n_patients": int(len(agg)), "n_dna_labelled": int(len(lab)), "n_pos": int(lab[dna_col].sum())}

    def auc(s, c):
        return roc_auc_score(s[dna_col], s[c])

    def rho_teacher(s, c):
        return spearmanr(s[c], s.teacher_heldout).statistic

    for arm in ["dna", "func"]:
        c = f"score_{arm}"
        rep[arm] = {"auroc_dna_label": auc(lab, c), "rho_heldout_rna_program": rho_teacher(agg, c)}
        if pathway == "STK11":
            r = agg.dropna(subset=["RPPA_LKB1"])
            rep[arm]["rho_lkb1_protein"] = spearmanr(r[c], r.RPPA_LKB1).statistic
            wt = r[r.STK11_class == "WT"]
            rep[arm]["rho_lkb1_protein_in_WT"] = spearmanr(wt[c], wt.RPPA_LKB1).statistic
        vus, wt = agg[agg[cls_col] == "VUS-like"][c], agg[agg[cls_col] == "WT"][c]
        rep[arm]["vus_auc_vs_wt"] = roc_auc_score(np.r_[np.ones(len(vus)), np.zeros(len(wt))], np.r_[vus, wt]) if len(vus) else None
        rep[arm]["cox"] = cox_hr(agg, c)
    rep["diff_auroc_ci"] = paired_boot(auc, lab)
    rep["diff_rho_rna_ci"] = paired_boot(rho_teacher, agg)
    if pathway == "STK11":
        r = agg.dropna(subset=["RPPA_LKB1"])
        rep["diff_rho_protein_ci"] = paired_boot(lambda s, c: -spearmanr(s[c], s.RPPA_LKB1).statistic, r)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pathways", default="STK11,NRF2")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    lab = pd.read_csv("data/processed/labels_luad.csv").set_index("patient")
    z = load_expression()
    bags_all = load_bags(lab.index)
    print(f"{len(bags_all)} LUAD bags, {len(set(bags_all) & set(z.index))} with RNA", flush=True)
    report = {}
    for pw in args.pathways.split(","):
        agg = run(pw, lab, z, bags_all, args)
        report[pw] = evaluate(pw, agg)
        print(json.dumps(report[pw], indent=1, default=float), flush=True)
    (RES / "report.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
