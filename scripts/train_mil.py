#!/usr/bin/env python3
"""Patient-level genomic target prediction from WSI tile embeddings ("mini-Paladin").

For each (cohort, target) task:
  * gated attention-MIL (Ilse et al. 2018) and a mean-pool logistic-regression baseline
  * patient-level 5-fold CV, either stratified ("random") or grouped by TCGA tissue
    source site ("site") to guard against site/batch leakage (Howard et al. 2021)
  * out-of-fold probabilities for every labelled patient; patients with an uncertain
    label (VUS-like) are never trained on and are scored by the fold-model ensemble
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATS = Path("data/processed/features")
OUT = Path("results")

# (task name, cohort, label column). "NSCLC" pools LUAD+LUSC: the coarse-subtype setting.
TASKS = [
    ("LUAD_EGFR", "LUAD", "EGFR"),
    ("LUAD_KRAS", "LUAD", "KRAS"),
    ("LUAD_TP53", "LUAD", "TP53"),
    ("LUAD_STK11", "LUAD", "STK11"),
    ("LUAD_KEAP1", "LUAD", "KEAP1"),
    ("LUAD_NRF2_pathway", "LUAD", "NRF2_pathway"),
    ("LUSC_TP53", "LUSC", "TP53"),
    ("NSCLC_TP53", "NSCLC", "TP53"),
    ("NSCLC_NRF2_pathway", "NSCLC", "NRF2_pathway"),
    ("NSCLC_subtype_LUSC", "NSCLC", "is_LUSC"),
]


class GatedABMIL(nn.Module):
    def __init__(self, d_in=768, d=256, dropout=0.25):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout))
        self.att_v = nn.Sequential(nn.Linear(d, 128), nn.Tanh())
        self.att_u = nn.Sequential(nn.Linear(d, 128), nn.Sigmoid())
        self.att_w = nn.Linear(128, 1)
        self.head = nn.Linear(d, 1)

    def forward(self, x):  # x: (n_tiles, d_in)
        h = self.embed(x)
        a = torch.softmax(self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1), dim=0)
        return self.head((a[:, None] * h).sum(0)).squeeze(-1), a


def load_bags(patients):
    bags = {}
    for p in patients:
        f = FEATS / f"{p}.h5"
        if f.exists():
            with h5py.File(f) as h:
                bags[p] = torch.from_numpy(h["features"][:].astype(np.float32))
    return bags


def train_abmil(bags, y, epochs, seed, device, max_tiles=512):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedABMIL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    pos_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(bags))
    model.train()
    for _ in range(epochs):
        for i in rng.permutation(len(bags)):
            x = bags[i]
            if len(x) > max_tiles:  # random tile subsampling = cheap augmentation
                x = x[rng.choice(len(x), max_tiles, replace=False)]
            logit, _ = model(x.to(device))
            loss = loss_fn(logit, torch.tensor(float(y[i]), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, bags, device):
    probs, atts = [], []
    for x in bags:
        logit, a = model(x.to(device))
        probs.append(torch.sigmoid(logit).item())
        atts.append(a.cpu().numpy())
    return np.array(probs), atts


def bootstrap_auc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            stats.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def cohort_table():
    luad = pd.read_csv("data/processed/labels_luad.csv").assign(cohort="LUAD", is_LUSC=0.0)
    lusc = pd.read_csv("data/processed/labels_lusc.csv").assign(cohort="LUSC", is_LUSC=1.0)
    df = pd.concat([luad, lusc]).set_index("patient")
    df["site"] = df.index.str[5:7]  # TCGA tissue source site code
    return df


def run_task(name, cohort, col, df, bags_all, scheme, args, device):
    sub = df if cohort == "NSCLC" else df[df.cohort == cohort]
    sub = sub[sub.index.isin(bags_all)]
    labelled = sub[sub[col].notna()]
    uncertain = sub[sub[col].isna()]
    y = labelled[col].values.astype(int)
    if y.sum() < 8 or (len(y) - y.sum()) < 8:
        print(f"skip {name}: too few positives/negatives ({y.sum()}/{len(y)})")
        return None

    if scheme == "site":
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(labelled, y, groups=labelled.site))
    else:
        splits = list(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(labelled, y))

    bags = [bags_all[p] for p in labelled.index]
    ubags = [bags_all[p] for p in uncertain.index]
    mean_x = np.stack([b.mean(0).numpy() for b in bags])
    mean_u = np.stack([b.mean(0).numpy() for b in ubags]) if len(ubags) else None

    oof_mil, oof_lin = np.zeros(len(y)), np.zeros(len(y))
    unc_mil, unc_lin = np.zeros(len(ubags)), np.zeros(len(ubags))
    attention = {}
    for k, (tr, te) in enumerate(splits):
        lin = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))
        lin.fit(mean_x[tr], y[tr])
        oof_lin[te] = lin.predict_proba(mean_x[te])[:, 1]

        model = train_abmil([bags[i] for i in tr], y[tr], args.epochs, args.seed + k, device)
        p, a = predict(model, [bags[i] for i in te], device)
        oof_mil[te] = p
        for i, att in zip(te, a):
            attention[labelled.index[i]] = att
        if len(ubags):
            unc_lin += lin.predict_proba(mean_u)[:, 1] / len(splits)
            pu, au = predict(model, ubags, device)
            unc_mil += pu / len(splits)
            for pid, att in zip(uncertain.index, au):
                attention[pid] = attention.get(pid, 0) + att / len(splits)

    pred = pd.concat([
        pd.DataFrame({"patient": labelled.index, "label": y, "prob_abmil": oof_mil, "prob_linear": oof_lin, "set": "oof"}),
        pd.DataFrame({"patient": uncertain.index, "label": np.nan, "prob_abmil": unc_mil, "prob_linear": unc_lin, "set": "uncertain"}),
    ])
    pred.to_csv(OUT / f"pred_{name}_{scheme}.csv", index=False)
    if scheme == args.attention_scheme:
        np.savez_compressed(OUT / "attention" / f"{name}.npz", **attention)

    row = {"task": name, "cohort": cohort, "target": col, "scheme": scheme, "n": int(len(y)), "n_pos": int(y.sum())}
    for m, p in [("abmil", oof_mil), ("linear", oof_lin)]:
        lo, hi = bootstrap_auc(y, p)
        row.update({f"auroc_{m}": roc_auc_score(y, p), f"auroc_{m}_lo": lo, f"auroc_{m}_hi": hi,
                    f"auprc_{m}": average_precision_score(y, p)})
    print(f"{name:22s} {scheme:6s} n={row['n']:4d} pos={row['n_pos']:4d}  "
          f"ABMIL {row['auroc_abmil']:.3f} [{row['auroc_abmil_lo']:.2f}-{row['auroc_abmil_hi']:.2f}]  "
          f"linear {row['auroc_linear']:.3f}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--schemes", default="random,site")
    ap.add_argument("--attention-scheme", default="random")
    ap.add_argument("--tasks", default="all")
    args = ap.parse_args()
    (OUT / "attention").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # tiny model: CPU beats MPS here
    df = cohort_table()
    bags_all = load_bags(df.index)
    print(f"feature bags: {sum(df.loc[list(bags_all)].cohort == 'LUAD')} LUAD, {sum(df.loc[list(bags_all)].cohort == 'LUSC')} LUSC")

    tasks = TASKS if args.tasks == "all" else [t for t in TASKS if t[0] in args.tasks.split(",")]
    rows = []
    for scheme in args.schemes.split(","):
        for name, cohort, col in tasks:
            r = run_task(name, cohort, col, df, bags_all, scheme, args, device)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps(vars(args) | {"n_bags": len(bags_all)}, indent=2))


if __name__ == "__main__":
    main()
