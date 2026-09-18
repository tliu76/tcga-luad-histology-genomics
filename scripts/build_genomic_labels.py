#!/usr/bin/env python3
"""Pull TCGA-LUAD PanCancer Atlas genomics from cBioPortal and derive patient-level targets.

Targets mirror the Paladin setup (oncogenic alteration per gene, plus a pathway-level
target in the style of Sanchez-Vega et al., Cell 2018). OncoKB requires a token, so
"oncogenic" is approximated with explicit, documented rules:

  * Oncogenes (EGFR, KRAS): known hotspots / canonical activating alleles only.
  * Tumor suppressors (TP53, STK11, KEAP1): truncating/splice mutations or deep deletion.
    TP53 missense is also counted (the vast majority are oncogenic).
    STK11/KEAP1 missense-only cases are labelled "uncertain" (VUS-like): excluded from
    training and scored afterwards, mirroring Fig. 5 of Boehm et al. 2025.
  * NRF2 pathway (KEAP1 / NFE2L2 / CUL3), gene lists from Sanchez-Vega et al. 2018.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://www.cbioportal.org/api"
OUT = Path("data/processed")

MUT_GENES = ["EGFR", "KRAS", "TP53", "STK11", "KEAP1", "NFE2L2", "CUL3", "SMARCA4", "NKX2-1"]
# NRF2 transcriptional targets used as an orthogonal read-out of pathway activity
NRF2_TARGETS = ["NQO1", "AKR1C1", "AKR1C2", "AKR1C3", "GCLM", "TXNRD1", "SRXN1", "G6PD", "ME1", "CYP4F11", "OSGIN1", "ABCC2"]
EXPR_GENES = ["STK11", "EGFR", "NKX2-1"] + NRF2_TARGETS

TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"}
NONSYN = TRUNCATING | {"Missense_Mutation", "In_Frame_Del", "In_Frame_Ins", "Splice_Region"}


def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def post(path, body, **params):
    r = requests.post(f"{API}{path}", json=body, params=params, timeout=300)
    r.raise_for_status()
    return r.json()


def entrez_ids(symbols):
    genes = post("/genes/fetch", symbols, geneIdType="HUGO_GENE_SYMBOL")
    return {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in genes}


def residue(protein_change: str) -> int | None:
    m = re.search(r"(\d+)", protein_change or "")
    return int(m.group(1)) if m else None


def egfr_activating(pc: str, vc: str) -> bool:
    pos = residue(pc)
    if pc in {"L858R", "L861Q", "S768I", "T790M"} or pc.startswith("G719"):
        return True
    if pos is None:
        return False
    if vc in {"In_Frame_Del", "In_Frame_Ins"} and 729 <= pos <= 761:  # exon 19
        return True
    if vc == "In_Frame_Ins" and 762 <= pos <= 775:  # exon 20 insertions
        return True
    return False


def kras_activating(pc: str) -> bool:
    return residue(pc) in {12, 13, 59, 61, 117, 146} and not pc.endswith("=")


def build(STUDY: str, tag: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = entrez_ids(sorted(set(MUT_GENES + EXPR_GENES)))
    sym = {v: k for k, v in ids.items()}

    # --- samples / clinical -------------------------------------------------------
    samples = get(f"/sample-lists/{STUDY}_sequenced")["sampleIds"]
    clin = get(f"/studies/{STUDY}/clinical-data", clinicalDataType="SAMPLE", projection="SUMMARY", pageSize=100000)
    pclin = get(f"/studies/{STUDY}/clinical-data", clinicalDataType="PATIENT", projection="SUMMARY", pageSize=100000)
    cs = pd.DataFrame(clin).pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    cp = pd.DataFrame(pclin).pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first")
    clinical = cp.join(cs, how="outer", rsuffix="_s")

    # --- mutations ------------------------------------------------------------------
    muts = pd.DataFrame(post(f"/molecular-profiles/{STUDY}_mutations/mutations/fetch",
                             {"sampleListId": f"{STUDY}_sequenced", "entrezGeneIds": [ids[g] for g in MUT_GENES]},
                             projection="DETAILED"))
    muts["gene"] = muts["entrezGeneId"].map(sym)
    muts = muts[["patientId", "sampleId", "gene", "proteinChange", "mutationType", "tumorAltCount", "tumorRefCount"]]
    muts.to_csv(OUT / f"mutations_selected_genes_{tag}.csv", index=False)

    # --- copy number (GISTIC: -2 deep del, 2 amp) -------------------------------------
    cna = pd.DataFrame(post(f"/molecular-profiles/{STUDY}_gistic/discrete-copy-number/fetch",
                            {"sampleListId": f"{STUDY}_cna", "entrezGeneIds": [ids[g] for g in MUT_GENES]},
                            discreteCopyNumberEventType="HOMDEL_AND_AMP"))
    cna["gene"] = cna["entrezGeneId"].map(sym)
    deepdel = cna[cna.alteration == -2].groupby("gene")["patientId"].apply(set).to_dict()
    amp = cna[cna.alteration == 2].groupby("gene")["patientId"].apply(set).to_dict()

    # --- expression + RPPA ------------------------------------------------------------
    expr = pd.DataFrame(post(f"/molecular-profiles/{STUDY}_rna_seq_v2_mrna/molecular-data/fetch",
                             {"sampleListId": f"{STUDY}_rna_seq_v2_mrna", "entrezGeneIds": [ids[g] for g in EXPR_GENES]}))
    expr["gene"] = expr["entrezGeneId"].map(sym)
    expr = expr.pivot_table(index="patientId", columns="gene", values="value", aggfunc="mean")
    log_expr = np.log2(expr + 1)
    z = (log_expr - log_expr.mean()) / log_expr.std()
    rna = pd.DataFrame({f"mRNA_{g}": log_expr[g] for g in ["STK11", "EGFR", "NKX2-1"]})
    rna["NRF2_score"] = z[[g for g in NRF2_TARGETS if g in z]].mean(axis=1)

    try:
        rp = pd.DataFrame(post(f"/molecular-profiles/{STUDY}_rppa/molecular-data/fetch",
                               {"sampleListId": f"{STUDY}_rppa", "entrezGeneIds": [ids["STK11"]]}))
        rna["RPPA_LKB1"] = rp.groupby("patientId")["value"].mean()
    except requests.HTTPError as e:
        print("RPPA unavailable:", e)

    # --- labels -----------------------------------------------------------------------
    patients = sorted(set(clinical.index) & set(s[:12] for s in samples))
    lab = pd.DataFrame(index=patients)
    nonsyn = muts[muts.mutationType.isin(NONSYN)]

    def pts(frame):
        return set(frame.patientId)

    g = {k: v for k, v in nonsyn.groupby("gene")}
    empty = nonsyn.iloc[:0]
    egfr = g.get("EGFR", empty)
    kras = g.get("KRAS", empty)
    lab["EGFR"] = lab.index.isin(pts(egfr[[egfr_activating(p, v) for p, v in zip(egfr.proteinChange, egfr.mutationType)]]))
    lab["KRAS"] = lab.index.isin(pts(kras[kras.proteinChange.map(kras_activating)]))
    lab["TP53"] = lab.index.isin(pts(g.get("TP53", empty)) | deepdel.get("TP53", set()))

    for gene in ["STK11", "KEAP1"]:
        gm = g.get(gene, empty)
        onc = pts(gm[gm.mutationType.isin(TRUNCATING)]) | deepdel.get(gene, set())
        anymut = pts(gm)
        col = pd.Series(0.0, index=lab.index)
        col[col.index.isin(anymut - onc)] = np.nan  # missense/in-frame only -> uncertain
        col[col.index.isin(onc)] = 1.0
        lab[gene] = col
        lab[f"{gene}_class"] = np.where(lab.index.isin(onc), "oncogenic", np.where(lab.index.isin(anymut), "VUS-like", "WT"))
        lab[f"{gene}_missense"] = [";".join(gm[gm.patientId == p].proteinChange) for p in lab.index]

    nfe2l2 = g.get("NFE2L2", empty)
    nfe2l2_hot = nfe2l2[nfe2l2.proteinChange.map(lambda p: (residue(p) or 0) in range(24, 35) or (residue(p) or 0) in range(75, 83))]
    cul3 = g.get("CUL3", empty)
    lab["NRF2_pathway"] = lab.index.isin(
        pts(g.get("KEAP1", empty)) | deepdel.get("KEAP1", set()) | pts(nfe2l2_hot) | amp.get("NFE2L2", set())
        | pts(cul3[cul3.mutationType.isin(TRUNCATING)]) | deepdel.get("CUL3", set()))
    lab["SMARCA4"] = lab.index.isin(pts(g.get("SMARCA4", empty)[lambda d: d.mutationType.isin(TRUNCATING)]) | deepdel.get("SMARCA4", set()))

    for c in ["EGFR", "KRAS", "TP53", "NRF2_pathway", "SMARCA4"]:
        lab[c] = lab[c].astype(float)

    keep = ["OS_MONTHS", "OS_STATUS", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AGE", "SEX", "SUBTYPE", "TMB_NONSYNONYMOUS",
            "FRACTION_GENOME_ALTERED", "ANEUPLOIDY_SCORE", "TISSUE_SOURCE_SITE", "ONCOTREE_CODE"]
    out = lab.join(clinical[[c for c in keep if c in clinical]]).join(rna)
    out.index.name = "patient"
    out.to_csv(OUT / f"labels_{tag}.csv")

    print(f"{len(out)} sequenced {tag.upper()} patients")
    for c in ["EGFR", "KRAS", "TP53", "STK11", "KEAP1", "NRF2_pathway", "SMARCA4"]:
        print(f"  {c:13s} pos={int(np.nansum(out[c])):4d}  neg={int((out[c] == 0).sum()):4d}  uncertain={int(out[c].isna().sum())}")
    print("  RPPA LKB1 available for", out.get("RPPA_LKB1", pd.Series(dtype=float)).notna().sum(), "patients")


def download_expression(dest: Path = Path("data/raw/cbio/luad_mrna_rsem.txt")) -> None:
    """Genome-wide LUAD RNA-seq (RSEM) from the cBioPortal datahub, for the transcriptomic teacher."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
           "luad_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    print(f"downloaded {dest}")


def main() -> None:
    download_expression()
    build("luad_tcga_pan_can_atlas_2018", "luad")
    build("lusc_tcga_pan_can_atlas_2018", "lusc")  # coarse-vs-granular comparison + LUAD/LUSC subtyping


if __name__ == "__main__":
    main()
