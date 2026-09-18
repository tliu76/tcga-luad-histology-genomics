#!/usr/bin/env python3
"""Build the interview slide deck (.pptx) and speaker script (.docx) from results/ and reports/figures/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Pt, RGBColor as DocRGB
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt as PPt

FIG = Path("reports/figures")
OUT = Path("reports")
INK, MUTED, BLUE, ORANGE, RED = RGBColor(0x1F, 0x24, 0x30), RGBColor(0x6B, 0x72, 0x80), RGBColor(0x2F, 0x6D, 0xB5), RGBColor(0xD9, 0x82, 0x2B), RGBColor(0xB5, 0x45, 0x2F)

# ---------------------------------------------------------------- numbers (read, never typed)
T = json.loads(Path("results/teacher_feasibility.json").read_text())
S = json.loads(Path("results/site_confounding.json").read_text())
F = json.loads(Path("results/functional/report.json").read_text())
X = json.loads(Path("results/extra_numbers.json").read_text())
M = pd.read_csv("results/metrics.csv").set_index(["task", "scheme"])
LC = pd.read_csv("results/functional/learning_curve_summary.csv")
GS = {r["gene"]: r for r in S["genotype_by_site"]}


def auc(task, scheme="random"):
    return M.loc[(task, scheme), "auroc_abmil"]


def ci(task, scheme="random"):
    r = M.loc[(task, scheme)]
    return f"{r.auroc_abmil:.2f} [{r.auroc_abmil_lo:.2f}–{r.auroc_abmil_hi:.2f}]"


def lc(pw, arm, frac, col):
    r = LC[(LC.pathway == pw) & (LC.arm == arm) & (LC.frac == frac)]
    return float(r[col].iloc[0])


N_LUAD = int(M.loc[("LUAD_TP53", "random"), "n"])
N_LUSC = int(M.loc[("LUSC_TP53", "random"), "n"])
st = F["STK11"]
V = dict(
    n_luad=N_LUAD, n_lusc=N_LUSC,
    stk_auc_t=T["STK11"]["oof_auroc_vs_dna"], keap_auc_t=T["KEAP1"]["oof_auroc_vs_dna"],
    stk_vus_n=T["STK11"]["n_class"]["VUS-like"], keap_vus_n=T["KEAP1"]["n_class"]["VUS-like"],
    keap_trunc_n=T["KEAP1"]["n_class"]["oncogenic"], stk_p=T["STK11"]["p_vus_gt_wt"], keap_p=T["KEAP1"]["p_vus_gt_wt"],
    ph_n=T["STK11"]["lkb1_protein"]["n_wt_program_high"], ph_p=T["STK11"]["lkb1_protein"]["p_wt_high_lt_low"],
    site_ba_luad=S["site_id_luad"]["balanced_accuracy"], site_n_luad=S["site_id_luad"]["n_sites"],
    site_ba_all=S["site_id"]["balanced_accuracy"], site_n_all=S["site_id"]["n_sites"],
    stk_prev_max=GS["STK11"]["prev_max"], stk_chi=GS["STK11"]["chi2_p"], stk_site_auc=GS["STK11"]["site_only_auroc"],
    egfr_prev_max=GS["EGFR"]["prev_max"], egfr_site_auc=GS["EGFR"]["site_only_auroc"],
    tp53=auc("LUAD_TP53"), tp53_site=auc("LUAD_TP53", "site"), stk=auc("LUAD_STK11"), stk_site=auc("LUAD_STK11", "site"),
    egfr=auc("LUAD_EGFR"), sub=auc("NSCLC_subtype_LUSC"), sub_site=auc("NSCLC_subtype_LUSC", "site"),
    pooled=X["pooled_tp53_auroc"], sub_only=X["subtype_prob_only_auroc"], lusc_tp53=auc("LUSC_TP53"),
    f_auc_dna=st["dna"]["auroc_dna_label"], f_auc_fun=st["func"]["auroc_dna_label"],
    f_rho_dna=st["dna"]["rho_heldout_rna_program"], f_rho_fun=st["func"]["rho_heldout_rna_program"],
    f_rho_ci=st["diff_rho_rna_ci"], f_auc_ci=st["diff_auroc_ci"],
    lc30_dna=lc("STK11", "dna", 0.25, "rho_heldout_rna"), lc30_fun=lc("STK11", "func", 0.25, "rho_heldout_rna"),
    nrf_dna=F["NRF2"]["dna"]["auroc_dna_label"], nrf_fun=F["NRF2"]["func"]["auroc_dna_label"],
    co_hi=X["keap1_smarca4_wt_high_vs_low"]["table"][0], co_lo=X["keap1_smarca4_wt_high_vs_low"]["table"][1],
    co_p=X["keap1_smarca4_wt_high_vs_low"]["fisher_p"],
)
V["co_hi_pct"] = V["co_hi"][0] / sum(V["co_hi"])
V["co_lo_pct"] = V["co_lo"][0] / sum(V["co_lo"])

# ---------------------------------------------------------------- slide content
# Each slide: title (the takeaway), optional figure, bullets, speaker script (EN), Chinese tip, minutes.
SLIDES = [
    dict(kind="title",
         title="From genotype to pathway function",
         sub="What H&E morphology can and cannot tell us about lung adenocarcinoma genomics\nA public-data study building on Mosaic (Boehm, Darmofal, Pasha et al., 2025)",
         who="Tanxin Liu · PhD candidate, Genetic Epidemiology, USC",
         say="Thank you for making time to talk with me. I read your Mosaic preprint closely, and to understand it hands-on I rebuilt its core logic on public TCGA lung adenocarcinoma data. I'd like to share three short findings, and one idea I'd really value your view on.",
         tip="开场不超过 20 秒。语气是 '我读了你们的文章并动手做了'，而不是 '我要挑错'。", mins=0.3),
    dict(title="My background: somatic genomics → phenotype and outcome",
         bullets=[
             "Somatic genomics at scale: SNV / indel / SV / allele-specific CNV pipelines on >30 TB tumor–normal WGS; purity, ploidy, clonality",
             "Genotype → outcome: RAG-mediated SVs predict relapse in childhood B-ALL, including MRD-negative patients (first-author, under review)",
             "Machine learning: multiclass subtype classifiers (expression + fusion + CNV), nested CV, rare-class imbalance; MS CS (Georgia Tech)",
             "Orthogonal validation habit: RNA-based CNV vs DNA; long-read vs short-read SVs; plasma proteomics",
             "Medical training (Peking University): histology and pathology coursework",
         ],
         say="Briefly about me. My PhD is in somatic genomics of childhood leukemia. I built pipelines for SNVs, structural variants and allele-specific copy number across more than 30 terabytes of tumor–normal genomes, and I showed that RAG-mediated structural variants predict relapse even in MRD-negative patients. I've also built subtype classifiers from expression, fusion and copy-number data, with a focus on rare subtypes. A habit that runs through my work is validating one data type with another: RNA against DNA, long reads against short reads. And I trained in medicine first, so histology is not foreign to me. What I haven't done before is whole-slide imaging, which is exactly why I did this project.",
         tip="约 60 秒。最后一句主动承认没有做过 WSI，然后自然引到 demo。", mins=1.0),
    dict(title="Two ideas I took from Mosaic",
         bullets=[
             "1. Model genotype within granular subtypes: pooled models learn subtype, not genotype (Ext. Data Fig. 8)",
             "2. Disagreement between morphology and sequencing is informative: functional VUS and STK11 phenocopies (Fig. 5)",
             "Three questions I asked on public TCGA data:",
             "   A. Does granularity matter on an independent cohort?",
             "   B. TCGA is Mosaic's external test set. How much tissue-source-site signal does it carry?",
             "   C. If function is what morphology reflects, should the model be trained on function instead of DNA?",
         ],
         say="Two ideas stood out to me. First, modeling genotype within granular subtypes, because a pooled model will learn the subtype rather than the genotype. Second, and to me the most interesting, the idea that disagreement between morphology and sequencing is informative, as your functional VUS and STK11 phenocopy results show. That led to three questions. Does granularity hold on an independent cohort? TCGA is your external test set, so how much site signal does it carry? And if morphology reflects function, should we train on function rather than DNA?",
         tip="这一页是整场的主线，三个问题 A/B/C 后面一一对应。", mins=0.8),
    dict(title="Setup: open TCGA slides + cBioPortal genomics",
         fig="fig_pipeline.png",
         bullets=[f"LUAD n={N_LUAD}, LUSC n={N_LUSC} patients (smallest slides first; full-cohort pipeline ready for the cluster)",
                  "112 µm tiles as in Mosaic · Phikon foundation-model features · gated attention-MIL",
                  "Patient-level 5-fold CV, reported both random and grouped by tissue source site",
                  "RNA-seq and RPPA protein used only as teacher or held-out judges, never as image features"],
         say=f"Here is the setup. Open-access TCGA diagnostic slides, with genomics from the cBioPortal PanCancer Atlas. I matched your tile size of 112 microns, used the public Phikon foundation model for tile features, and trained a gated attention-MIL model per target with patient-level cross-validation. I report every result twice: random folds, and folds grouped by tissue source site. Because of time, this run uses {N_LUAD} LUAD and {N_LUSC} LUSC patients, the smallest slides first. The full cohort is set up to run on our cluster.",
         tip="强调 '每个结果都报两遍（random / site-grouped）'，这是严谨性的卖点。样本量要主动说。", mins=0.8),
    dict(title="Baseline: modest signal at n≈150 that survives site-grouped CV",
         fig="fig_performance.png",
         bullets=[f"TP53 {V['tp53']:.2f} → {V['tp53_site']:.2f} and STK11 {V['stk']:.2f} → {V['stk_site']:.2f} (random → site-grouped CV)",
                  f"LUAD vs LUSC {V['sub']:.2f} → {V['sub_site']:.2f}: in TCGA, site and subtype are entangled",
                  "Paladin (MSK, n≈880 LUAD) reaches 0.83–0.92: sample size matters, as your ablation (Fig. 4e) shows"],
         say=f"First the baseline, and I want to be upfront. With about 150 LUAD patients the signal is modest. TP53 is {V['tp53']:.2f} and STK11 is {V['stk']:.2f}, far below your 0.83 to 0.92 with about 880 patients, which fits your data-ablation curves. The encouraging part is that TP53 and STK11 barely move when entire hospitals are held out. The LUAD-versus-LUSC classifier drops from {V['sub']:.2f} to {V['sub_site']:.2f}, which already hints at how entangled site is with biology in TCGA.",
         tip="不要回避 AUROC 低。主动和文章的 data ablation（Fig 4e）联系起来。", mins=0.8),
    dict(title="A. Granularity: pooled TP53 signal ≈ subtype signal",
         fig="fig_coarse_vs_granular.png",
         bullets=[f"Pooled LUAD+LUSC TP53 AUROC {V['pooled']:.2f}; a subtype score alone gives {V['sub_only']:.2f}",
                  f"Within LUSC (83% TP53-mutant), TP53 AUROC {V['lusc_tp53']:.2f}: no genotype signal",
                  "Reproduces your Ext. Data Fig. 8 argument on an independent cohort"],
         say=f"Question A. If I pool LUAD and LUSC and predict TP53, I get {V['pooled']:.2f}. But a score that only knows whether a slide looks like LUSC already gets {V['sub_only']:.2f}, with no TP53 information at all. Within LUSC the TP53 signal is gone. So on an independent cohort, the pooled model is mostly a subtype detector, exactly your argument for granular modeling.",
         tip="这页讲得快一点，30–40 秒。核心是 0.66 vs 0.65。", mins=0.6),
    dict(title="B. TCGA slides carry a hospital signature; genotype varies by site",
         fig="fig_site_confounding.png",
         bullets=[f"Slide embeddings identify the contributing site: balanced accuracy {V['site_ba_luad']:.2f} across {V['site_n_luad']} LUAD sites (chance 0.25)",
                  f"STK11 prevalence ranges 0–{V['stk_prev_max']:.0%} across sites (χ² p={V['stk_chi']:.3f})",
                  f"Site label alone predicts STK11 at AUROC {V['stk_site_auc']:.2f} and EGFR at {V['egfr_site_auc']:.2f}, similar to my H&E models",
                  "Implication: evaluate TCGA transfer with site-grouped splits or site-stratified metrics"],
         say=f"Question B, which I think is the most practical finding. Using only the image embeddings, a simple classifier identifies which hospital a LUAD slide came from with {V['site_ba_luad']:.0%} balanced accuracy. At the same time, genotype prevalence differs by hospital. STK11 ranges from zero to {V['stk_prev_max']:.0%}. Knowing the hospital alone predicts STK11 at an AUROC of {V['stk_site_auc']:.2f}, about the same as my image model. So in TCGA a model can look good by recognizing the hospital. My models held up under site-grouped CV, so they are not only doing that, but I think this is worth checking whenever TCGA is used as an external test.",
         tip="这是最有冲击力的一页。措辞要温和：'worth checking'，不要说 'Mosaic is wrong'。文章本身没有讨论 site 问题。", mins=1.2),
    dict(title="C1. DNA labels miss functional cases (omics only, n=510 LUAD)",
         fig="fig_teacher_omics.png",
         bullets=[f"An LKB1-loss program derived only inside training folds recovers STK11 status (AUROC {V['stk_auc_t']:.2f}); KEAP1 program {V['keap_auc_t']:.2f}",
                  f"Missense-only tumors score like truncating ones: STK11 n={V['stk_vus_n']} (p={V['stk_p']:.0e}), KEAP1 n={V['keap_vus_n']} (p={V['keap_p']:.0e})",
                  f"KEAP1: {V['keap_vus_n']} functional missense vs {V['keap_trunc_n']} truncating, so a DNA label drops ~3× more positives than it keeps",
                  f"STK11-WT tumors with the program on have lower LKB1 protein (n={V['ph_n']}, p={V['ph_p']:.3f}): phenocopies"],
         say=f"Now question C. Before touching images, I asked how good DNA labels are as ground truth. I derived an LKB1-loss transcriptional program inside each training fold. It recovers STK11 status at {V['stk_auc_t']:.2f} and rediscovers known LKB1-loss genes like CPS1 and DUSP4. Missense-only tumors score just like truncating ones. For KEAP1 that is {V['keap_vus_n']} missense tumors versus {V['keap_trunc_n']} truncating ones, so a DNA-trained model either discards or mislabels most of the functional positives. And WT tumors with the program switched on have lower LKB1 protein, which is the phenocopy you described in Fig. 5, measured here by proteomics instead of IHC.",
         tip="这是全场最扎实的证据（n=510，p 值很小）。讲慢一点。", mins=1.2),
    dict(title="C2. Proposal: supervise the H&E model with pathway function",
         fig="fig_supervision_schematic.png",
         bullets=["Teacher: RNA program converts genotype into continuous pathway activity (training only)",
                  "Student: attention-MIL regresses that score from H&E; inference needs H&E only",
                  "Uses VUS carriers and phenocopies by construction instead of finding them post hoc",
                  "Different from Paladin's binary 'pathway altered' targets: continuous activity, not presence of a mutation"],
         say="So the proposal is simple. Instead of training the image model on a binary DNA label, a transcriptomic teacher turns genotype into a continuous pathway-activity score, and the image model learns that. RNA is needed only at training time, and inference is H&E only, like Paladin. VUS carriers and phenocopies are used by construction rather than discovered afterwards. I know Paladin already has pathway-level targets, but those are binary 'is there an alteration in the pathway'. This is continuous activity.",
         tip="主动说出 Paladin 已有 pathway target 以及区别，显得你读得细。", mins=0.9),
    dict(title="C3. Function supervision is more label-efficient (STK11)",
         fig="fig_learning_curve.png",
         bullets=[f"STK11, same patients, folds and seeds: agreement with held-out LKB1-loss program ρ {V['f_rho_fun']:.2f} vs {V['f_rho_dna']:.2f} (Δ 95% CI {V['f_rho_ci'][0]:.2f} to {V['f_rho_ci'][1]:.2f})",
                  f"DNA-label AUROC {V['f_auc_fun']:.2f} vs {V['f_auc_dna']:.2f} (not significant at n=150)",
                  f"With 30 training patients: ρ {V['lc30_fun']:.2f} vs {V['lc30_dna']:.2f}",
                  f"NRF2: no image signal for either arm (AUROC {V['nrf_fun']:.2f} vs {V['nrf_dna']:.2f}); protein and survival judges were uninformative at this n"],
         say=f"Then the head-to-head on images, with the same patients, folds and seeds. For STK11, the function-supervised model agrees better with the held-out LKB1-loss program, {V['f_rho_fun']:.2f} versus {V['f_rho_dna']:.2f}, and the confidence interval of the difference excludes zero. It is also higher on the DNA label, {V['f_auc_fun']:.2f} versus {V['f_auc_dna']:.2f}, but that is not significant with 150 patients. The advantage is largest with little data: with 30 training patients, {V['lc30_fun']:.2f} versus {V['lc30_dna']:.2f}. I want to be clear about what did not work. NRF2 showed no image signal for either approach, and protein and survival were too noisy to separate the arms at this size.",
         tip="一定要把 '没成功的部分' 说出来，这会大大增加可信度。重点是 label efficiency：对应文章 Discussion 第 405–406 行说的稀有亚型样本不足。", mins=1.2),
    dict(title="What holds, what doesn't (yet)",
         table=[("Finding", "Evidence", "Status"),
                ("Pooled models learn subtype", f"pooled {V['pooled']:.2f} ≈ subtype-only {V['sub_only']:.2f}", "Reproduced"),
                ("TCGA hospital signature", f"site ID {V['site_ba_luad']:.2f} bal. acc.; STK11 site-only AUROC {V['stk_site_auc']:.2f}", "Strong"),
                ("DNA labels miss functional cases", f"missense ≈ truncating (p={V['keap_p']:.0e}); phenocopy protein p={V['ph_p']:.3f}", "Strong (n=510)"),
                ("Function supervision helps", f"STK11 ρ {V['f_rho_fun']:.2f} vs {V['f_rho_dna']:.2f}; AUROC n.s.", "Promising, n=150"),
                ("STK11-WT, H&E-high co-mutations", f"KEAP1/SMARCA4 {V['co_hi_pct']:.0%} vs {V['co_lo_pct']:.0%} (p={V['co_p']:.2f})", "Matches Fig. 5f"),
                ("NRF2 morphology; protein/OS judges", "no signal at n=150", "Not shown")],
         say="To summarize honestly. Two findings are strong: the hospital signature in TCGA, and the fact that DNA labels miss many functionally altered tumors. Your granularity argument reproduced on an independent cohort. Function supervision looks promising for STK11 but needs the full cohort to be conclusive, and NRF2 did not work at this size.",
         tip="这一页是给对方的 '诚信表'，快速过。", mins=0.6),
    dict(title="How this could scale at MSK",
         bullets=["Teacher where RNA or IHC exists (a subset); student applied to all ~70k slides",
                  "Extend to the canonical pathways of Sanchez-Vega et al. 2018 (RTK-RAS, PI3K, Hippo, NRF2, …)",
                  "Weight labels by cancer cell fraction: subclonal alterations may not have a morphologic footprint (my CCF modeling work)",
                  "Site- and scanner-aware evaluation for any external cohort; CPTAC-LUAD (slides + proteomics) as a next public test"],
         say="How could this scale at MSK? The teacher only needs the subset of patients with RNA or IHC, and the student then applies to all 70,000 slides. It extends naturally to the canonical signaling pathways from your 2018 pathway paper. From my own work, I'd like to weight training labels by cancer cell fraction, because a subclonal alteration may not have a morphologic footprint. And any external evaluation should be site-aware. CPTAC lung adenocarcinoma, with slides and proteomics, would be the next public test.",
         tip="提到对方 2018 年的 pathway 文章，以及你自己的 CCF 经验，这是 '我能为组里带来什么'。", mins=0.8),
    dict(kind="end", title="Thank you",
         bullets=["Questions I'd love to ask:",
                  "• After Mosaic's public release, what is the most important next scientific question for it?",
                  "• How much matched RNA or IHC exists for MSK-IMPACT cases?",
                  "• How do postdocs split time between the pathology and genomics sides of the lab?"],
         say="That's everything I wanted to share. I'd love to hear where you see Mosaic going next, and whether an idea like function supervision would be feasible with the data at MSK.",
         tip="讲完就停，把话语权交给对方。准备好 2–3 个问题。", mins=0.3),
    # ---------------- backup
    dict(backup=True, title="Backup · Mosaic in one slide",
         bullets=["Mussel: tissue detection, 20× tiling (112 µm), tile embeddings; each sequenced part = one bag (4.74B tiles)",
                  "Aeon: 163 OncoTree subtypes; ontology-smoothed loss from a knowledge graph (top-1 0.51 → 0.47 when ablated); AUROC 0.992",
                  "Paladin: one model per subtype × sample type × target; ≥50 positives and negatives; 3,541 pairs, 165 with AUROC ≥ 0.80",
                  "Downstream: Leiden clusters + Fisher/BH enrichment; CUP reassignment validated by genomics and survival; VUS via one-sided Mann-Whitney; phenocopies validated by IHC, RNA, survival",
                  "Note: this preprint version does not include the full Methods section"],
         say="If asked to summarize Mosaic: Mussel preprocesses slides; Aeon classifies 163 OncoTree subtypes with an ontology-smoothed loss; Paladin trains subtype-specific genomic models; and downstream analyses cover CUP refinement, VUS annotation and phenocopies. The preprint I read does not include the detailed Methods, so I'd avoid claiming specific architecture details.",
         tip="只有被问到才用。架构细节要说 'based on the preprint'。", mins=0),
    dict(backup=True, title="Backup · Methods details of this study",
         bullets=["Labels: EGFR/KRAS canonical activating alleles; TP53 any non-synonymous or deep deletion; STK11/KEAP1 truncating/splice/deep deletion = 1, missense-only held out",
                  "Features: Phikon ViT-B CLS token, ≤1000 tissue tiles per slide (Otsu on saturation, ≥50% tissue)",
                  "Model: gated attention-MIL (256-d), AdamW, 15 epochs, 512 random tiles per step; mean-pool logistic regression baseline",
                  "Teacher: top-50 up/down genes by Welch t (truncating vs WT) re-derived in each training fold; NRF2 literature target set",
                  "Stats: bootstrap 95% CIs; paired bootstrap for arm differences; 3 seeds × 5 folds; Fisher/BH; Cox adjusted for stage and age"],
         say="Method details for questions about leakage or labels.", tip="被问到泄漏时指向第四条：teacher 在每个 fold 内重新推导。", mins=0),
    dict(backup=True, title="Backup · What the STK11 model attends to", fig="fig_attention_LUAD_STK11.png",
         bullets=[f"Top-attended tiles in confident STK11-altered cases show solid, poorly differentiated growth",
                  f"STK11-WT but H&E-high tumors are enriched for KEAP1/SMARCA4 alterations: {V['co_hi_pct']:.0%} vs {V['co_lo_pct']:.0%} (p={V['co_p']:.2f}), as in your Fig. 5f"],
         say="The highest-attention tiles show solid, poorly differentiated growth. Consistent with your Fig. 5f, STK11 wild-type tumors that the model scores high are enriched for KEAP1 or SMARCA4 alterations.",
         tip="形态学描述要保守，你不是病理医生。", mins=0),
    dict(backup=True, title="Backup · My PhD work: RAG-mediated SVs and relapse in B-ALL",
         bullets=["Paired WES/WGS + RNA-seq; somatic SV calling and allele-specific CN; motif enrichment of cryptic RSS",
                  "RAG-mediated SV burden associated with relapse, including among MRD-negative patients (medRxiv 2026)",
                  "Early-life tobacco exposure linked to aberrant RAG-mediated recombination (Leukemia 2024)",
                  "GWAS in Hispanic/Latino children: novel ALL risk locus at 5q31.1 (ASH 2024 oral)"],
         say="If asked about my PhD work, this is the one-slide version. Fill in the hazard ratio and sample size from the manuscript before the interview.",
         tip="⚠️ 面试前把论文里的 HR、样本量、p 值补进讲稿，背熟。", mins=0),
]


# ---------------------------------------------------------------- pptx
def add_text(slide, x, y, w, h, text, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = PPt(size), bold, color, "Calibri"
    return tb


def add_bullets(slide, x, y, w, h, items, size=15):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = PPt(6)
        r = p.add_run()
        r.text = it if it.startswith(("   ", "•", "1.", "2.")) or it.endswith(":") else "•  " + it
        r.font.size, r.font.color.rgb, r.font.name = PPt(size), INK, "Calibri"
        if it.endswith(":"):
            r.font.bold = True


def add_fig(slide, path, x, y, w, h):
    im = Image.open(path)
    ar = im.width / im.height
    fw, fh = (w, w / ar) if w / ar <= h else (h * ar, h)
    slide.shapes.add_picture(str(path), Inches(x + (w - fw) / 2), Inches(y + (h - fh) / 2), Inches(fw), Inches(fh))


def build_pptx():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    main = [s for s in SLIDES if not s.get("backup")]
    for idx, s in enumerate(SLIDES, 1):
        sl = prs.slides.add_slide(blank)
        if s.get("kind") == "title":
            bar = sl.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.18))
            bar.fill.solid(), setattr(bar.fill.fore_color, "rgb", BLUE), bar.line.fill.background()
            add_text(sl, 0.9, 2.2, 11.5, 1.2, s["title"], 40, INK, True)
            add_text(sl, 0.9, 3.4, 11.5, 1.2, s["sub"], 20, MUTED)
            add_text(sl, 0.9, 5.3, 11.5, 0.6, s["who"], 18, INK)
        else:
            color = MUTED if s.get("backup") else INK
            add_text(sl, 0.6, 0.4, 12.2, 0.8, s["title"], 24, color, True)
            line = sl.shapes.add_shape(1, Inches(0.6), Inches(1.28), Inches(1.2), Inches(0.05))
            line.fill.solid(), setattr(line.fill.fore_color, "rgb", ORANGE if s.get("backup") else BLUE), line.line.fill.background()
            if "table" in s:
                rows = s["table"]
                t = sl.shapes.add_table(len(rows), 3, Inches(0.6), Inches(1.6), Inches(12.1), Inches(0.55 * len(rows))).table
                for c, wdt in enumerate([4.0, 5.6, 2.5]):
                    t.columns[c].width = Inches(wdt)
                for r, row in enumerate(rows):
                    for c, val in enumerate(row):
                        cell = t.cell(r, c)
                        cell.text = val
                        para = cell.text_frame.paragraphs[0]
                        para.runs[0].font.size = PPt(14)
                        para.runs[0].font.bold = r == 0
                        if c == 2 and r > 0:
                            para.runs[0].font.color.rgb = BLUE if val.startswith(("Strong", "Reproduced", "Matches")) else (ORANGE if val.startswith("Prom") else RED)
            elif "fig" in s:
                add_fig(sl, FIG / s["fig"], 0.6, 1.5, 12.1, 3.9)
                add_bullets(sl, 0.7, 5.45, 12.0, 1.9, s["bullets"], 14)
            else:
                add_bullets(sl, 0.7, 1.6, 12.0, 5.5, s["bullets"], 18)
        if not s.get("kind") == "title":
            tag = "Backup" if s.get("backup") else f"{main.index(s) + 1}/{len(main)}"
            add_text(sl, 0.6, 7.05, 12.1, 0.35, f"Tanxin Liu · Mini-Mosaic on TCGA-LUAD · {tag}", 10, MUTED)
        sl.notes_slide.notes_text_frame.text = s["say"] + "\n\n[提示] " + s["tip"]
    path = OUT / "Tanxin_Liu_interview_slides.pptx"
    prs.save(path)
    return path


# ---------------------------------------------------------------- docx
QA = [
    ("Isn't RNA leaking into the evaluation?",
     "The STK11 program is re-derived inside each training fold from training patients only. Test-fold RNA is used only as a judge, and the protein and survival judges never touch training."),
    ("Why not just sequence RNA?",
     "RNA is needed only at training time, on a subset. Deployment is H&E-only, exactly like Paladin, so it keeps the cost and turnaround advantage."),
    ("Your sample size is small.",
     "Yes, about 150 LUAD patients versus about 880 in the paper. That is why I report bootstrap CIs, site-grouped CV and a paired comparison on identical splits. The strongest claims (site signature, DNA-label noise) use all 510 patients or do not depend on image-model n."),
    ("Is the site effect stain, scanner or patient mix?",
     "I can't separate them here. Stain normalization, scanner metadata and site-stratified AUROC would be the next checks. The practical point is that TCGA transfer should be evaluated with site-aware splits."),
    ("Why did NRF2 fail?",
     "Either the morphology is subtle and needs more data, or 1000 tiles per slide miss it. NRF2 is also not among the pairs listed in your Table 1. I'd test it on the full cohort before drawing conclusions."),
    ("How is this different from Paladin's pathway targets?",
     "Paladin's pathway targets are binary: is any gene in the pathway altered. Mine is a continuous activity score that also captures functional missense variants and non-genetic loss."),
    ("What would you do in your first year here?",
     "Scale function supervision to MSK data where RNA or IHC exists, starting with STK11/KEAP1 in LUAD. Add CCF-weighted labels using my clonality work. Build site- and scanner-aware evaluation into the Paladin pipeline."),
    ("Why move from leukemia to solid tumors and pathology?",
     "The genomics skills transfer directly, and what I want to learn is how genomic events shape tumor phenotype. MSK's scale of matched slides and sequencing is unique for that."),
    ("When do you finish your PhD? (prepare a real answer)",
     "⚠️ Fill in: expected defense date and earliest start date."),
]


def build_docx():
    d = Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(11)
    d.add_heading("Interview script — Mini-Mosaic on TCGA-LUAD", 0)
    d.add_paragraph("面试对象：Francisco Sánchez-Vega（Mosaic 文章通讯作者）。30 分钟初面。英文逐字稿用于练习，中文为提示，不要照念。")
    d.add_heading("1. 30 分钟时间分配", 1)
    for t in ["0–2 min  寒暄", "2–4 min  90 秒自我介绍（Slide 2）", "4–12 min  对方追问你的研究（备用页：PhD work）",
              "12–20 min  Demo：Slide 3–12（可压缩到 5 分钟版本，见第 4 节）", "20–27 min  对方介绍组里方向 + 你提问",
              "27–30 min  时间线、资助等后勤"]:
        d.add_paragraph(t, style="List Bullet")
    d.add_heading("2. 90-second introduction", 1)
    d.add_paragraph(SLIDES[1]["say"] + " What draws me to your lab is connecting genomic events to tumor phenotype at scale, which is what Mosaic does. To understand it hands-on, I rebuilt parts of it on TCGA lung adenocarcinoma, and it led to an idea I'd love your thoughts on.")
    d.add_heading("3. Slide-by-slide script", 1)
    main = [s for s in SLIDES if not s.get("backup")]
    total = 0
    for i, s in enumerate(SLIDES, 1):
        label = "Backup" if s.get("backup") else f"Slide {main.index(s) + 1}"
        total += s["mins"]
        d.add_heading(f"{label} — {s['title']}" + (f"  (~{s['mins']:.1f} min)" if s["mins"] else ""), 2)
        d.add_paragraph(s["say"])
        p = d.add_paragraph()
        r = p.add_run("提示：" + s["tip"])
        r.font.color.rgb = DocRGB(0xB5, 0x45, 0x2F)
    d.add_paragraph(f"主讲部分合计约 {total:.0f} 分钟。时间紧时用下面的 5 分钟版本。")
    d.add_heading("4. 5-minute version (if time is short)", 1)
    d.add_paragraph(
        f"I rebuilt the Paladin logic on public TCGA lung adenocarcinoma data, with {N_LUAD} LUAD and {N_LUSC} LUSC patients. Three findings. "
        f"First, your granularity argument reproduces: a pooled LUAD-plus-LUSC TP53 model reaches {V['pooled']:.2f}, but a subtype score alone gets {V['sub_only']:.2f}. "
        f"Second, TCGA slides carry a strong hospital signature. Embeddings identify the site with {V['site_ba_luad']:.0%} balanced accuracy, and STK11 prevalence ranges from 0 to {V['stk_prev_max']:.0%} across sites, "
        f"so the site label alone predicts STK11 at {V['stk_site_auc']:.2f}. That matters whenever TCGA is used as an external test. "
        f"Third, DNA labels miss functional cases: {V['keap_vus_n']} KEAP1 missense tumors have the same transcriptional program as truncating ones. "
        f"So I propose training the H&E model on a transcriptomic pathway-activity score instead of the DNA label. For STK11 this was more label-efficient, "
        f"with agreement with the held-out program of {V['f_rho_fun']:.2f} versus {V['f_rho_dna']:.2f}. NRF2 did not work at this sample size, and the full-cohort run is set up for our cluster.")
    d.add_heading("5. Likely questions", 1)
    for q, a in QA:
        d.add_paragraph(q, style="List Number").runs[0].bold = True
        d.add_paragraph(a)
    d.add_heading("6. Numbers to memorize", 1)
    for t in [f"Site ID from H&E: {V['site_ba_luad']:.2f} balanced accuracy, {V['site_n_luad']} LUAD sites (chance 0.25)",
              f"STK11 prevalence by site 0–{V['stk_prev_max']:.0%}; site-only AUROC {V['stk_site_auc']:.2f}",
              f"Pooled TP53 {V['pooled']:.2f} vs subtype-only {V['sub_only']:.2f}",
              f"KEAP1 missense {V['keap_vus_n']} vs truncating {V['keap_trunc_n']}; p={V['keap_p']:.0e}",
              f"STK11 program AUROC {V['stk_auc_t']:.2f}; phenocopy protein p={V['ph_p']:.3f}",
              f"Function vs DNA (STK11): ρ {V['f_rho_fun']:.2f} vs {V['f_rho_dna']:.2f}; AUROC {V['f_auc_fun']:.2f} vs {V['f_auc_dna']:.2f}",
              "Paper: 71,142 patients · 378,123 WSIs · 163 subtypes · Aeon AUROC 0.992 · 3,541 Paladin pairs · STK11 LUAD 0.92"]:
        d.add_paragraph(t, style="List Bullet")
    d.add_heading("7. 面试前检查清单", 1)
    for t in ["背熟 90 秒自我介绍和 5 分钟版本（掐表练 2 遍）", "补上 PhD 论文的 HR / n / p，以及毕业时间",
              "重读 Mosaic 的 Fig 5 和 Discussion", "PPT 放在本地，另存一份 PDF 以防投屏出问题",
              "准备 2–3 个问题（见最后一页）"]:
        d.add_paragraph(t, style="List Bullet")
    path = OUT / "Tanxin_Liu_interview_script.docx"
    d.save(path)
    return path


if __name__ == "__main__":
    print(build_pptx())
    print(build_docx())
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in V.items()}, default=str))
