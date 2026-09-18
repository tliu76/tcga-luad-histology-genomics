# Interview Talking Points

## One-sentence pitch

I rebuilt the core Mosaic/Paladin analyses on public TCGA lung adenocarcinoma slides and genomics.
The logic I test is the paper's own: stratify by granular subtype, score VUS, and look for phenocopies.
I then propose one change: supervise the H&E model with **pathway function** measured by a
transcriptomic teacher, instead of **DNA genotype**.

## Story arc (about 5 minutes)

1. **What I took from the paper.**
   * Granular subtypes matter. Pooled models learn subtype, not genotype (Ext. Data Fig. 8).
   * The most novel result is Fig. 5. Morphology disagrees with sequencing in *informative* ways:
     functional VUS, and STK11 phenocopies with low mRNA and poor survival.
2. **Reproduction on public data.** TCGA-LUAD diagnostic slides, open access, with no token needed.
   * 112 µm tiles, as in the paper
   * Phikon features and gated ABMIL
   * patient-level CV, plus **site-grouped CV** (TCGA site batch effects)
3. **The gap.** Paladin uses DNA as ground truth and finds phenocopies post hoc. The paper itself
   lists epigenetic silencing, missed variants and functional VUS as limits of DNA labels
   (lines 86–91, 402–404).
4. **The proposal.** A transcriptomic teacher turns genotype into a continuous pathway-activity
   score, and the H&E student regresses it. Training is multimodal; inference is H&E only.
   This fits the lab's pathway-level view (Sanchez-Vega *et al.*, Cell 2018).
5. **Evidence so far** (omics only, n = 510, done):
   * The in-fold LKB1-loss program recovers STK11 status at AUROC 0.88.
     It rediscovers CPS1, DUSP4 and PDE4D without being told.
   * KEAP1 program: AUROC 0.93. ρ = 0.67 with the literature NRF2 target-gene score.
   * **73 KEAP1 missense tumours score like truncating ones** (p = 3e-30). That is 3× the truncating
     positives, and a DNA-label model throws them away or calls them negative.
     22 STK11 missense tumours behave the same way (p = 1.6e-10).
   * STK11-WT tumours with a high program score have lower LKB1 protein by RPPA (p = 0.016).
     These are phenocopies, with an orthogonal protein readout.
6. **Decisive experiment** (running on CARC): DNA-supervised vs function-supervised ABMIL,
   with identical folds and seeds. The judges are data neither model saw: LKB1 protein, OS, and
   held-out RNA.

## Likely questions and answers

* **"Isn't RNA leaking into the evaluation?"**
  The STK11 program is re-derived inside each training fold, and test-fold RNA is only a judge.
  The protein and survival judges are fully independent of training.
* **"Why not just sequence RNA?"**
  RNA is used only at training time, on a research cohort. Deployment is H&E only, the same as Paladin.
* **"Your n is tiny."**
  Yes. About 470 vs about 880 LUAD primaries in the paper. I report bootstrap CIs and a
  site-grouped CV. The claim is about *relative* performance of two supervision schemes on
  identical splits, which is less sensitive to n than absolute AUROC.
* **"Why site-grouped CV?"**
  TCGA tissue-source sites differ in stain and scanner, and in patient mix.
  Howard *et al.* (Nat Commun 2021) showed that this inflates genotype prediction.
* **"How would this scale at MSK?"**
  MSK-IMPACT has no RNA for most cases. The teacher could instead be:
  * MSK-IMPACT + RNA subsets
  * IHC (as in Fig. 5e)
  * a multi-gene pathway score
  Train the teacher where RNA exists, and apply the student everywhere.
* **Next steps.**
  * CPTAC-LUAD external validation, with public slides and proteomics
  * other pathways (RTK-RAS, PI3K, Hippo from the 2018 paper)
  * attention maps to *describe* the LKB1-loss morphology (open question in the Discussion)

## Responsible limitations statement

This is an exploratory public-data demonstration. It is not a diagnostic model, does not establish
causality, and needs independent external validation before any clinical interpretation.
