# Mini-Mosaic: H&E morphology ↔ pathway function in TCGA lung adenocarcinoma

A public-data study built on Boehm, Darmofal, Pasha *et al.*,
[*Integrated histopathologic modeling of detailed tumor subtypes and actionable biomarkers*](https://doi.org/10.1101/2025.08.14.670351)
(bioRxiv 2025). It uses TCGA diagnostic H&E slides and cBioPortal genomics in place of MSK-IMPACT.
It (1) reproduces the paper's key analyses at small scale and (2) tests a proposed change to how
the genomic-inference models are supervised.

> Exploratory research demo on public data. 

## The proposal: supervise morphology with pathway function, not DNA genotype

Paladin learns **H&E → DNA alteration**. That label is noisy in ways the paper itself documents:
epigenetic silencing and missed variants are labelled wild-type, and functional VUS are
ambiguous. Phenocopies are found *after* training (paper Fig. 5).

Here a **transcriptomic teacher** converts genotype into a continuous **pathway-activity score**:

* **LKB1-loss program**: top-50 up and top-50 down genes, truncating vs WT, derived *inside each training fold*
* **NRF2 target-gene program**: NQO1, AKR1C1-3, GCLM, TXNRD1, SRXN1, …; fixed a priori

An **H&E student** (gated ABMIL) then regresses that score. Training is multimodal; inference
needs only H&E. The student can learn from VUS carriers, and phenocopies are labelled correctly
by construction.

**Head-to-head** (`scripts/train_functional.py`): DNA-supervised vs function-supervised ABMIL,
with identical patients, folds and seeds. They are judged on data that neither model trained on:

* DNA AUROC
* held-out RNA program
* **LKB1 protein (RPPA)**
* stage/age-adjusted overall survival
* VUS scoring

### Verified: omics-only feasibility (`scripts/teacher_feasibility.py`, n = 510 LUAD)

| | STK11 / LKB1 | KEAP1 / NRF2 |
|---|---|---|
| In-fold program recovers DNA status (OOF AUROC) | **0.88** | **0.93** |
| Top program genes (unsupervised recovery of known biology) | INHA, ODC1, PDE4D, CPS1, DUSP4 | TRIM16L, PGD, CBR1, TALDO1, SRXN1, G6PD, AKR1C1 |
| Median program score: WT / missense-only / truncating | −0.56 / **1.80** / 1.54 | −0.59 / **1.65** / 1.72 |
| Missense-only > WT (Mann-Whitney) | p = 1.6e-10 (n = 22) | p = 3e-30 (n = 73) |
| WT but program-high → lower LKB1 protein | n = 46, p = 0.016 | – |

So DNA-based labels discard or mislabel 22 STK11 and 73 KEAP1 functionally altered tumours
(the KEAP1 cases are 3× the truncating positives), plus WT phenocopies. These are the cases a
function-supervised H&E model can use. Whether morphology tracks function better than genotype
is what the slide-level experiment decides.

## Results (local: LUAD n = 153, LUSC n = 128; smallest slides first)

| Question | Result | Figure |
|---|---|---|
| Baseline H&E → genotype (LUAD) | TP53 0.69, STK11 0.65, EGFR 0.61 AUROC. TP53 and STK11 hold under site-grouped CV (0.67 / 0.65). Paladin reports 0.83–0.92 at n ≈ 880. | `fig_performance.png` |
| Granular vs coarse | Pooled LUAD+LUSC TP53 AUROC 0.66, vs 0.65 from a subtype score alone. Within LUSC, TP53 AUROC is 0.45. | `fig_coarse_vs_granular.png` |
| TCGA site signal | H&E embeddings identify the LUAD tissue source site with 0.99 balanced accuracy (4 sites, chance 0.25). STK11 prevalence ranges 0–36% by site (χ² p = 0.002), and site alone predicts STK11 at AUROC 0.64. | `fig_site_confounding.png` |
| DNA labels vs function (omics, n = 510) | See the feasibility table above. | `fig_teacher_omics.png` |
| DNA- vs function-supervised H&E (STK11) | Agreement with the held-out LKB1-loss program: ρ 0.47 vs 0.31 (Δ 95% CI 0.02–0.30). DNA-label AUROC 0.72 vs 0.65 (n.s.). Largest gain at 30 training patients (ρ 0.25 vs 0.09). | `fig_learning_curve.png` |
| NRF2 pathway | No image signal for either arm at this n. | – |
| STK11-WT, H&E-high tumours | KEAP1/SMARCA4-altered 25% vs 12% (Fisher p = 0.049), consistent with paper Fig. 5f. | `fig_attention_LUAD_STK11.png` |

**  Not significant at this sample size:**

* LKB1 protein (RPPA) and overall survival as judges of the H&E models
* Leiden-cluster mutation enrichment after FDR correction


## Background from the paper

1. **Inference (Paladin, Table 1).** Oncogenic `EGFR`, `KRAS`, `TP53`, `STK11`, `KEAP1` and NRF2-pathway
   alteration within LUAD, with gated ABMIL and a mean-pool logistic-regression baseline.
2. **Granular vs coarse (Ext. Data Fig. 8).** TP53 inference in pooled LUAD+LUSC vs within each subtype.
   The pooled model's score is compared with a LUAD-vs-LUSC subtype model to show it learns subtype.
3. **VUS and phenocopies (Fig. 5).** Missense STK11/KEAP1 are held out and scored. STK11-WT, H&E-high
   cases are checked against STK11 mRNA and LKB1 protein, and against overall survival.
4. **Unsupervised structure (Fig. 4a).** Leiden clusters of slide embeddings with Fisher/BH mutation enrichment.
5. **Attention maps (Fig. 4f).**

## Pipeline

| Step | Script | Notes |
|---|---|---|
| Labels + RNA | `scripts/build_genomic_labels.py` | cBioPortal PanCancer Atlas: mutations, GISTIC, RNA-seq, RPPA, clinical |
| Slide manifest | `scripts/query_gdc_metadata.py` | Open-access diagnostic FFPE slides, one per patient (no GDC token needed) |
| Download | `scripts/download_slides.py` | GDC open-data mirror on AWS (`s3://tcga-2-open`), resumable |
| Tiles + features | `scripts/extract_features.py` | 224 px @ 0.5 µm/px = **112 µm tiles** (as in the paper); Otsu tissue mask; [Phikon](https://huggingface.co/owkin/phikon) ViT-B |
| Teacher check | `scripts/teacher_feasibility.py` | Omics only, seconds |
| Mini-Paladin | `scripts/train_mil.py` | 10 tasks; 5-fold patient-level CV, **random and site-grouped** |
| Proposal | `scripts/train_functional.py` | DNA vs function supervision, 3 seeds |
| Analyses | `scripts/analyze.py`, `scripts/plot_attention.py` | Figures → `reports/figures/`, numbers → `results/summary.json` |

### Label definitions (OncoKB-free proxy)

* `EGFR`: L858R, exon 19 in-frame deletions, exon 20 insertions, G719X, L861Q, S768I, T790M
* `KRAS`: codons 12/13/59/61/117/146
* `TP53`: any non-synonymous mutation or deep deletion
* `STK11`, `KEAP1`: truncating/splice or deep deletion = 1; **missense-only = uncertain (held out)**
* `NRF2_pathway`: any KEAP1 alteration, NFE2L2 Neh2 hotspots or amplification, CUL3 truncation/deep deletion
  (Sanchez-Vega *et al.*, Cell 2018)

## Running on CARC

Everything large lives on `/scratch1/$USER/mosaic`. Total: about 720 GB of slides and about 6 GB of features.

**1. One-time setup.**

```bash
cd /project/<your_pi>_<id>/$USER            # or /scratch1/$USER; avoid /home1 (100 GB quota)
git clone <this repo> && cd tcga-luad-histology-genomics
module load conda
conda create -n mosaic python=3.11 -y && conda activate mosaic
pip install -r requirements.txt
```


**2. Cache the Phikon weights** on the login node, since compute nodes may lack internet.

```bash
source slurm/env.sh
python -c "from transformers import ViTModel; ViTModel.from_pretrained('owkin/phikon')"
```

**3. Quick test (optional), before the full run:**

```bash
python scripts/build_genomic_labels.py && python scripts/query_gdc_metadata.py
python scripts/teacher_feasibility.py       # reproduces the table above, no slides needed
```

**4. Submit the full pipeline.**

```bash
DL=$(sbatch --parsable slurm/01_download.sbatch)
FX=$(sbatch --parsable --dependency=after:$DL slurm/02_extract.sbatch)
sbatch --dependency=afterok:$FX slurm/03_train_analyze.sbatch
squeue -u $USER; tail -f logs/mosaic-dl-*.out
```

**Knobs** (environment variables at submit time):

* `MAX_LUSC` (default: all)
* `MAX_TILES` (default 4000 per slide)
* `EPOCHS` (default 20)
* `SEEDS` (default 3)

For a faster first pass, use `MAX_LUSC=200` and `MAX_TILES=1000`.

The downloader is resumable. If a job times out, resubmit it and completed slides are skipped.
For very large transfers, CARC recommends the data-transfer nodes (`hpc-transfer1.usc.edu`).
You can also run `scripts/download_slides.py --out $SLIDES` there directly.

**Outputs:**

* `results/metrics.csv`, `results/summary.json`, `results/functional/report.json`
* figures in `reports/figures/`

## Run locally 

```bash
pip install -r requirements.txt
python scripts/build_genomic_labels.py && python scripts/query_gdc_metadata.py
python scripts/download_slides.py --max-lusc 50 &   # Ctrl-C any time; smallest slides come first
python scripts/extract_features.py --watch --max-tiles 1000
python scripts/train_mil.py && python scripts/train_functional.py && python scripts/analyze.py
```

## Design choices

* **Patient-level splits only.** Tiles from one patient never cross folds.
* **Site-grouped CV.** TCGA tissue-source sites carry stain and scanner signatures that correlate with
  genotype prevalence. Reporting both CV schemes shows how much performance survives on unseen hospitals.
* **Missense variants are not called negative.** Held out, then scored.
* **No leakage from the teacher.** The LKB1-loss program is re-derived inside each training fold.
  Test-fold RNA is used only for evaluation.
* **Orthogonal judges.** Protein and survival are never used for training.

