# TCGA-LUAD Histology-Genomics Demo

A compact, reproducible computational-pathology project that tests whether features learned from H&E whole-slide images improve patient-level prediction of genomic phenotypes in lung adenocarcinoma (TCGA-LUAD).

## Why this project

The first milestone is deliberately small: predict `EGFR` mutation status from diagnostic H&E slides. The project then provides a clean path to test `TP53`, focal copy-number events, molecular subtypes, and clinical outcomes. It is designed as a portfolio project, not a clinical diagnostic model.

## Research question

> Can pretrained histology representations from H&E whole-slide images predict genomic phenotypes in TCGA-LUAD, and do they add value beyond routine clinical variables?

## Project structure

```text
configs/       Analysis settings and cohort definition
data/          Local data only; never commit slides or patient-level data
docs/          Study notes and a data dictionary
scripts/       Command-line entry points
src/           Reusable Python code
tests/         Small unit tests
```

## Milestone 1: a one-week proof of concept

1. Query the GDC for public TCGA-LUAD diagnostic H&E slide metadata and clinical/mutation labels.
2. Download a small development cohort (for example, 80-150 patients) of slides.
3. Extract tissue patches and obtain pretrained pathology embeddings.
4. Fit a patient-level multiple-instance-learning (MIL) model for `EGFR` mutation status.
5. Report patient-level AUROC, PR-AUC, calibration, and representative high-attention patches.

Do not interpret an association as causal. Use patient-level splits, never tile-level splits, to prevent information leakage.

## Quick start

```bash
conda env create -f environment.yml
conda activate tcga-pathology
python scripts/query_gdc_metadata.py --config configs/luad_egfr.yaml
pytest -q
```

The metadata query is safe to run without downloading any slides. Review the output manifest before requesting slide files from the GDC portal/API.

## Data sources

- H&E diagnostic whole-slide images, clinical data, and mutation calls: [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/)
- Workflow sanity-check dataset: [CAMELYON16](https://camelyon16.grand-challenge.org/Download/)

## Portfolio deliverables

- A public README with question, cohort definition, and limitations.
- Versioned configuration files and an environment specification.
- A cohort manifest with GDC IDs, but no raw image files.
- A concise results notebook or report after analysis.

## Next extensions

- Predict `TP53` mutation or a copy-number phenotype.
- Integrate mutation/CNV features with H&E embeddings in a late-fusion model.
- Replace slide-level prediction with an interpretable question: which histologic patterns are associated with EGFR-mutant tumors?
- Evaluate external generalization only with a clearly independent cohort.
