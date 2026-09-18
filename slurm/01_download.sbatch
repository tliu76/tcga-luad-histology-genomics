#!/bin/bash
#SBATCH --job-name=mosaic-dl
#SBATCH --account=CHANGE_ME
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Labels + manifest + all LUAD and LUSC diagnostic slides (~720 GB) to $SLIDES.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" && source slurm/env.sh
python scripts/build_genomic_labels.py
python scripts/query_gdc_metadata.py
python scripts/download_slides.py --out "$SLIDES" --max-lusc ${MAX_LUSC:-478} --workers 16
