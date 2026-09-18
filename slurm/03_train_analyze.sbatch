#!/bin/bash
#SBATCH --job-name=mosaic-mil
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" && source slurm/env.sh
python scripts/teacher_feasibility.py                     # omics-only check (seconds)
python scripts/train_mil.py --epochs ${EPOCHS:-20}        # mini-Paladin: 10 tasks x 2 CV schemes
python scripts/train_functional.py --epochs ${EPOCHS:-20} --seeds ${SEEDS:-3}   # DNA vs function supervision
python scripts/analyze.py
SLIDES_DIR=$SLIDES python scripts/plot_attention.py
