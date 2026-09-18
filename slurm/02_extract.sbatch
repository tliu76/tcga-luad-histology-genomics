#!/bin/bash
#SBATCH --job-name=mosaic-feat
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
# 4 GPU shards (split by patient ID). --watch keeps polling while 01_download is still running,
# and exits after 30 min without new slides.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" && source slurm/env.sh
python scripts/extract_features.py --slides "$SLIDES" --max-tiles ${MAX_TILES:-4000} \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" --watch --idle-exit 1800
