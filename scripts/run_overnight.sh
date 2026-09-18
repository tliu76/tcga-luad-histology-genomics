#!/bin/bash
# Local end-to-end run: top up LUAD slides, extract features while downloading, then train + analyse.
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
mkdir -p logs
step() { echo "=== $(date '+%H:%M:%S') $*"; }

step "download (LUAD up to ${MAX_LUAD:-350}, no new LUSC)"
if [ "${SKIP_DOWNLOAD:-0}" = 1 ]; then true; else $PY scripts/download_slides.py --max-luad ${MAX_LUAD:-350} --max-lusc 0 > logs/download2.log 2>&1; fi &
DL=$!
step "extract while downloading"
while kill -0 $DL 2>/dev/null; do
  $PY scripts/extract_features.py --max-tiles 1000 >> logs/extract2.log 2>&1
  sleep 30
done
$PY scripts/extract_features.py --max-tiles 1000 >> logs/extract2.log 2>&1
step "features: $(ls data/processed/features/*.h5 | wc -l)"

step "teacher feasibility";   $PY scripts/teacher_feasibility.py > logs/teacher.log 2>&1
step "train_mil";             $PY scripts/train_mil.py --epochs 15 > logs/train_mil.log 2>&1
step "train_functional";      $PY scripts/train_functional.py --epochs 15 --seeds 3 > logs/train_functional.log 2>&1
step "site confounding";      $PY scripts/site_confounding.py > logs/site.log 2>&1
step "analyze";               $PY scripts/analyze.py > logs/analyze.log 2>&1
step "attention";             $PY scripts/plot_attention.py > logs/attention.log 2>&1
step "done"
