#!/usr/bin/env bash
set -euo pipefail

python run_biopd_fmri.py \
  --analysis full \
  --data_dir ./data/fmri \
  --labels_path ./data/fmri/sherlock_labels_coded_expanded.csv \
  --out_dir ./results/fmri \
  --gpu 0 \
  --seed 0
