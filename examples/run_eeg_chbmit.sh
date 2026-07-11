#!/usr/bin/env bash
set -euo pipefail

python run_biopd_eeg.py \
  --data_root ./data_chb \
  --output_dir ./results/eeg \
  --gpu 0 \
  --seed 42
