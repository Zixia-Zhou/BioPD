#!/usr/bin/env bash
set -euo pipefail

python run_biopd_cell.py \
  --h5ad ./data/cell/paul15.h5ad \
  --dataset paul15 \
  --seed 0 \
  --cuda_device 0 \
  --out_dir ./results/cell
