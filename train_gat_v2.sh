#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp

python train_gat_v2.py \
  --model-path /root/autodl-tmp/model_merged_sparse_v2_stage4_fix \
  --dataset-root /root/autodl-tmp/dataset_sparse_v2 \
  --train-limit 1200 \
  --eval-limit 240 \
  --epochs 24 \
  --batch-size 48 \
  --always-output /root/autodl-tmp/gat_model_v2_always.pt \
  --gated-output /root/autodl-tmp/gat_model_v2_gated.pt
