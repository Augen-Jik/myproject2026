#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp

python validate_sparse_v2_strict.py \
  --model-path /root/autodl-tmp/model_merged_sparse_v2 \
  --output-json /root/autodl-tmp/results/sparse_v2_validation_strict_pre.json \
  --output-summary-csv /root/autodl-tmp/results/sparse_v2_validation_strict_pre_summary.csv \
  --output-detail-csv /root/autodl-tmp/results/sparse_v2_validation_strict_pre_details.csv

python generate_stage4_sparse_v2.py \
  --model-path /root/autodl-tmp/model_merged_sparse_v2 \
  --out-dir /root/autodl-tmp/dataset_sparse_v2_stage4

python run_sft_lora.py --config /root/autodl-tmp/sft_config_lora_sparse_v2_stage4.yaml

python merge_lora.py \
  --base /root/autodl-tmp/Qwen2.5-1.5B-Instruct \
  --lora /root/autodl-tmp/model_lora_sparse_v2_stage4 \
  --output /root/autodl-tmp/model_merged_sparse_v2_stage4

python validate_sparse_v2_strict.py \
  --model-path /root/autodl-tmp/model_merged_sparse_v2_stage4 \
  --output-json /root/autodl-tmp/results/sparse_v2_validation_strict_stage4.json \
  --output-summary-csv /root/autodl-tmp/results/sparse_v2_validation_strict_stage4_summary.csv \
  --output-detail-csv /root/autodl-tmp/results/sparse_v2_validation_strict_stage4_details.csv
