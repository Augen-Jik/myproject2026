#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp

python generate_dataset_sparse.py --out-dir /root/autodl-tmp/dataset_sparse_v2
python run_sft_lora.py --config /root/autodl-tmp/sft_config_lora_sparse_v2_stage1.yaml
python run_sft_lora.py --config /root/autodl-tmp/sft_config_lora_sparse_v2_stage2.yaml
python run_sft_lora.py --config /root/autodl-tmp/sft_config_lora_sparse_v2_stage3.yaml
python merge_lora.py \
  --base /root/autodl-tmp/Qwen2.5-1.5B-Instruct \
  --lora /root/autodl-tmp/model_lora_sparse_v2 \
  --output /root/autodl-tmp/model_merged_sparse_v2
