#!/bin/bash
# 训练完成后执行：合并 LoRA 权重 + 重新运行对比实验
set -e
echo "=== Step 1: 合并 Sparse-LoRA v3 权重 ==="
python merge_lora.py \
    --base   /root/autodl-tmp/Qwen2.5-1.5B-Instruct \
    --lora   /root/autodl-tmp/model_lora_sparse \
    --output /root/autodl-tmp/model_merged_sparse \
    --dtype  bfloat16

echo ""
echo "=== Step 2: 重新运行对比实验（非LLM方法，快速验证） ==="
python compare/run_compare.py

echo ""
echo "=== 完成！启动 Streamlit 查看结果 ==="
echo "streamlit run code/app_fixed.py --server.port 6006"
