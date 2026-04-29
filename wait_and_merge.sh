#!/bin/bash
# 等待训练完成后自动合并
echo "[$(date)] 等待训练进程完成..."
while pgrep -f "run_sft_lora" > /dev/null; do
    sleep 30
done
echo "[$(date)] 训练完成！开始合并..."
cd /root/autodl-tmp
python merge_lora.py \
    --base   /root/autodl-tmp/Qwen2.5-1.5B-Instruct \
    --lora   /root/autodl-tmp/model_lora_sparse \
    --output /root/autodl-tmp/model_merged_sparse \
    --dtype  bfloat16
echo "[$(date)] 合并完成！"
