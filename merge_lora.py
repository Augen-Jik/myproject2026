#!/usr/bin/env python3
"""
merge_lora.py — 将 LoRA 适配器权重合并到基座模型
合并后的完整模型可直接用于推理，无需 peft 依赖。

内存策略：
  - 默认在 CPU 上合并，完全规避 GPU OOM 问题
  - 若 CPU RAM 也不足，自动启用 low_cpu_mem_usage + torch.float16 降精度
  - 合并完成后立刻释放 LoRA 对象，再保存，减少峰值内存占用

用法:
  python merge_lora.py \
      --base   /root/autodl-tmp/Qwen2.5-1.5B-Instruct \
      --lora   /root/autodl-tmp/model_lora_qwen \
      --output /root/autodl-tmp/model_merged_qwen
"""

import argparse, gc, shutil, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--base",   required=True, help="基座模型路径")
parser.add_argument("--lora",   required=True, help="LoRA 适配器路径")
parser.add_argument("--output", required=True, help="合并后模型的保存路径")
parser.add_argument("--dtype",  default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="保存精度（默认 bfloat16；RAM 不足时自动降为 float16）")
args = parser.parse_args()

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}
save_dtype = dtype_map[args.dtype]

out_path = Path(args.output)
out_path.mkdir(parents=True, exist_ok=True)


def load_and_merge(dtype) -> None:
    """加载基座 + LoRA，合并后保存。全程 CPU，规避 GPU OOM。"""
    print(f"\n📦 加载基座模型（CPU，dtype={dtype}）: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="cpu",           # ★ 固定在 CPU 合并，保证 GPU 显存零占用
        trust_remote_code=True,
        low_cpu_mem_usage=True,     # 逐层加载，降低峰值 RAM
    )

    print(f"🔗 加载 LoRA 适配器: {args.lora}")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora,
        torch_dtype=dtype,
    )

    print("🔀 合并权重（merge_and_unload）...")
    model = model.merge_and_unload()

    # 立刻释放 PeftModel 包装，减少后续保存阶段的内存峰值
    del base_model
    gc.collect()

    print(f"💾 保存合并模型到: {args.output}")
    model.save_pretrained(args.output, safe_serialization=True)
    del model
    gc.collect()

    print("💾 保存 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)


# ── 带 OOM 保护的两级降级策略 ───────────────────────────────
try:
    load_and_merge(save_dtype)

except (RuntimeError, MemoryError) as e:
    err_str = str(e).lower()
    if "memory" in err_str or "oom" in err_str or "out of memory" in err_str:
        # 第一级降级：改用 float16 节省约 1/4 内存
        if save_dtype != torch.float16:
            print(f"\n⚠️  内存不足（{e}），自动降级到 float16 重试...")
            try:
                # 清理可能残留的对象
                gc.collect()
                load_and_merge(torch.float16)
            except (RuntimeError, MemoryError) as e2:
                print(f"\n❌ float16 仍然 OOM（{e2}）")
                print("建议：")
                print("  1. 关闭其他占用内存的进程后重试")
                print("  2. 在更大内存的机器（建议 ≥16GB RAM）上执行此脚本")
                print("  3. 或跳过合并，直接在推理时通过 peft.PeftModel 加载 LoRA")
                # 清理不完整的输出目录
                if out_path.exists():
                    shutil.rmtree(out_path)
                sys.exit(1)
        else:
            print(f"\n❌ float16 模式下仍然 OOM: {e}")
            if out_path.exists():
                shutil.rmtree(out_path)
            sys.exit(1)
    else:
        # 非内存问题，直接抛出
        raise

print("\n✅ 合并完成！")
print(f"   基座模型  : {args.base}")
print(f"   LoRA 权重 : {args.lora}")
print(f"   合并输出  : {args.output}")

# 打印合并后文件列表
files = sorted(out_path.iterdir())
total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024**2
print(f"   文件数量  : {len(files)} 个  总大小: {total_mb:.0f} MB")
print("\n🚀 下一步：在 app_fixed.py 里加载此路径，模型名包含 'LoRA' 即可自动路由到 _infer_lora")

