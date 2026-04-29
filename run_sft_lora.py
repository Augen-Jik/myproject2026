#!/usr/bin/env python3
"""
run_sft_lora.py — LoRA 微调脚本（Qwen / DeepSeek-R1 通用）

核心改进（相对于 run_sft_fixed.py / run_sft_r1_v2.py）：
  [FIX1]  引入 LoRA（r=16）防止全量微调的灾难性遗忘
  [FIX2]  max_seq_length 统一提升至 2048，容纳 <think> 推理链
  [FIX3]  使用 apply_chat_template 统一两模型的预处理格式
  [FIX4]  响应端 max_new_tokens 提升到 1800，保证 98 条全输出
  [FIX5]  推理验证时从 </think> 后解析，与训练数据格式完全对齐
  [FIX6]  正则兼容更多格式（冒号/空格/星号）

用法:
  python run_sft_lora.py --config sft_config_lora_qwen.yaml   # Qwen
  python run_sft_lora.py --config sft_config_lora_r1.yaml     # R1
"""

import argparse, json, math, os, re, time
import torch
import yaml
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

import sys
CODE_DIR = "/root/autodl-tmp/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from sparse_utils import build_sparse_task_prompt, parse_sparse_output_bundle

# ── 命令行参数 ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="YAML 配置文件路径")
args = parser.parse_args()

print(f"📄 加载配置: {args.config}")
with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MODEL_PATH  = config["model"]["name_or_path"]
TRUST_RC    = config["model"].get("trust_remote_code", True)
TRAIN_PATH  = config["data"]["train_path"]
EVAL_PATH   = config["data"]["eval_path"]
MAX_LEN     = config["data"]["max_seq_length"]   # 建议 2048
OUTPUT_DIR  = config["training"]["output_dir"]
LR          = float(config["training"]["learning_rate"])
EPOCHS      = config["training"]["num_train_epochs"]
BS          = config["training"].get("per_device_train_batch_size", 4)
ACCUM       = config["training"].get("gradient_accumulation_steps", 4)
LOG_STEPS   = config["training"].get("logging_steps", 20)
SAVE_STEPS  = config["training"].get("save_steps", 200)
EVAL_STEPS  = config["training"].get("eval_steps", SAVE_STEPS)
WARMUP_RATIO = float(config["training"].get("warmup_ratio", 0.1))
LR_SCHEDULER_TYPE = config["training"].get("lr_scheduler_type", "linear")
SAVE_TOTAL_LIMIT = int(config["training"].get("save_total_limit", 3))
SEED        = config.get("seed", 42)

# LoRA 超参（可在 yaml 里覆盖）
LORA_R      = config.get("lora", {}).get("r", 16)
LORA_ALPHA  = config.get("lora", {}).get("alpha", 32)
LORA_DROP   = config.get("lora", {}).get("dropout", 0.05)
INIT_ADAPTER = config.get("lora", {}).get("init_from")
RESUME_CHECKPOINT = config.get("training", {}).get("resume_from_checkpoint")
SMOKE_MODE = config.get("smoke_test", {}).get("mode", "auto")

# ── GPU 验证 ─────────────────────────────────────────────────
print("\n🔍 验证 GPU 环境...")
assert torch.cuda.is_available(), "❌ 需要 GPU！"
gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
use_bf16 = torch.cuda.is_bf16_supported()
print(f"✅ GPU: {torch.cuda.get_device_name(0)} | {gpu_mem:.1f}GB | "
      f"{'BF16' if use_bf16 else 'FP16'}")

try:
    import flash_attn
    has_flash = True
    print(f"✅ Flash Attention: {flash_attn.__version__}")
except ImportError:
    has_flash = False
    print("⚠️  Flash Attention 未安装，使用标准注意力")

# ── 加载 Tokenizer ───────────────────────────────────────────
print(f"\n🔍 加载 Tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=TRUST_RC, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Qwen 系列用 right padding，R1（Qwen2 架构）也适用
tokenizer.padding_side = "right"
print(f"✅ eos_token={repr(tokenizer.eos_token)}  eos_id={tokenizer.eos_token_id}  pad_id={tokenizer.pad_token_id}")

# 验证 chat_template
try:
    test = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True)
    print(f"✅ chat_template 验证通过: {repr(test[:60])}")
except Exception as e:
    raise RuntimeError(f"❌ chat_template 不可用: {e}")

# ── 加载基座模型 ─────────────────────────────────────────────
print(f"\n🔍 加载基座模型（LoRA 模式，不做全量微调）...")
model_kwargs = dict(
    trust_remote_code=TRUST_RC,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    # load_in_4bit 和 load_in_8bit 已从新版 transformers 的 from_pretrained 中移除
    # LoRA 全精度微调不需要量化，直接删除即可
)
if has_flash:
    model_kwargs["attn_implementation"] = "flash_attention_2"

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
model.enable_input_require_grads()   # LoRA 必须开启
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
if getattr(model, "generation_config", None) is not None:
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"✅ 基座模型加载完成 ({time.time()-t0:.1f}s)")

# ── 注入 LoRA ────────────────────────────────────────────────
print(f"\n🔍 注入 LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROP,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
if INIT_ADAPTER:
    print(f"🔁 从已有 LoRA 初始化: {INIT_ADAPTER}")
    model = PeftModel.from_pretrained(model, INIT_ADAPTER, is_trainable=True)
else:
    model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 数据预处理 ───────────────────────────────────────────────
# 统一使用 apply_chat_template，Qwen/R1 均可
# 训练数据已包含 <think>...</think>，无需特殊处理

def preprocess(examples):
    input_ids_list, labels_list, masks = [], [], []

    for msg_str in examples["messages"]:
        msgs = json.loads(msg_str) if isinstance(msg_str, str) else msg_str

        user_content = next((m["content"] for m in msgs if m["role"] == "user"),  "")
        asst_content = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        # 完整对话文本（含 assistant 回复）
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content},
             {"role": "assistant", "content": asst_content}],
            tokenize=False, add_generation_prompt=False)

        # 仅 user 提示部分（用于计算 response 起始位置）
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True)

        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors=None,
        )
        # prefix_enc 不 truncate，以精确获取 prefix 长度
        prefix_enc = tokenizer(
            prefix_text,
            truncation=False,
            add_special_tokens=False,
            return_tensors=None,
        )

        ids    = full_enc["input_ids"]
        mask   = full_enc["attention_mask"]
        plen   = len(prefix_enc["input_ids"])

        # Response-only masking：只对 assistant 回复计算 loss
        labels = [-100] * len(ids)
        for i in range(plen, len(ids)):
            if mask[i] == 1:
                labels[i] = ids[i]

        input_ids_list.append(ids)
        labels_list.append(labels)
        masks.append(mask)

    return {
        "input_ids":      input_ids_list,
        "labels":         labels_list,
        "attention_mask": masks,
    }


print(f"\n🔍 预处理数据集（max_seq_length={MAX_LEN}）...")


def load_and_preprocess(path: str, name: str):
    ds = hf_load_dataset("parquet", data_dir=path)
    split = list(ds.keys())[0]
    ds = ds[split]
    print(f"  [{name}] 原始样本: {len(ds)}")
    ds = ds.map(
        preprocess,
        batched=True,
        batch_size=100,
        num_proc=2,
        remove_columns=ds.column_names,
        desc=f"预处理{name}",
    )
    # 验证 labels 分布
    sample_labels = ds[0]["labels"]
    ignore = sum(1 for l in sample_labels if l == -100)
    learn  = sum(1 for l in sample_labels if l != -100)
    print(f"  [{name}] labels: 忽略={ignore}  学习={learn}")
    if learn == 0:
        raise ValueError(f"❌ [{name}] 没有可学习的 token，请增大 max_seq_length")
    if learn < 200:
        print(f"  ⚠️  [{name}] 可学习 token 仅 {learn}，建议检查数据或增大 max_seq_length")
    return ds


train_dataset = load_and_preprocess(TRAIN_PATH, "训练集")
eval_dataset  = load_and_preprocess(EVAL_PATH,  "验证集")

# ── 训练参数 ──────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BS,
    per_device_eval_batch_size  = BS,
    gradient_accumulation_steps = ACCUM,
    gradient_checkpointing      = True,
    learning_rate               = LR,
    num_train_epochs            = EPOCHS,
    warmup_ratio                = WARMUP_RATIO,
    lr_scheduler_type           = LR_SCHEDULER_TYPE,
    weight_decay                = 0.01,
    max_grad_norm               = 1.0,
    logging_steps               = LOG_STEPS,
    save_steps                  = SAVE_STEPS,
    eval_strategy               = "steps",
    eval_steps                  = EVAL_STEPS,
    save_total_limit            = SAVE_TOTAL_LIMIT,
    load_best_model_at_end      = False,
    bf16                        = use_bf16,
    fp16                        = (not use_bf16),
    dataloader_num_workers      = 2,
    dataloader_pin_memory       = True,
    report_to                   = "none",
    seed                        = SEED,
    remove_unused_columns       = False,
    logging_dir                 = os.path.join(OUTPUT_DIR, "tb_logs"),
)

global_batch = BS * ACCUM
micro_batches_per_epoch = math.ceil(len(train_dataset) / BS)
trainer_updates_per_epoch = math.ceil(micro_batches_per_epoch / ACCUM)
expected_max_steps = math.ceil(EPOCHS * trainer_updates_per_epoch)

print(f"\n🚀 开始 LoRA SFT 训练")
print(f"   模型        : {MODEL_PATH}")
print(f"   训练样本    : {len(train_dataset)}  验证样本: {len(eval_dataset)}")
print(
    "   Epochs      : "
    f"{EPOCHS}  Trainer更新/epoch: {trainer_updates_per_epoch}  预计max_steps: {expected_max_steps}"
)
print(f"   Batch       : {BS} × accum {ACCUM} = {global_batch} global")
print(f"   学习率      : {LR}")
print(f"   调度        : {LR_SCHEDULER_TYPE}  warmup_ratio={WARMUP_RATIO}")
print(f"   LoRA        : r={LORA_R}, alpha={LORA_ALPHA}")
print(f"   Max seq len : {MAX_LEN}")
print(f"   输出目录    : {OUTPUT_DIR}")
print("-" * 60)

# ── 训练 ─────────────────────────────────────────────────────
trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = eval_dataset,
    data_collator = default_data_collator,
)

t_start = time.time()
trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)
elapsed = time.time() - t_start
print(f"\n✅ 训练完成！耗时 {elapsed/3600:.1f}h ({elapsed/60:.0f}min)")

# ── 保存 ─────────────────────────────────────────────────────
print("\n💾 保存 LoRA 适配器权重...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA 权重已保存 → {OUTPUT_DIR}")
print("   (如需合并到基座，可后续运行 merge_lora.py)")

# ── 推理验证 ─────────────────────────────────────────────────
print("\n🔍 推理验证（测试输出格式与解析率）...")
model.eval()

is_sparse_task = SMOKE_MODE == "sparse" or (
    SMOKE_MODE == "auto" and ("sparse" in TRAIN_PATH.lower() or "sparse" in OUTPUT_DIR.lower())
)

test_cases = [
    "黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北畅通无阻。",
    "经六路农业路至红专路段向北施工封闭；经八路向北正常通行。",
    "早高峰07:00-09:00黄河路向东严重拥堵，09:30后恢复正常；当前时间08:30，请按当前时段规划。",
]

smoke_rows = []

for idx, instruction in enumerate(test_cases, 1):
    print(f"\n  [测试 {idx}]")
    if is_sparse_task:
        instruction = build_sparse_task_prompt(instruction)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_LEN
    ).to(model.device)

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1800,     # [FIX4] 足够输出 98 条权重 + think 链
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        parse_target = decoded.split("</think>")[-1].strip() if "</think>" in decoded else decoded
        preview = parse_target if len(parse_target) < 200 else parse_target[:100] + "..." + parse_target[-60:]
        print(f"     输入: {instruction[:60]}...")
        print(f"     输出: {preview}")

        if is_sparse_task:
            bundle = parse_sparse_output_bundle(decoded)
            print(
                f"     解析: anchors={bundle.get('anchor_count', 0)} "
                f"mapped_edges={len(bundle.get('mapped_weights', {}))} "
                f"conf={bundle.get('parse_confidence', 0.0):.2f}"
            )
            if bundle.get("parse_fail"):
                print("     状态: ❌ 稀疏锚点解析失败")
            elif bundle.get("parse_confidence", 0.0) >= 0.75:
                print("     状态: 🎉 稀疏锚点输出稳定")
            else:
                print("     状态: ⚠️  稀疏锚点可解析，但稳定度仍可提升")
            smoke_rows.append(
                {
                    "instruction": instruction,
                    "output_preview": preview,
                    "anchor_count": bundle.get("anchor_count", 0),
                    "mapped_edges": len(bundle.get("mapped_weights", {})),
                    "parse_confidence": bundle.get("parse_confidence", 0.0),
                    "parse_fail": bundle.get("parse_fail", False),
                }
            )
        else:
            WEIGHT_PAT = re.compile(r'([RC]\d+[RC]\d+_[EWNS])\s*[:：]\s*(\d+(?:\.\d+)?)')
            parsed = WEIGHT_PAT.findall(parse_target)
            if len(parsed) < 10:
                parsed = WEIGHT_PAT.findall(decoded)
            print(f"     解析: {len(parsed)} 条权重")
            if len(parsed) >= 90:
                print("     状态: 🎉 输出格式正确！")
            elif len(parsed) >= 60:
                print(f"     状态: ⚠️  解析 {len(parsed)} 条，可继续增加 epoch")
            else:
                print(f"     状态: ❌ 仅 {len(parsed)} 条，建议检查数据格式或增加训练量")
            smoke_rows.append(
                {
                    "instruction": instruction,
                    "output_preview": preview,
                    "parsed_weights": len(parsed),
                }
            )

    except Exception as e:
        print(f"     状态: ❌ 推理出错: {e}")
        smoke_rows.append({"instruction": instruction, "error": str(e)})

with open(os.path.join(OUTPUT_DIR, "smoke_results.json"), "w", encoding="utf-8") as f:
    json.dump(smoke_rows, f, ensure_ascii=False, indent=2)

# ── 完成摘要 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"🎉 全部完成！")
print(f"   模型路径    : {OUTPUT_DIR}")
print(f"   训练耗时    : {elapsed/3600:.1f}h")
print(f"   下一步      :")
print(f"     1. streamlit run app_fixed.py --server.port 6006")
print(f"     2. 若需合并 LoRA：python merge_lora.py --base {MODEL_PATH} --lora {OUTPUT_DIR}")
print("=" * 60)
