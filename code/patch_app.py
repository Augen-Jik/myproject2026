#!/usr/bin/env python3
"""
patch_app.py — 对 app_fixed.py 进行三处精准修复

修复内容：
  [PATCH-1] _extract_weights_from_text：增强正则，兼容 <think> 输出后的权重格式
  [PATCH-2] 新增 _infer_lora()：专供新 LoRA 模型的推理方法（chat_template + CoT 解析）
  [PATCH-3] generate_weights()：识别 "LoRA" 模型名，路由到 _infer_lora

用法:
  python patch_app.py                           # 默认修改 app_fixed.py
  python patch_app.py --input app_fixed.py --output app_fixed_patched.py
"""

import argparse, shutil, re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="app_fixed.py")
parser.add_argument("--output", default=None,
                    help="输出文件；默认覆盖原文件（自动备份为 .bak）")
args = parser.parse_args()

src = Path(args.input)
dst = Path(args.output) if args.output else src

# 自动备份
bak = src.with_suffix(".py.bak")
shutil.copy2(src, bak)
print(f"✅ 已备份原文件 → {bak}")

code = src.read_text(encoding="utf-8")
orig_len = len(code)


# ═══════════════════════════════════════════════════════════════
# PATCH-1: 增强 _extract_weights_from_text
#   旧版在遇到 <think>...</think> 长文本时，因正则先 replace \n→, 再匹配，
#   可能匹配到 think 链里伪造的数字对。
#   修复：先把 </think> 前的内容截掉，再做提取。
# ═══════════════════════════════════════════════════════════════
OLD_EXTRACT = '''    def _extract_weights_from_text(_self, text):
        """多策略从文本中提取 EdgeID:数字 对，兼容各种模型输出格式"""
        if not text:
            return {}
        # 预处理：统一分隔符、去除markdown标记
        cleaned = (text
                   .replace("：", ":").replace("，", ",")
                   .replace("**", "").replace("__", "")
                   .replace("\\n", ",").replace("\\t", ",")
                   .replace(" ", ""))
        valid_h = re.compile(r"^R[0-4]C[0-4]_[EW]$")
        valid_v = re.compile(r"^C[0-5]R[0-3]_[NS]$")
        wd = {}
        for eid, w in re.findall(r"(R\\d+C\\d+_[EW]|C\\d+R\\d+_[NS]):(\\d+(?:\\.\\d+)?)", cleaned):
            if (valid_h.match(eid) or valid_v.match(eid)):
                val = float(w)
                if 0 <= val <= 10:
                    wd[eid] = val
        return wd'''

# [PATCH-1] 新版提取函数：
#   Step1 —— 若含 </think>，先取其后内容；否则取全文（与用户建议完全一致）
#   Step2 —— 统一预处理后用改进正则匹配
#   Step3 —— 若 </think> 后段落匹配到 0 条，fallback 到全文再匹配一次
#   全程无递归，无 __wrapped__，逻辑清晰
NEW_EXTRACT = '''    def _extract_weights_from_text(_self, text):
        """多策略从文本中提取 EdgeID:数字 对，兼容各种模型输出格式
        [PATCHED] 先在 </think> 之后截断再匹配，防止 CoT 链干扰解析
        """
        if not text:
            return {}

        def _parse(src: str) -> dict:
            """核心解析：预处理 → 正则匹配 → 范围校验"""
            cleaned = (src
                       .replace("：", ":").replace("，", ",")
                       .replace("**", "").replace("__", "").replace("*", "")
                       .replace("\\n", ",").replace("\\t", ",")
                       .replace(" ", ""))
            valid_h = re.compile(r"^R[0-4]C[0-4]_[EW]$")
            valid_v = re.compile(r"^C[0-5]R[0-3]_[NS]$")
            result = {}
            # 兼容中英文冒号；原有正则同时保留以向后兼容
            for eid, w in re.findall(
                r"([RC]\\d+[RC]\\d+_[EWNS])\\s*[:：]\\s*(\\d+(?:\\.\\d+)?)",
                cleaned
            ):
                if (valid_h.match(eid) or valid_v.match(eid)):
                    val = float(w)
                    if 0 <= val <= 10:
                        result[eid] = val
            return result

        # Step 1: 优先从 </think> 之后解析（用户建议核心改动）
        if "</think>" in text:
            after_think = text.split("</think>")[-1]
            wd = _parse(after_think)
            # Step 3: 若 </think> 后解析为 0，fallback 全文（防止格式异常时丢权重）
            if not wd:
                wd = _parse(text)
        else:
            wd = _parse(text)

        return wd'''

# ── 执行 PATCH-1 ─────────────────────────────────────────────
if OLD_EXTRACT in code:
    code = code.replace(OLD_EXTRACT, NEW_EXTRACT)
    print("✅ PATCH-1 applied: _extract_weights_from_text 增强完成（</think> 截断 + 全文 fallback）")
else:
    print("⚠️  PATCH-1 跳过: 未找到原始 _extract_weights_from_text（可能已修改）")


# ═══════════════════════════════════════════════════════════════
# PATCH-2: 在 _infer_r1_sft 之后插入新方法 _infer_lora
#   供 LoRA 合并后的模型（Qwen-LoRA / R1-LoRA）使用
#   使用 apply_chat_template，正确处理 <think> 输出
# ═══════════════════════════════════════════════════════════════
LORA_METHOD = '''
    def _infer_lora(_self, tokenizer, model, constraint, device):
        """LoRA 合并模型推理（Qwen-LoRA / R1-LoRA 通用）
        [PATCHED] 使用 apply_chat_template，兼容 <think> CoT 输出
        训练数据格式: apply_chat_template + <think>推理链</think>\\n权重列表
        max_new_tokens=1800 保证 <think> + 98条权重能完整输出
        """
        # 提示语与训练数据完全对齐（generate_dataset_cot.py 格式）
        instruction = (
            "你是交通路径权重生成助手，根据描述为路段分配拥堵权重（0-10，越大越拥堵）。\\n"
            "路网：农业路=R0,红专路=R1,政七街=R2,黄河路=R3,纬五路=R4；"
            "经一路=C0,经三路=C1,经六路=C2,经八路=C3,花园路=C4,未来路=C5。\\n"
            f"{constraint}"
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1800, padding=False
        ).to(device)
        in_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1800,      # 足够容纳 <think> + 98条权重
                do_sample=False,
                repetition_penalty=1.05,  # 轻微惩罚防止死循环，不截断权重列表
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()

        # 优先从 </think> 之后提取权重
        if "</think>" in decoded:
            struct = decoded.split("</think>")[-1].strip()
            # 验证有效性，若解析过少则 fallback 全文
            quick_check = len(re.findall(
                r'[RC]\\d+[RC]\\d+_[EWNS]\\s*[:：]\\s*\\d', struct))
            if quick_check < 10:
                struct = decoded
        else:
            # 没有 think 标签：找第一个 EdgeID 开始的位置
            m = re.search(r'(R[0-4]C[0-4]_[EW]|C[0-5]R[0-3]_[NS])\\s*[:：]', decoded)
            struct = decoded[m.start():] if m else decoded

        logging.info(f"LoRA raw[:200]: {repr(decoded[:200])}")
        logging.info(f"LoRA struct[:200]: {repr(struct[:200])}")
        return decoded, struct, time.time() - t0

'''

# 在 _infer_r1_sft 方法结束后插入（找 _infer_r1_raw 定义前）
ANCHOR_BEFORE_R1_RAW = "    def _infer_r1_raw("
if ANCHOR_BEFORE_R1_RAW in code and "_infer_lora" not in code:
    code = code.replace(ANCHOR_BEFORE_R1_RAW,
                        LORA_METHOD + "    def _infer_r1_raw(")
    print("✅ PATCH-2 applied: 新增 _infer_lora 方法")
elif "_infer_lora" in code:
    print("⚠️  PATCH-2 跳过: _infer_lora 已存在")
else:
    print("⚠️  PATCH-2 跳过: 未找到 _infer_r1_raw 锚点")


# ═══════════════════════════════════════════════════════════════
# PATCH-3: generate_weights 中增加 LoRA 模型路由
#   识别 model_name 中的 "LoRA" 关键字，调用 _infer_lora
# ═══════════════════════════════════════════════════════════════
OLD_ROUTE = '''            _is_r1_sft = "R1" in model_name and "SFT" in model_name
            _is_r1_raw = "R1" in model_name and "Raw" in model_name

            if _is_r1_sft:
                # R1-SFT: 用apply_chat_template + 训练时的指令格式（无边ID列表）
                raw, struct, infer_time = _self._infer_r1_sft(tokenizer, model, constraint, device)
            elif _is_r1_raw:
                # R1-Raw: 用apply_chat_template + 完整边ID列表（zero-shot约束）
                raw, struct, infer_time = _self._infer_r1_raw(tokenizer, model, constraint, device)
            else:
                # Qwen-SFT / Qwen-Raw: ### Instruction格式
                raw, struct, infer_time = _self._infer_qwen(tokenizer, model, constraint, device)'''

NEW_ROUTE = '''            _is_lora    = "LoRA" in model_name or "lora" in model_name
            _is_r1_sft  = "R1" in model_name and "SFT" in model_name and not _is_lora
            _is_r1_raw  = "R1" in model_name and "Raw" in model_name and not _is_lora

            if _is_lora:
                # [PATCHED] LoRA 合并模型（Qwen-LoRA / R1-LoRA）：CoT 格式
                raw, struct, infer_time = _self._infer_lora(tokenizer, model, constraint, device)
            elif _is_r1_sft:
                # R1-SFT: 用apply_chat_template + 训练时的指令格式
                raw, struct, infer_time = _self._infer_r1_sft(tokenizer, model, constraint, device)
            elif _is_r1_raw:
                # R1-Raw: 用apply_chat_template + 完整边ID列表（zero-shot约束）
                raw, struct, infer_time = _self._infer_r1_raw(tokenizer, model, constraint, device)
            else:
                # Qwen-SFT / Qwen-Raw: ### Instruction格式（旧模型保持兼容）
                raw, struct, infer_time = _self._infer_qwen(tokenizer, model, constraint, device)'''

if OLD_ROUTE in code:
    code = code.replace(OLD_ROUTE, NEW_ROUTE)
    print("✅ PATCH-3 applied: generate_weights 增加 LoRA 路由")
else:
    print("⚠️  PATCH-3 跳过: 未找到原始路由代码（可能已修改）")


# ═══════════════════════════════════════════════════════════════
# PATCH-4: 撤销之前错误注入的 </think>\n\n
#   如果 _infer_r1_sft 中有 + "</think>\n\n"，删除之
# ═══════════════════════════════════════════════════════════════
if '"</think>\\n\\n"' in code or '"</think>\\\\n\\\\n"' in code:
    # 找到注入行并删除
    lines = code.splitlines()
    new_lines = []
    removed = 0
    for line in lines:
        if ('</think>' in line and '\\n\\n' in line and
                ('prompt' in line or 'prompt_a' in line) and
                '+' in line):
            print(f"  🗑  删除注入行: {line.strip()}")
            removed += 1
        else:
            new_lines.append(line)
    if removed:
        code = "\n".join(new_lines)
        print(f"✅ PATCH-4 applied: 已删除 {removed} 处 </think> 强行注入")
    else:
        print("ℹ️  PATCH-4: 未找到 </think> 注入行（可能已清除）")
else:
    print("ℹ️  PATCH-4: 未检测到 </think> 注入（无需处理）")


# ── 写入文件 ─────────────────────────────────────────────────
dst.write_text(code, encoding="utf-8")
new_len = len(code)
print(f"\n✅ 补丁完成 → {dst}")
print(f"   原始大小: {orig_len:,} 字符  补丁后: {new_len:,} 字符  差值: {new_len-orig_len:+,}")
print(f"   备份文件: {bak}")
print("\n下一步：")
print("  1. python generate_dataset_cot.py   # 生成 CoT 数据集")
print("  2. python run_sft_lora.py --config sft_config_lora_qwen.yaml  # 训练 Qwen-LoRA")
print("  3. python run_sft_lora.py --config sft_config_lora_r1.yaml    # 训练 R1-LoRA")
print("  4. python merge_lora.py --base <基座路径> --lora <LoRA路径> --output <合并路径>")
print("  5. streamlit run app_fixed.py --server.port 6006")
