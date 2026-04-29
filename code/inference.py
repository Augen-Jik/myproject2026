"""Inference helpers for LLM weight generation and fallback orchestration."""

import logging
import os
import re
import time

import torch

from scenarios import classify_scene_type
from sparse_utils import build_sparse_task_prompt as shared_build_sparse_task_prompt
from sparse_utils import parse_sparse_output_bundle


SPARSE_OUTPUT_TEMPLATE = (
    "TASK=sparse_output\n"
    "SCENE_TYPE={scene_type}\n"
    "OUTPUT_OBJECT=异常语义锚点\n"
    "OUTPUT_FORMAT=ANCHOR|ROAD=道路名|DIR=方向|RANGE=范围|LEVEL=等级\n"
    "CONFLICT_POLICY=同一路段同方向同范围若冲突，保留更严重异常；不要补全全量边权。\n"
    "{event_lines}\n"
    "仅输出异常语义锚点，不要解释。"
)

BASELINE_SPARSE_OUTPUT_TEMPLATE = (
    "TASK=sparse_output\n"
    "交通描述：{constraint}\n"
    "请直接输出异常语义锚点，格式：ANCHOR|ROAD=道路名|DIR=方向|RANGE=范围|LEVEL=等级。\n"
    "不要输出全量边权，不要解释。"
)

_LEVEL_HINTS = [
    ("封闭", ("全封", "封闭", "管制", "禁行", "禁止通行")),
    ("极度拥堵", ("极度拥堵", "极度", "堵塞")),
    ("严重拥堵", ("严重拥堵", "严重事故", "追尾事故", "追尾")),
    ("中度拥堵", ("中度拥堵", "中度", "缓行", "排队")),
    ("轻度拥堵", ("轻微拥堵", "轻微", "较慢")),
    ("畅通", ("畅通", "通畅", "无阻", "顺畅")),
    ("正常", ("正常", "正常通行")),
]

_DIR_TOKENS = [
    ("向东", "向东"),
    ("东向", "向东"),
    ("向西", "向西"),
    ("西向", "向西"),
    ("向北", "向北"),
    ("北向", "向北"),
    ("向南", "向南"),
    ("南向", "向南"),
]


def _strip_legacy_generation_suffix(constraint: str, total_edges: int) -> str:
    base = str(constraint or "")
    old_endings = [
        f"请为全部{total_edges}条路段生成权重（0-10，越大越拥堵）。",
        f"请按「路段ID:权重值」格式输出所有{total_edges}条路段权重。",
        f"请为郑州市金水区路网的{total_edges}条路段分配拥堵权重，越大越拥堵。",
        "请严格按格式输出各路段权重（路段ID:数值），用于自动驾驶路径规划。",
        f"基于上述路况，为全路网{total_edges}条路段输出权重（0-10），请勿遗漏任何路段。",
    ]
    for ending in old_endings:
        base = base.replace(ending, "")
    return base.strip().rstrip("。；; ")


def _extract_level_label(clause: str) -> str:
    for label, keywords in _LEVEL_HINTS:
        if any(keyword in clause for keyword in keywords):
            return label
    return "待判断"


def _extract_time_label(clause: str) -> str:
    patterns = [
        r"当前时间\s*([0-2]?\d[:：][0-5]\d)",
        r"([0-2]?\d[:：][0-5]\d\s*[-~到至]\s*[0-2]?\d[:：][0-5]\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clause)
        if match:
            return match.group(1).replace("：", ":").replace(" ", "")
    for token in ("早高峰", "晚高峰", "早高峰前", "晚高峰前", "平峰", "午间", "夜间", "工作日", "周末", "节假日"):
        if token in clause:
            return token
    for token in ("后恢复", "恢复正常", "散场", "切换", "临时管制"):
        if token in clause:
            return token
    return "未指明"


def _extract_propagation_label(clause: str) -> str:
    if any(token in clause for token in ("外溢", "波及", "扩散", "蔓延", "回溢", "传播", "排队至", "影响到")):
        range_hint = re.search(
            r"(农业路|红专路|政七街|黄河路|纬五路|经一路|经三路|经六路|经八路|花园路|未来路)[至到](农业路|红专路|政七街|黄河路|纬五路|经一路|经三路|经六路|经八路|花园路|未来路)",
            clause,
        )
        if range_hint:
            return f"波及:{range_hint.group(1)}至{range_hint.group(2)}"
        return "存在传播/外溢"
    return "无明显传播"


def _extract_conflict_label(clause: str) -> str:
    if any(token in clause for token in ("升级", "但", "然而", "同时", "恢复", "切换", "优先以", "以当前时段")):
        return "存在冲突/切换"
    return "单约束"


def _extract_prompt_event(engine, clause: str) -> dict | None:
    all_roads = list(getattr(engine, "_ROW_MAP", {}).keys()) + list(getattr(engine, "_COL_MAP", {}).keys())
    if not all_roads:
        return None
    road_pattern = "|".join(re.escape(name) for name in all_roads)
    range_match = re.search(rf"({road_pattern})[至到]({road_pattern})", clause)
    range_text = "全线" if any(token in clause for token in ("全线", "全段", "全程")) else (f"{range_match.group(1)}至{range_match.group(2)}" if range_match else "局部")

    dir_cn = next((canonical for token, canonical in _DIR_TOKENS if token in clause), None)
    if dir_cn in ("向东", "向西"):
        road = next((name for name in engine._ROW_MAP if name in clause), None)
    elif dir_cn in ("向北", "向南"):
        road = next((name for name in engine._COL_MAP if name in clause), None)
    else:
        road = next((name for name in engine._ROW_MAP if name in clause), None)
        if road is None:
            road = next((name for name in engine._COL_MAP if name in clause), None)

    if road is None and range_match:
        road = range_match.group(1)

    level = _extract_level_label(clause)
    if road is None and level == "待判断":
        return None

    return {
        "road": road or "未识别",
        "dir": dir_cn or ("双向" if "双向" in clause else "未指明"),
        "range": range_text,
        "level": level,
        "time": _extract_time_label(clause),
        "propagation": _extract_propagation_label(clause),
        "conflict": _extract_conflict_label(clause),
        "text": clause.strip(),
    }


def build_sparse_baseline_prompt(_engine, constraint: str) -> str:
    return BASELINE_SPARSE_OUTPUT_TEMPLATE.format(
        constraint=str(constraint or "").strip(),
    )


def build_sparse_task_prompt(engine, constraint: str) -> str:
    total_edges = len(getattr(engine, "_ALL_EDGES", []))
    prompt = shared_build_sparse_task_prompt(constraint, total_edges=total_edges)
    scene_type = "simple_local"
    for line in prompt.splitlines():
        if line.startswith("SCENE_TYPE="):
            scene_type = line.split("=", 1)[1].strip() or "simple_local"
            break
    setattr(engine, "_last_scene_type", scene_type)
    return prompt


def build_sparse_instruction(engine, constraint: str, prompt_mode: str = "spatiotemporal_enhanced_prompt") -> str:
    if prompt_mode == "baseline_sparse_prompt":
        setattr(engine, "_last_scene_type", classify_scene_type(text=constraint))
        return build_sparse_baseline_prompt(engine, constraint)
    if prompt_mode == "spatiotemporal_enhanced_prompt":
        return build_sparse_task_prompt(engine, constraint)
    raise ValueError(f"unknown sparse prompt_mode: {prompt_mode}")


def infer_qwen(engine, tokenizer, model, constraint, device):
    """Qwen/SFT推理：### Instruction格式，与训练完全对齐，保持97/98解析率。"""
    prompt = (
        "### Instruction:\n"
        "你是交通路径权重生成助手，根据描述为路段分配拥堵权重（0-10，越大越拥堵）。\n"
        "只输出键值对，格式：路段ID:权重值, 路段ID:权重值\n"
        "示例：R3C0_E:8.5, C4R1_N:1.5, R0C2_E:5.0\n"
        "权重映射：封闭=9.5, 严重拥堵=7.5, 中度拥堵=5.0, 正常=2.0, 畅通=1.2\n"
        "路网：农业路=R0,红专路=R1,政七街=R2,黄河路=R3,纬五路=R4；"
        "经一路=C0,经三路=C1,经六路=C2,经八路=C3,花园路=C4,未来路=C5\n\n"
        f"交通描述：{constraint}\n\n### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900, padding=True).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    struct = raw.split("### Response:")[-1].strip() if "### Response:" in raw else raw
    return raw, struct, time.time() - t0


def infer_r1_sft(engine, tokenizer, model, constraint, device):
    """R1-SFT双格式推理：chat_template与###Instruction并行尝试，取解析更优结果。"""
    full_instr = (
        "你是交通路径权重生成助手，根据描述为路段分配拥堵权重（0-10，越大越拥堵）。\n"
        "只输出键值对，格式：路段ID:权重值, 路段ID:权重值\n"
        "权重映射：封闭=9.5, 严重拥堵=7.5, 中度拥堵=5.0, 正常=2.0, 畅通=1.2\n"
        "路网：农业路=R0,红专路=R1,政七街=R2,黄河路=R3,纬五路=R4；"
        "经一路=C0,经三路=C1,经六路=C2,经八路=C3,花园路=C4,未来路=C5\n"
        f"交通描述：{constraint}"
    )

    def _run_infer(prompt_text, add_special=False, max_len=1200, penalty=1.0):
        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            add_special_tokens=add_special,
            padding=False,
        ).to(device)
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=3000,
                do_sample=False,
                temperature=1.0,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=penalty,
            )
        decoded = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
        if "</think>" in decoded:
            import re as _re

            after = decoded.split("</think>")[-1].strip()
            n = len(_re.findall(r"R[0-4]C[0-4]_[EW]:[0-9]|C[0-5]R[0-3]_[NS]:[0-9]", after))
            struct = after if n >= 5 else decoded
        else:
            import re as _re

            m = _re.search(r"(R[0-4]C[0-4]_[EW]|C[0-5]R[0-3]_[NS])\s*:\s*\d", decoded)
            struct = decoded[m.start() :] if m else decoded
        return decoded, struct

    t0 = time.time()
    prompt_a = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_instr}],
        tokenize=False,
        add_generation_prompt=True,
    )
    raw_a, struct_a = _run_infer(prompt_a, add_special=False, penalty=1.0)
    wd_a = engine._extract_weights_from_text(struct_a) or engine._extract_weights_from_text(raw_a)
    logging.info(f"R1-SFT raw[:200]: {repr(raw_a[:200])}")
    logging.info(f"R1-SFT struct[:200]: {repr(struct_a[:200])}")
    logging.info(f"R1-SFT wd_a count: {len(wd_a)}")

    if len(wd_a) < 30:
        prompt_b = (
            "### Instruction:\n"
            + full_instr.replace(f"交通描述：{constraint}", f"交通描述：{constraint}\n\n### Response:\n")
        )
        raw_b, struct_b = _run_infer(prompt_b, add_special=True, max_len=900, penalty=1.05)
        wd_b = engine._extract_weights_from_text(struct_b) or engine._extract_weights_from_text(raw_b)
        if len(wd_b) > len(wd_a):
            raw_a, struct_a = raw_b, struct_b
            logging.info(f"R1-SFT: ### format wins ({len(wd_b)} vs {len(wd_a)})")
        else:
            logging.info(f"R1-SFT: chat_template format used ({len(wd_a)} vs {len(wd_b)})")

    return raw_a, struct_a, time.time() - t0


def infer_lora(engine, tokenizer, model, constraint, device):
    """LoRA 合并模型通用推理（Qwen-LoRA / R1-LoRA）。"""
    import re as _re
    import time as _time

    total_edges = len(engine._ALL_EDGES)
    _endings_kw = ["生成权重", "输出权重", "分配", "路段权重", "权重值"]
    if not any(kw in constraint for kw in _endings_kw):
        instruction = constraint.rstrip("。") + f"。请为全部{total_edges}条路段生成权重（0-10，越大越拥堵）。"
    else:
        instruction = constraint

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096, padding=False).to(device)
    in_len = inputs["input_ids"].shape[1]

    t0 = _time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=3000,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()

    if "</think>" in decoded:
        struct = decoded.split("</think>")[-1].strip()
        quick = len(_re.findall(r"[RC]\d+[RC]\d+_[EWNS]\s*[:：]\s*\d", struct))
        if quick < 10:
            struct = decoded
    else:
        m = _re.search(r"(R[0-4]C[0-4]_[EW]|C[0-5]R[0-3]_[NS])\s*[:：]", decoded)
        struct = decoded[m.start() :] if m else decoded

    logging.info(f"LoRA raw[:200]: {repr(decoded[:200])}")
    return decoded, struct, _time.time() - t0


def infer_sparse(engine, tokenizer, model, constraint, device, prompt_mode: str = "spatiotemporal_enhanced_prompt"):
    """稀疏输出模型专用推理（Sparse-LoRA，异常语义锚点格式）。"""
    import re as _re
    import time as _time

    instruction = build_sparse_instruction(engine, constraint, prompt_mode=prompt_mode)
    setattr(engine, "_last_sparse_instruction", instruction)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    setattr(engine, "_last_sparse_prompt", prompt)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768, padding=False).to(device)
    in_len = inputs["input_ids"].shape[1]

    t0 = _time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()

    if "</think>" in decoded:
        struct = decoded.split("</think>")[-1].strip()
    else:
        anchor_pos = decoded.find("ANCHOR|ROAD=")
        if anchor_pos >= 0:
            struct = decoded[anchor_pos:]
        else:
            edge_match = _re.search(r"(R[0-4]C[0-4]_[EW]|C[0-5]R[0-3]_[NS])\s*:", decoded)
            struct = decoded[edge_match.start():] if edge_match else decoded

    logging.info(f"Sparse raw[:180]: {repr(decoded[:180])}")
    return decoded, struct, _time.time() - t0


def infer_sparse_no_st(engine, tokenizer, model, constraint, device):
    """No-SpatioTemporal ablation: keep the same sparse model/parser, remove structured prompt injection only."""
    return infer_sparse(
        engine,
        tokenizer,
        model,
        constraint,
        device,
        prompt_mode="baseline_sparse_prompt",
    )


def infer_r1_raw(engine, tokenizer, model, constraint, device):
    """R1-Raw推理：未微调基座，zero-shot需要完整边ID列表约束，防止幻觉。"""
    total_edges = len(engine._ALL_EDGES)
    r1_instr = (
        f"你是交通路径权重生成助手。请根据交通描述，为郑州金水区路网全部{total_edges}条路段分配拥堵权重。\n"
        "【输出要求】思考完成后只输出权重键值对，格式：路段ID:数值\n"
        "权重映射：封闭=9.5, 严重拥堵=7.5, 中度拥堵=5.0, 正常=2.0, 畅通=1.2\n"
        "路网对照：农业路=R0行,红专路=R1行,政七街=R2行,黄河路=R3行,纬五路=R4行；"
        "经一路=C0列,经三路=C1列,经六路=C2列,经八路=C3列,花园路=C4列,未来路=C5列\n"
        "合法路段ID（只能用这98个，禁用中文路名）：\n"
        f"{engine._EDGE_LIST_STR}\n"
        f"交通描述：{constraint}"
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": r1_instr}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1200,
        add_special_tokens=False,
        padding=False,
    ).to(device)
    input_len = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=3000,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    if "</think>" in raw:
        struct = raw.split("</think>")[-1].strip()
    else:
        m = re.search(r"(R[0-4]C[0-4]_[EW]|C[0-5]R[0-3]_[NS])\s*:\s*\d", raw)
        struct = raw[m.start() :] if m else raw
    return raw, struct, time.time() - t0


def generate_weights(
    engine,
    constraint,
    model_name,
    model_manager,
    use_gat=False,
    gat_model_path="/root/autodl-tmp/gat_model_v2_gated.pt",
    gat_alpha=0.65,
    mode=None,
    prompt_mode: str = "spatiotemporal_enhanced_prompt",
):
    """Generate edge weights while preserving existing fallback semantics and outputs."""
    try:
        t_func0 = time.perf_counter()
        run_mode = (mode or engine.config.get("RUN_MODE", "demo_mode") or "demo_mode").strip().lower()
        if run_mode not in {"demo_mode", "experiment_mode"}:
            logging.warning(f"未知mode={run_mode}，回退demo_mode")
            run_mode = "demo_mode"
        fallback_events = []
        fallback_used = False
        fallback_reason = "none"
        parse_bundle = {
            "anchors": [],
            "anchor_count": 0,
            "mapped_weights": {},
            "edge_confidence": {},
            "parse_confidence": 0.0,
            "scene_type": classify_scene_type(text=constraint),
            "protected_edges": [],
        }
        gat_meta = {
            "gat_mode": "disabled",
            "high_conf_edges": [],
            "low_conf_edges": [],
            "missing_edges": [],
            "protected_edges": [],
        }
        setattr(engine, "_last_sparse_parse", parse_bundle)
        setattr(engine, "_last_gat_meta", gat_meta)

        cold_start = model_name not in getattr(model_manager, "models", {})
        t_load0 = time.perf_counter()
        tokenizer, model = model_manager.get_model(model_name)
        init_ms = (time.perf_counter() - t_load0) * 1000 if cold_start else 0.0
        device = engine.config.get("DEVICE", "cpu")

        is_sparse = "Sparse" in model_name or "sparse" in model_name
        is_lora = ("LoRA" in model_name or "lora" in model_name) and not is_sparse
        is_r1_sft = "R1" in model_name and "SFT" in model_name and not is_lora and not is_sparse
        is_r1_raw = "R1" in model_name and "Raw" in model_name and not is_lora and not is_sparse

        t_infer0 = time.perf_counter()
        if is_sparse:
            raw, struct, infer_time = infer_sparse(
                engine,
                tokenizer,
                model,
                constraint,
                device,
                prompt_mode=prompt_mode,
            )
        elif is_lora:
            raw, struct, infer_time = infer_lora(engine, tokenizer, model, constraint, device)
        elif is_r1_sft:
            raw, struct, infer_time = infer_r1_sft(engine, tokenizer, model, constraint, device)
        elif is_r1_raw:
            raw, struct, infer_time = infer_r1_raw(engine, tokenizer, model, constraint, device)
        else:
            raw, struct, infer_time = infer_qwen(engine, tokenizer, model, constraint, device)
        infer_ms = max(0.0, (time.perf_counter() - t_infer0) * 1000.0)
        smooth_ms = 0.0

        wd_struct = engine._extract_weights_from_text(struct)
        wd = dict(wd_struct)
        if len(wd) < 3:
            wd_raw = engine._extract_weights_from_text(raw)
            if len(wd_raw) > len(wd):
                fallback_events.append(f"parse_source:struct->raw ({len(wd)}->{len(wd_raw)})")
                wd = wd_raw

        raw_llm_weights = dict(wd)
        llm_parsed_count = len(raw_llm_weights)
        explicit_wd = dict(raw_llm_weights)
        final_base_weights = dict(raw_llm_weights)

        if is_sparse:
            parse_bundle = parse_sparse_output_bundle(
                raw or struct,
                scene_type=getattr(engine, "_last_scene_type", None),
            )
            setattr(engine, "_last_sparse_parse", parse_bundle)
            mapped_weights = dict(parse_bundle.get("mapped_weights", {}))
            if mapped_weights:
                if len(mapped_weights) != len(raw_llm_weights):
                    fallback_events.append(
                        f"sparse_anchor_parser ({len(raw_llm_weights)}->{len(mapped_weights)})"
                    )
                raw_llm_weights = mapped_weights
                llm_parsed_count = len(raw_llm_weights)
                explicit_wd = dict(raw_llm_weights)
                final_base_weights = dict(raw_llm_weights)
                logging.info(
                    "Sparse语义锚点解析: anchors=%s mapped_edges=%s conf=%.2f",
                    parse_bundle.get("anchor_count", 0),
                    llm_parsed_count,
                    parse_bundle.get("parse_confidence", 0.0),
                )
        else:
            if llm_parsed_count < 10 and raw:
                wd_chinese = engine._parse_chinese_weights(raw)
                if len(wd_chinese) > llm_parsed_count:
                    fallback_events.append(f"chinese_parser ({llm_parsed_count}->{len(wd_chinese)})")
                    logging.info(f"R1中文解析成功: {len(wd_chinese)}条")
                    raw_llm_weights = wd_chinese
                    llm_parsed_count = len(wd_chinese)
                    explicit_wd = dict(raw_llm_weights)
                    final_base_weights = dict(raw_llm_weights)

            if llm_parsed_count < 10:
                if run_mode == "demo_mode":
                    logging.warning(f"LLM仅显式解析{llm_parsed_count}条，demo_mode 启用规则兜底")
                    fallback_events.append(f"rule_fallback_enabled (explicit={llm_parsed_count})")
                    fallback_used = True
                    fallback_reason = "rule_fallback_enabled"
                    final_base_weights = engine._rule_parse_weights(constraint)
                    raw = (raw or "") + f"\n[⚠️ demo_mode: 模型显式输出仅{llm_parsed_count}条，已用规则解析兜底]"
                else:
                    logging.warning(f"LLM仅显式解析{llm_parsed_count}条，experiment_mode 禁止规则兜底")
                    fallback_events.append(f"rule_fallback_blocked (explicit={llm_parsed_count})")
                    fallback_used = False
                    fallback_reason = "rule_fallback_blocked"
                    raw = (raw or "") + f"\n[⚠️ experiment_mode: 模型显式输出仅{llm_parsed_count}条，禁止规则兜底]"

        final_planning_weights = {e: 2.0 for e in engine._ALL_EDGES}
        final_planning_weights.update(final_base_weights)

        gat_applied = False
        gat_alpha_used = gat_alpha
        if use_gat:
            try:
                import copy
                import sys

                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                if not os.path.exists(gat_model_path):
                    logging.warning(f"GAT模型不存在({gat_model_path})，跳过GAT以保护结果")
                    fallback_events.append("gat_skipped_missing_model")
                    gat_meta["gat_mode"] = "skipped_missing_model"
                    raise FileNotFoundError(f"GAT未训练: {gat_model_path}")

                from gat_smoother import smooth_weights

                t_s0 = time.perf_counter()
                final_planning_weights, gat_alpha_used, gat_meta = smooth_weights(
                    copy.deepcopy(final_planning_weights),
                    model_path=gat_model_path,
                    alpha=gat_alpha,
                    device=device,
                    edge_confidence=parse_bundle.get("edge_confidence") if is_sparse else None,
                    protected_edges=parse_bundle.get("protected_edges") if is_sparse else None,
                    return_metadata=True,
                )
                smooth_ms = (time.perf_counter() - t_s0) * 1000
                gat_applied = True
                logging.info(f"GAT完成，权重数: {len(final_planning_weights)} | mode={gat_meta.get('gat_mode')}")
            except FileNotFoundError as fe:
                logging.warning(f"GAT跳过(未训练): {fe}")
            except Exception as ge:
                gat_meta["gat_mode"] = f"skipped_error:{type(ge).__name__}"
                fallback_events.append(f"gat_skipped_error:{type(ge).__name__}")
                logging.warning(f"GAT跳过: {ge}")

        setattr(engine, "_last_gat_meta", gat_meta)

        total_ms = max(0.0, (time.perf_counter() - t_func0) * 1000.0)
        if total_ms + 1e-6 < (infer_ms + smooth_ms):
            total_ms = infer_ms + smooth_ms

        timing_info = {
            "init_ms": round(init_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "smooth_ms": round(smooth_ms, 2),
            "total_ms": round(total_ms, 2),
            "cold_start": bool(cold_start),
            "scene_type": parse_bundle.get("scene_type", classify_scene_type(text=constraint)),
            "anchor_count": int(parse_bundle.get("anchor_count", 0)),
            "parse_confidence": round(float(parse_bundle.get("parse_confidence", 0.0)), 3),
            "gat_mode": gat_meta.get("gat_mode", "disabled"),
            "high_conf_edges": list(gat_meta.get("high_conf_edges", [])),
            "low_conf_edges": list(gat_meta.get("low_conf_edges", [])),
            "missing_edges": list(gat_meta.get("missing_edges", [])),
        }

        logging.info(
            f"Weights {infer_time:.2f}s | explicit={llm_parsed_count} | lora={is_lora} | gat={gat_applied} | scene={timing_info['scene_type']}"
        )
        return (
            struct,
            raw_llm_weights,
            final_planning_weights,
            infer_time,
            raw,
            llm_parsed_count,
            gat_applied,
            gat_alpha_used,
            explicit_wd,
            run_mode,
            fallback_used,
            fallback_reason,
            timing_info,
        )
    except Exception as exc:
        logging.error(f"generate_weights error: {exc}")
        return (
            "",
            {},
            {},
            0.0,
            f"ERROR: {exc}",
            0,
            False,
            gat_alpha,
            {},
            (mode or engine.config.get("RUN_MODE", "demo_mode") or "demo_mode"),
            False,
            f"generate_weights_error:{type(exc).__name__}",
            {
                "init_ms": 0.0,
                "infer_ms": 0.0,
                "smooth_ms": 0.0,
                "total_ms": 0.0,
                "cold_start": False,
                "scene_type": classify_scene_type(text=constraint),
                "anchor_count": 0,
                "parse_confidence": 0.0,
                "gat_mode": "error",
                "high_conf_edges": [],
                "low_conf_edges": [],
                "missing_edges": [],
            },
        )
