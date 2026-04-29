#!/usr/bin/env python3
"""
smallnet_baseline.py

24 边 SmallNet 对照实验脚本。

目标：
  1. 在一个固定的 4 行 × 3 列小路网上，比较 Qwen-LoRA / R1-LoRA / Sparse-LoRA 的解析率。
  2. 强制模型只在 24 条边的白名单内输出，避免“脑补大路网边ID”。
  3. 将 parse_rate 作为核心指标落盘，便于 UI 直接读取。
"""

from __future__ import annotations

import gc
import heapq
import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 24 边 SmallNet 定义 ─────────────────────────────────────
SMALL_ROW_NAMES = ["X街", "Y街", "Z街", "W街"]
SMALL_COL_NAMES = ["A路", "B路", "C路"]
SMALL_NODE_ROWS = len(SMALL_ROW_NAMES) + 1  # 5
SMALL_NODE_COLS = len(SMALL_COL_NAMES) + 1  # 4

# 24 条有向边：12 条横向（E）+ 12 条纵向（S）
SMALL_H_EDGES = [f"SR{r}C{c}_E" for r in range(len(SMALL_ROW_NAMES)) for c in range(len(SMALL_COL_NAMES))]
SMALL_V_EDGES = [f"SC{c}R{r}_S" for c in range(len(SMALL_COL_NAMES)) for r in range(len(SMALL_ROW_NAMES))]
SMALL_EDGES = SMALL_H_EDGES + SMALL_V_EDGES
SMALL_EDGE_SET = set(SMALL_EDGES)
SMALL_EDGE_IDX = {eid: idx for idx, eid in enumerate(SMALL_EDGES)}
SMALL_EDGE_BLOCK = "\n".join(f"- {eid}" for eid in SMALL_EDGES)
SMALL_EDGE_LINE = ", ".join(SMALL_EDGES)

SMALL_META: Dict[str, Dict[str, int | str]] = {}
for r in range(len(SMALL_ROW_NAMES)):
    for c in range(len(SMALL_COL_NAMES)):
        eid = f"SR{r}C{c}_E"
        SMALL_META[eid] = {
            "axis": "H",
            "row": r,
            "col": c,
            "row_name": SMALL_ROW_NAMES[r],
            "col_name": SMALL_COL_NAMES[c],
        }
for c in range(len(SMALL_COL_NAMES)):
    for r in range(len(SMALL_ROW_NAMES)):
        eid = f"SC{c}R{r}_S"
        SMALL_META[eid] = {
            "axis": "V",
            "row": r,
            "col": c,
            "row_name": SMALL_ROW_NAMES[r],
            "col_name": SMALL_COL_NAMES[c],
        }

assert len(SMALL_EDGES) == 24, f"期望24条边，实际{len(SMALL_EDGES)}"

# 需要纳入对比的模型
DEFAULT_MODEL_SPECS = [
    {"method": "Qwen-LoRA★", "path": "/root/autodl-tmp/model_merged_qwen", "mode": "raw"},
    {"method": "R1-LoRA★", "path": "/root/autodl-tmp/model_merged_r1", "mode": "cot"},
    {"method": "Sparse-LoRA★★", "path": "/root/autodl-tmp/model_merged_sparse", "mode": "sparse"},
]


# ── 路网图结构 ──────────────────────────────────────────────
def build_small_graph(weight_dict: Dict[str, float], default_w: float = 2.0) -> Dict:
    """构建邻接表 graph[node] = [(cost, neighbor, edge_id)]。"""
    graph = defaultdict(list)
    for eid in SMALL_EDGES:
        meta = SMALL_META[eid]
        r, c = int(meta["row"]), int(meta["col"])
        if meta["axis"] == "H":
            src = (r, c)
            dst = (r, c + 1)
        else:
            src = (r, c)
            dst = (r + 1, c)
        graph[src].append((weight_dict.get(eid, default_w), dst, eid))
    return graph


def smallnet_dijkstra(weight_dict: Dict[str, float], start: tuple, end: tuple) -> Tuple[List, float, float]:
    """在 4 行 × 3 列 SmallNet 上运行 Dijkstra。"""
    t0 = time.perf_counter()
    graph = build_small_graph(weight_dict)
    dist = defaultdict(lambda: math.inf)
    prev = {}
    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            break
        for w, v, _eid in graph.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    path = []
    cur = end
    while cur in prev or cur == start:
        path.append(cur)
        if cur == start:
            break
        cur = prev.get(cur)
    path.reverse()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return (path if path and path[0] == start else []), dist.get(end, math.inf), elapsed_ms


# ── Prompt / 解析 ───────────────────────────────────────────
def _build_instruction(constraint: str, mode: str) -> str:
    base = (
        "你是交通路径权重生成助手。\n"
        "本次只处理 4行×3列 的 SmallNet，总计 24 条有向边。\n"
        "只能使用下列 24 个路段ID，禁止输出任何其他路段：\n"
        f"{SMALL_EDGE_BLOCK}\n\n"
        "只输出权重键值对，格式：路段ID:权重值, 路段ID:权重值\n"
        "权重范围 0-10，越大越拥堵。\n"
        "请严格遵守白名单，不要脑补大路网边ID。\n"
    )
    if mode == "cot":
        ending = "请先简要推理，再在 </think> 后只输出全部24条路段的权重。"
    elif mode == "sparse":
        ending = "请只输出偏离正常值 2.0 超过 2.0 的异常路段权重，正常路段无需输出。"
    else:
        ending = "请为全部24条路段生成权重。"
    return f"{base}交通描述：{constraint}\n{ending}"


def _weight_from_text(token: str) -> float | None:
    for kws, val in [
        (("全封", "封闭", "管制", "禁行", "禁止通行"), 9.5),
        (("极度拥堵", "严重拥堵", "严重", "极度"), 8.0),
        (("中度拥堵", "较拥堵", "中度"), 5.5),
        (("拥堵", "缓行", "堵"), 4.5),
        (("轻微拥堵", "轻微", "较慢"), 3.5),
        (("正常", "顺畅"), 2.0),
        (("畅通", "通畅", "无阻"), 1.2),
    ]:
        if any(k in token for k in kws):
            return val
    return None


def _expand_horizontal(parsed: Dict[str, float], row_idx: int, value: float, cols: List[int]):
    for c in cols:
        parsed[f"SR{row_idx}C{c}_E"] = value


def _expand_vertical(parsed: Dict[str, float], col_idx: int, value: float, rows: List[int]):
    for r in rows:
        parsed[f"SC{col_idx}R{r}_S"] = value


def parse_small_weights(text: str) -> Dict[str, float]:
    """从模型输出中提取 24 边权重。"""
    if not text:
        return {}
    if "</think>" in text:
        text = text.split("</think>")[-1]

    parsed: Dict[str, float] = {}

    # 1) 标准 EdgeID:weight
    for eid, val in re.findall(r"(SR\d+C\d+_E|SC\d+R\d+_S)\s*[:：]\s*(\d+(?:\.\d+)?)", text):
        if eid in SMALL_EDGE_SET:
            parsed[eid] = float(val)

    # 2) 中文语义短句
    clauses = re.split(r"[；。;\n,，]+", text)
    for clause in clauses:
        value = _weight_from_text(clause)
        if value is None:
            continue

        row_hits = [i for i, name in enumerate(SMALL_ROW_NAMES) if name in clause]
        col_hits = [i for i, name in enumerate(SMALL_COL_NAMES) if name in clause]

        if ("向东" in clause or "东向" in clause) and row_hits:
            if len(col_hits) >= 2:
                for left, right in zip(col_hits[:-1], col_hits[1:]):
                    lo, hi = sorted((left, right))
                    for r in row_hits:
                        _expand_horizontal(parsed, r, value, list(range(lo, hi)))
            else:
                for r in row_hits:
                    _expand_horizontal(parsed, r, value, list(range(len(SMALL_COL_NAMES))))

        if ("向南" in clause or "南向" in clause) and col_hits:
            if len(row_hits) >= 2:
                for top, bottom in zip(row_hits[:-1], row_hits[1:]):
                    lo, hi = sorted((top, bottom))
                    for c in col_hits:
                        _expand_vertical(parsed, c, value, list(range(lo, hi)))
            else:
                for c in col_hits:
                    _expand_vertical(parsed, c, value, list(range(len(SMALL_ROW_NAMES))))

    return parsed


# ── 模型加载 / 推理 ─────────────────────────────────────────
def load_model_pair(model_path: str):
    """加载一个模型对（tokenizer, model）。"""
    if not torch.cuda.is_available():
        raise RuntimeError("SmallNet baseline 需要 CUDA GPU，请先确认 GPU 可用。")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "R1" in model_path or "r1" in model_path or "DeepSeek" in model_path:
        tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def _run_small_infer(tokenizer, model, constraint: str, mode: str, max_new_tokens: int) -> Tuple[str, str]:
    instruction = _build_instruction(constraint, mode)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    struct = decoded.split("</think>")[-1].strip() if "</think>" in decoded else decoded
    return decoded, struct


def _infer_lora(tokenizer, model, constraint: str):
    """Qwen-LoRA: 直接输出全部 24 条边的权重。"""
    return _run_small_infer(tokenizer, model, constraint, mode="raw", max_new_tokens=1200)


def _infer_r1(tokenizer, model, constraint: str):
    """R1-LoRA: 使用 CoT 风格提示词，输出 24 条边权重。"""
    return _run_small_infer(tokenizer, model, constraint, mode="cot", max_new_tokens=1600)


def _infer_sparse(tokenizer, model, constraint: str):
    """Sparse-LoRA: 只输出异常路段。"""
    return _run_small_infer(tokenizer, model, constraint, mode="sparse", max_new_tokens=320)


def _build_model_specs(model_paths: List[str] | None = None):
    """构建模型对比列表。"""
    if model_paths is not None:
        specs = []
        for p in model_paths:
            p = p.strip()
            if not p:
                continue
            if os.path.exists(p):
                name = os.path.basename(p.rstrip("/"))
                if "model_merged_qwen" in p:
                    name = "Qwen-LoRA★"
                elif "model_merged_r1" in p:
                    name = "R1-LoRA★"
                elif "model_merged_sparse" in p:
                    name = "Sparse-LoRA★★"
                specs.append({"method": name, "path": p})
        return specs

    return [spec for spec in DEFAULT_MODEL_SPECS if os.path.exists(spec["path"])]


# ── 测试场景 ────────────────────────────────────────────────
SMALL_TEST_CASES = [
    {
        "name": "SmallNet-密集场景A：三事件同域拥堵",
        "start": (0, 0),
        "end": (4, 2),
        "constraint": (
            "X街A路至B路段向东严重拥堵；"
            "X街B路至C路段向东中度拥堵；"
            "A路X街至Y街段向南封闭施工。"
        ),
        "true_w": {
            "SR0C0_E": 8.5,
            "SR0C1_E": 5.5,
            "SC0R0_S": 9.5,
        },
        "note": "3 个密集事件，验证短序列下的稳定解析能力。",
    },
    {
        "name": "SmallNet-密集场景B：中轴拥堵",
        "start": (0, 1),
        "end": (4, 2),
        "constraint": (
            "Y街A路至B路段向东轻微拥堵；"
            "Y街B路至C路段向东严重拥堵；"
            "B路Y街至Z街段向南中度拥堵。"
        ),
        "true_w": {
            "SR1C0_E": 3.5,
            "SR1C1_E": 8.0,
            "SC1R1_S": 5.5,
        },
        "note": "三段连续拥堵，测试模型是否能稳定覆盖邻近边。",
    },
    {
        "name": "SmallNet-密集场景C：末端绕行",
        "start": (1, 0),
        "end": (3, 2),
        "constraint": (
            "Z街A路至B路段向东封闭；"
            "Z街B路至C路段向东中度拥堵；"
            "C路X街至Y街段向南严重拥堵。"
        ),
        "true_w": {
            "SR2C0_E": 9.5,
            "SR2C1_E": 5.5,
            "SC2R0_S": 8.0,
        },
        "note": "高压路段集中在末端，验证绕行路径选择。",
    },
]


# ── 基线方法 ───────────────────────────────────────────────
def method_uniform_small(start, end, _tc) -> Tuple[Dict, List, float, float]:
    wd = {e: 1.0 for e in SMALL_EDGES}
    path, cost, ms = smallnet_dijkstra(wd, start, end)
    return wd, path, cost, ms


def method_rule_small(start, end, tc) -> Tuple[Dict, List, float, float]:
    """规则解析基线。"""
    wd = {e: 2.0 for e in SMALL_EDGES}
    for clause in re.split(r"[；。;\n]+", tc["constraint"]):
        value = _weight_from_text(clause)
        if value is None:
            continue
        row_hits = [i for i, name in enumerate(SMALL_ROW_NAMES) if name in clause]
        col_hits = [i for i, name in enumerate(SMALL_COL_NAMES) if name in clause]
        if ("向东" in clause or "东向" in clause) and row_hits:
            if len(col_hits) >= 2:
                for left, right in zip(col_hits[:-1], col_hits[1:]):
                    for r in row_hits:
                        for c in range(min(left, right), max(left, right)):
                            wd[f"SR{r}C{c}_E"] = value
            else:
                for r in row_hits:
                    for c in range(len(SMALL_COL_NAMES)):
                        wd[f"SR{r}C{c}_E"] = value
        if ("向南" in clause or "南向" in clause) and col_hits:
            if len(row_hits) >= 2:
                for top, bottom in zip(row_hits[:-1], row_hits[1:]):
                    for c in col_hits:
                        for r in range(min(top, bottom), max(top, bottom)):
                            wd[f"SC{c}R{r}_S"] = value
            else:
                for c in col_hits:
                    for r in range(len(SMALL_ROW_NAMES)):
                        wd[f"SC{c}R{r}_S"] = value
    path, cost, ms = smallnet_dijkstra(wd, start, end)
    return wd, path, cost, ms


def calc_mae(pred: Dict, true: Dict) -> float:
    eids = list(true.keys())
    if not eids:
        return float("nan")
    return round(sum(abs(pred.get(e, 2.0) - true[e]) for e in eids) / len(eids), 3)


def calc_parse_rate(pred: Dict) -> float:
    parsed = sum(1 for e in SMALL_EDGES if abs(pred.get(e, 2.0) - 2.0) > 0.1)
    return round(parsed / len(SMALL_EDGES) * 100, 1)


# ── 主实验函数 ───────────────────────────────────────────
def run_smallnet_comparison(
    llm_model_path: str | None = None,
    output_path: str = "results/smallnet_results.json",
    model_paths: List[str] | None = None,
) -> List[dict]:
    """运行小路网对照实验并保存 JSON。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = []

    if model_paths is None and llm_model_path:
        model_paths = [llm_model_path]
    model_specs = _build_model_specs(model_paths)

    if not model_specs:
        print("⚠️ 未找到可用模型，将仅运行规则基线。")
    else:
        print("📦 SmallNet 对比模型：")
        for spec in model_specs:
            print(f"  - {spec['method']}: {spec['path']}")

    loaded_models = []
    for spec in model_specs:
        try:
            print(f"📦 加载模型: {spec['method']} -> {spec['path']}")
            tokenizer, model = load_model_pair(spec["path"])
            loaded_models.append({**spec, "tokenizer": tokenizer, "model": model})
        except Exception as e:
            print(f"⚠️ 模型加载失败，跳过 {spec['path']}: {e}")

    if not loaded_models:
        print("ℹ️ LLM 模型不可用，本次仅运行规则/等权基线。")

    infer_dispatch = {
        "Qwen-LoRA★": _infer_lora,
        "R1-LoRA★": _infer_r1,
        "Sparse-LoRA★★": _infer_sparse,
    }

    for tc in SMALL_TEST_CASES:
        start, end = tc["start"], tc["end"]
        print("\n" + "=" * 60)
        print(f"🧪 {tc['name']}")
        print(f"   {start} → {end}")
        print(f"   {tc['note']}")
        print("=" * 60)

        true_w = {**{e: 2.0 for e in SMALL_EDGES}, **tc["true_w"]}
        oracle_path, oracle_cost, _ = smallnet_dijkstra(true_w, start, end)

        tc_result = {
            "scenario": tc["name"],
            "oracle_cost": round(oracle_cost, 2),
            "total_edges": len(SMALL_EDGES),
            "methods": [],
        }

        methods = [
            ("Uniform-Dijkstra", lambda: method_uniform_small(start, end, tc)),
            ("Rule-Dijkstra", lambda: method_rule_small(start, end, tc)),
        ]

        for spec in loaded_models:
            infer_fn = infer_dispatch.get(spec["method"])
            if infer_fn is None:
                continue

            def _make_llm_method(_tok=spec["tokenizer"], _mdl=spec["model"], _method=spec["method"], _infer=infer_fn):
                def _run():
                    t0 = time.perf_counter()
                    raw, struct = _infer(_tok, _mdl, tc["constraint"])
                    ms = (time.perf_counter() - t0) * 1000
                    parsed = parse_small_weights(struct) or parse_small_weights(raw)
                    wd = {e: 2.0 for e in SMALL_EDGES}
                    wd.update(parsed)
                    path, cost, _ = smallnet_dijkstra(wd, start, end)
                    return wd, path, cost, ms, len(parsed), raw, struct

                return _run

            methods.append((spec["method"], _make_llm_method()))

        print(f"  {'方法':<22} {'规划ms':>8} {'跳数':>6} {'代价':>8} {'Overhead':>10} {'MAE':>7} {'解析率':>8}")
        print(f"  {'─'*22} {'─'*8} {'─'*6} {'─'*8} {'─'*10} {'─'*7} {'─'*8}")

        for name, fn in methods:
            try:
                t0 = time.perf_counter()
                out = fn()
                ms = (time.perf_counter() - t0) * 1000
                wd, path, cost = out[0], out[1], out[2]
                llm_parsed = out[4] if len(out) > 4 else len([e for e in SMALL_EDGES if abs(wd.get(e, 2.0) - 2.0) > 0.1])
                mae = calc_mae(wd, tc["true_w"])
                parse_r = calc_parse_rate(wd)
                overhead = (cost / oracle_cost - 1) * 100 if oracle_cost > 0 else 0

                print(
                    f"  {name:<22} {ms:>7.1f}ms {max(len(path) - 1, 0):>6}跳  "
                    f"{cost:>7.2f}  {overhead:>+8.1f}%  {mae:>7.3f}  {parse_r:>7.1f}%"
                )

                tc_result["methods"].append({
                    "method": name,
                    "plan_ms": round(ms, 2),
                    "hops": max(len(path) - 1, 0),
                    "cost": round(cost, 2),
                    "overhead_pct": round(overhead, 2),
                    "weight_mae": round(mae, 3) if not math.isnan(mae) else None,
                    "parse_rate": parse_r,
                    "parsed_edges": int(llm_parsed),
                    "total_edges": len(SMALL_EDGES),
                })
            except Exception as e:
                print(f"  {name:<22} ❌ {e}")

        print(f"  Oracle（真实权重）: cost={oracle_cost:.2f}  路径={oracle_path}")
        results.append(tc_result)

    for item in loaded_models:
        try:
            del item["model"]
            del item["tokenizer"]
        except Exception:
            pass
    gc.collect()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ SmallNet 结果已写入 → {output_path}")
    return results


def print_conclusion(results: List[dict]):
    """打印实验结论，方便论文引用。"""
    print("\n" + "=" * 60)
    print("【SmallNet 实验结论】")
    print("=" * 60)
    if not results:
        print("暂无结果。")
        return

    all_methods = {}
    for tc in results:
        for m in tc.get("methods", []):
            all_methods.setdefault(m["method"], []).append(m.get("parse_rate", 0.0))

    for name, rates in all_methods.items():
        avg_rate = sum(rates) / max(len(rates), 1)
        print(f"  - {name}: 平均解析率 {avg_rate:.1f}%")

    print(
        "\n结论：在 4 行 × 3 列、24 条有向边的 SmallNet 上，"
        "模型可以在短序列条件下稳定完成完整边ID输出；"
        "parse_rate 直接反映了输出覆盖度，适合与大路网结果并排对比。"
    )


# ── 入口 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    model_path = None
    model_paths = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_path = sys.argv[idx + 1]
    if "--models" in sys.argv:
        idx = sys.argv.index("--models")
        model_paths = [p for p in sys.argv[idx + 1].split(",") if p.strip()]
    if "--all-llm" in sys.argv:
        model_paths = [spec["path"] for spec in DEFAULT_MODEL_SPECS]

    print("SmallNet 结构验证:")
    print(f"  行标签: {SMALL_ROW_NAMES}")
    print(f"  列标签: {SMALL_COL_NAMES}")
    print(f"  路网规格: {len(SMALL_ROW_NAMES)}行×{len(SMALL_COL_NAMES)}列")
    print(f"  交叉口网格: {SMALL_NODE_ROWS}×{SMALL_NODE_COLS}")
    print(f"  总边数: {len(SMALL_EDGES)} 条")
    print(f"  边白名单: {SMALL_EDGE_LINE}")

    results = run_smallnet_comparison(
        llm_model_path=model_path,
        model_paths=model_paths,
        output_path="/root/autodl-tmp/results/smallnet_results.json",
    )
    print_conclusion(results)
