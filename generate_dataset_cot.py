#!/usr/bin/env python3
"""
generate_dataset_cot.py — 带坐标映射思维链的 SFT 数据集生成器 v3
核心改进：
  1. 在 assistant 回复中注入 <think>…</think> 推理链
     让模型学会「中文路名 → 边ID → 权重」的显式映射过程
  2. 兼容 Qwen 和 DeepSeek-R1（统一 chat_template 格式）
  3. 保留原有的空间相关性、时段偏移、事件注入逻辑
  4. 数据量扩充到 5000/500
  5. 内置验证：保存前自动校验 98 条全覆盖
"""

import random, os, json, re
from pathlib import Path
from copy import deepcopy
import pyarrow as pa
import pyarrow.parquet as pq

# ── 超参 ────────────────────────────────────────────────────
TRAIN_SIZE  = 5000
EVAL_SIZE   = 500
OUT_DIR     = "/root/autodl-tmp/dataset_cot"
RANDOM_SEED = 42
MAX_RETRY   = 10
random.seed(RANDOM_SEED)

# ── 路网定义 ────────────────────────────────────────────────
COL_NAMES = ["经一路", "经三路", "经六路", "经八路", "花园路", "未来路"]
ROW_NAMES = ["农业路", "红专路", "政七街", "黄河路", "纬五路"]
COL_TYPE  = ['M', 'S', 'A', 'S', 'A', 'M']
ROW_TYPE  = ['A', 'S', 'M', 'A', 'M']

DIR_CN  = {'E': '向东', 'W': '向西', 'N': '向北', 'S': '向南'}
DIR_OPP = {'E': 'W', 'W': 'E', 'N': 'S', 'S': 'N'}

# 中文路名 → 坐标索引（用于生成 think 推理链）
ROW_IDX = {name: i for i, name in enumerate(ROW_NAMES)}
COL_IDX = {name: i for i, name in enumerate(COL_NAMES)}

# ── 路段注册 ────────────────────────────────────────────────
EDGES: list[str] = []
EDGE_META: dict  = {}

for r in range(len(ROW_NAMES)):
    for c in range(len(COL_NAMES) - 1):
        for d in ['E', 'W']:
            eid = f"R{r}C{c}_{d}"
            EDGES.append(eid)
            EDGE_META[eid] = {
                'road': ROW_NAMES[r], 'seg': f"{COL_NAMES[c]}至{COL_NAMES[c+1]}段",
                'dir': DIR_CN[d], 'rtype': ROW_TYPE[r],
                'row': r, 'col': c, 'axis': 'H',
                'opp': f"R{r}C{c}_{DIR_OPP[d]}",
            }

for c in range(len(COL_NAMES)):
    for r in range(len(ROW_NAMES) - 1):
        for d in ['N', 'S']:
            eid = f"C{c}R{r}_{d}"
            EDGES.append(eid)
            EDGE_META[eid] = {
                'road': COL_NAMES[c], 'seg': f"{ROW_NAMES[r]}至{ROW_NAMES[r+1]}段",
                'dir': DIR_CN[d], 'rtype': COL_TYPE[c],
                'row': r, 'col': c, 'axis': 'V',
                'opp': f"C{c}R{r}_{DIR_OPP[d]}",
            }

assert len(EDGES) == 98, f"路段数量错误: {len(EDGES)} ≠ 98"

# ── 邻居索引（空间相关性）──────────────────────────────────
def _build_neighbors():
    nb = {e: [] for e in EDGES}
    for eid in EDGES:
        m = EDGE_META[eid]
        r, c = m['row'], m['col']
        if m['axis'] == 'H':
            d = 'E' if eid.endswith('_E') else 'W'
            for dc in [-1, 1]:
                nc = c + dc
                if 0 <= nc <= len(COL_NAMES) - 2:
                    nb[eid].append(f"R{r}C{nc}_{d}")
        else:
            d = 'N' if eid.endswith('_N') else 'S'
            for dr in [-1, 1]:
                nr = r + dr
                if 0 <= nr <= len(ROW_NAMES) - 2:
                    nb[eid].append(f"C{c}R{nr}_{d}")
    return nb

NEIGHBORS = _build_neighbors()

# ── 拥堵等级与权重范围 ──────────────────────────────────────
LEVELS = {
    "畅通":     (0.3,  1.2),
    "正常":     (1.5,  2.5),
    "轻微拥堵": (3.0,  4.5),
    "中度拥堵": (5.0,  6.5),
    "严重拥堵": (7.0,  9.0),
    "极度拥堵": (9.0, 10.0),
    "封闭":     (9.5, 10.0),
}
LEVEL_KEYS = list(LEVELS.keys())
LEVEL_IDX  = {k: i for i, k in enumerate(LEVEL_KEYS)}

# 拥堵等级 → 典型权重（用于 think 链描述，避免小数精度问题）
LEVEL_WEIGHT_HINT = {
    "畅通": 0.8, "正常": 2.0, "轻微拥堵": 3.8,
    "中度拥堵": 5.5, "严重拥堵": 8.0, "极度拥堵": 9.5, "封闭": 9.8,
}

BASE_PROB = {
    'A': [0.15, 0.22, 0.22, 0.20, 0.12, 0.06, 0.03],
    'S': [0.20, 0.27, 0.22, 0.16, 0.09, 0.04, 0.02],
    'M': [0.28, 0.30, 0.20, 0.13, 0.06, 0.02, 0.01],
}

TIME_SLOTS = {
    "早高峰(7-9时)":   {'A': [0, -1,  0,  1,  1,  0, 0]},
    "上午(9-11时)":    {'A': [0,  0,  0,  0,  0,  0, 0]},
    "午间(11-13时)":   {'A': [1,  0, -1,  0,  0,  0, 0]},
    "下午(14-17时)":   {'A': [1,  0,  0,  0, -1,  0, 0]},
    "晚高峰(17-19时)": {'A': [0, -2,  0,  1,  2,  0, 0]},
    "夜间(19-23时)":   {'A': [2,  1, -1, -1, -1,  0, 0]},
}

EVENTS = {
    "封闭":     ["施工全封闭", "重大活动封路管制", "道路临时封闭"],
    "极度拥堵": ["多车连环追尾严重拥堵", "大型展会散场极度拥堵"],
    "严重拥堵": ["发生两车追尾事故", "早高峰严重拥堵", "信号灯故障严重拥堵"],
    "中度拥堵": ["中度拥堵行驶缓慢", "车流量较大", "路口排队较长"],
    "轻微拥堵": ["轻微拥堵", "车流略有缓慢"],
    "正常":     ["正常通行", "车流稳定"],
    "畅通":     ["畅通无阻", "车流顺畅"],
}

ENDINGS = [
    "请为全部98条路段生成权重（0-10，越大越拥堵）。",
    "请按「路段ID:权重值」格式输出所有98条路段权重。",
    "请为郑州市金水区路网的98条路段分配拥堵权重，越大越拥堵。",
    "请严格按格式输出各路段权重（路段ID:数值），用于自动驾驶路径规划。",
    "基于上述路况，为全路网98条路段输出权重（0-10），请勿遗漏任何路段。",
    # 带路网索引提示的 Ending，强化坐标映射
    ("路网索引：行R0-R4对应农业路至纬五路，列C0-C5对应经一路至未来路。"
     "请为全部98条路段按「路段ID:权重」格式输出权重（0-10）。"),
    ("其中：黄河路=R3，花园路=C4，经六路=C2，农业路=R0，经三路=C1。"
     "请严格按路段ID输出所有98条权重，格式：路段ID:数值。"),
]


# ── 权重采样 ────────────────────────────────────────────────
def sample_weight(level: str) -> float:
    lo, hi = LEVELS[level]
    x = random.betavariate(2.0, 2.0)
    return round(lo + x * (hi - lo), 1)


# ── 等级分配（带空间相关性+时段）──────────────────────────
def assign_levels(time_slot=None) -> dict:
    lvs = {}
    for eid in EDGES:
        rt = EDGE_META[eid]['rtype']
        probs = list(BASE_PROB[rt])
        if time_slot and time_slot in TIME_SLOTS:
            adj = TIME_SLOTS[time_slot].get('A', [0]*7)
            probs = [max(0.001, p + a*0.02) for p, a in zip(probs, adj)]
            s = sum(probs)
            probs = [p/s for p in probs]
        lvs[eid] = random.choices(LEVEL_KEYS, weights=probs, k=1)[0]

    # 空间扩散
    high = {"严重拥堵", "极度拥堵", "封闭"}
    for _ in range(2):
        updates = {}
        for eid in EDGES:
            if lvs[eid] in high:
                for nb in NEIGHBORS.get(eid, []):
                    if nb in EDGES and lvs[nb] not in high:
                        ci = LEVEL_IDX[lvs[nb]]
                        si = LEVEL_IDX[lvs[eid]]
                        if random.random() < 0.40 and ci < si - 1:
                            updates[nb] = LEVEL_KEYS[ci + 1]
        lvs.update(updates)

    # 对向相关性
    for eid in EDGES:
        opp = EDGE_META[eid].get('opp')
        if opp and opp in lvs:
            ci = LEVEL_IDX[lvs[eid]]
            oi = LEVEL_IDX[lvs[opp]]
            if abs(ci - oi) > 2:
                lvs[opp] = LEVEL_KEYS[ci - 1] if ci > oi else LEVEL_KEYS[ci + 1]
    return lvs


def inject_event(lvs: dict) -> tuple[dict, str, str]:
    """注入重大事件，返回 (updated_lvs, event_desc, event_level)"""
    lvs = deepcopy(lvs)
    arterials = [e for e in EDGES if EDGE_META[e]['rtype'] == 'A']
    center = random.choice(arterials)
    event_level = random.choice(["封闭", "极度拥堵", "严重拥堵"])
    lvs[center] = event_level
    for nb in NEIGHBORS.get(center, []):
        if nb in lvs:
            ci = LEVEL_IDX[lvs[nb]]
            new_i = min(len(LEVEL_KEYS)-1, ci + random.randint(1, 2))
            lvs[nb] = LEVEL_KEYS[new_i]
    m = EDGE_META[center]
    evt_desc = f"{m['road']}{m['seg']}{m['dir']}{random.choice(EVENTS[event_level])}"
    return lvs, evt_desc, event_level


# ── Think 链生成（4种句式模板，随机选取防止过拟合）──────────
def _fmt_event(tpl_id: int, idx: int, eid: str, level: str,
               evt_desc: str, w: float, weights: dict,
               seen_eids: set) -> list[str]:
    """
    返回单个事件的推理行（list，可能 1-2 行）。
    tpl_id 0-3 对应 4 种不同句式，由调用方随机决定。
    """
    m = EDGE_META[eid]
    dir_code = {'向东': 'E', '向西': 'W', '向北': 'N', '向南': 'S'}[m['dir']]
    if m['axis'] == 'H':
        rid, cid = m['row'], m['col']
        eid_label = f"R{rid}C{cid}_{dir_code}"
    else:
        rid, cid = m['row'], m['col']
        eid_label = f"C{cid}R{rid}_{dir_code}"

    road, seg, dirn = m['road'], m['seg'], m['dir']

    if tpl_id == 0:
        # 标准"查表"句式
        if m['axis'] == 'H':
            line = (f"事件{idx}：「{evt_desc}」→ "
                    f"道路 {road}=R{rid}，路段 {seg}→C{cid}，"
                    f"方向 {dirn}→{dir_code}，"
                    f"路段ID={eid_label}，等级={level}，权重={w}。")
        else:
            line = (f"事件{idx}：「{evt_desc}」→ "
                    f"道路 {road}=C{cid}，路段 {seg}→R{rid}，"
                    f"方向 {dirn}→{dir_code}，"
                    f"路段ID={eid_label}，等级={level}，权重={w}。")

    elif tpl_id == 1:
        # "目标路段"句式
        line = (f"目标路段{idx}：{road}（{'R' if m['axis']=='H' else 'C'}"
                f"{rid if m['axis']=='H' else cid}）"
                f"{dirn}方向，状态={level}，"
                f"分配 {eid_label}:权重 {w}。")

    elif tpl_id == 2:
        # "分析事件"句式
        line = (f"分析事件{idx}：{evt_desc}。"
                f"查表：{road}→{'R'+str(rid) if m['axis']=='H' else 'C'+str(cid)}，"
                f"区段→{'C'+str(cid) if m['axis']=='H' else 'R'+str(rid)}，"
                f"{dirn}→{dir_code}。"
                f"路段 {eid_label} 权重={w}（{level}）。")

    else:
        # "简洁推断"句式
        coord_str = (f"R{rid}C{cid}" if m['axis'] == 'H' else f"C{cid}R{rid}")
        line = (f"Step{idx}: {evt_desc} ⟹ "
                f"{road}={'R'+str(rid) if m['axis']=='H' else 'C'+str(cid)}, "
                f"seg={'C'+str(cid) if m['axis']=='H' else 'R'+str(rid)}, "
                f"dir={dir_code} ⟹ {eid_label}={w}")

    result = [line]
    seen_eids.add(eid)

    # 邻近传播注记（最多 2 个，同样随机句式）
    nb_affected = [
        nb for nb in NEIGHBORS.get(eid, [])
        if nb in weights and weights[nb] > 4.0 and nb not in seen_eids
    ][:2]
    if nb_affected:
        nb_strs = [f"{nb}:{weights[nb]}" for nb in nb_affected]
        if tpl_id in (0, 1):
            result.append(f"  ↳ 拥堵传播至邻近路段：{', '.join(nb_strs)}")
        elif tpl_id == 2:
            result.append(f"  连带影响：{', '.join(nb_strs)}")
        else:
            result.append(f"  propagate→ {', '.join(nb_strs)}")
        seen_eids.update(nb_affected)

    return result


# 收尾语（4种，与句式模板一一对应）
_THINK_CLOSINGS = [
    ("其余未提及路段按道路等级分配默认权重（畅通/正常区间 0.3–2.5）。",
     "按 路段ID:权重 格式输出全部98条路段权重，不得遗漏。"),
    ("未描述路段依据道路类型赋予正常权重（1.2–2.5）。",
     "输出所有98条路段权重，格式：路段ID:数值。"),
    ("其他路段路况正常，权重维持基础值（≤2.5）。",
     "完整输出98条权重键值对，不可缺漏任何路段。"),
    ("remaining edges: default normal weight (1.2-2.5).",
     "Output all 98 edge weights as EdgeID:value pairs."),
]

# 开头语（4种）
_THINK_HEADERS = [
    "【坐标映射与权重推导】",
    "【路段ID查表过程】",
    "【交通事件解析】",
    "## Coordinate Mapping",
]


def build_think_chain(described_events: list[tuple], weights: dict) -> str:
    """
    described_events: list of (eid, level, event_desc)
    随机选取 4 种句式模板之一，防止训练过拟合单一格式。
    返回完整的 <think>…</think> 文本（不含换行后的权重部分）。
    """
    tpl_id = random.randint(0, 3)   # 整条 think 链统一使用一种模板风格
    seen_eids: set[str] = set()

    lines = ["<think>", _THINK_HEADERS[tpl_id]]

    for idx, (eid, level, evt_desc) in enumerate(described_events, 1):
        w = weights[eid]
        lines.extend(_fmt_event(tpl_id, idx, eid, level, evt_desc, w,
                                weights, seen_eids))

    closing = _THINK_CLOSINGS[tpl_id]
    lines.append(closing[0])
    lines.append(closing[1])
    lines.append("</think>")
    return "\n".join(lines)


# ── 样本生成 ────────────────────────────────────────────────
def generate_sample(force_event: bool = False) -> dict:
    time_slot = random.choice(list(TIME_SLOTS.keys()) + [None, None])
    lvs = assign_levels(time_slot)

    injected_event_info = None
    if force_event or random.random() < 0.35:
        lvs, evt_desc, evt_level = inject_event(lvs)
        injected_event_info = (evt_desc, evt_level)

    weights = {e: sample_weight(lvs[e]) for e in EDGES}

    # 选取要描述的路段（偏向主干道+极端状态）
    major   = [e for e in EDGES if EDGE_META[e]['rtype'] == 'A']
    extreme = [e for e in EDGES if lvs[e] in ("封闭", "极度拥堵", "畅通")]
    routine = [e for e in EDGES if e not in major and e not in extreme]

    n_major   = min(random.randint(2, 5), len(major))
    n_extreme = min(random.randint(1, 4), len(extreme))
    n_routine = min(random.randint(1, 3), len(routine))

    pool = list(set(
        random.sample(major, n_major) +
        random.sample(extreme, min(n_extreme, len(extreme))) +
        random.sample(routine, min(n_routine, len(routine)))
    ))
    random.shuffle(pool)
    described_eids = pool[:random.randint(6, 14)]

    # 构建描述文字和事件元数据
    parts = []
    described_events = []  # (eid, level, desc_text)
    for eid in described_eids:
        lv = lvs[eid]
        m  = EDGE_META[eid]
        evt = random.choice(EVENTS[lv])
        txt = f"{m['road']}{m['seg']}{m['dir']}{evt}"
        parts.append(txt)
        described_events.append((eid, lv, txt))

    time_prefix = f"【{time_slot}】" if time_slot else ""
    instruction = (
        time_prefix +
        "；".join(parts) +
        "。" +
        random.choice(ENDINGS)
    )

    # 构建 <think> 推理链
    think_block = build_think_chain(described_events, weights)

    # 构建权重输出（顺序与 EDGES 一致）
    weight_str = ", ".join(f"{e}:{weights[e]}" for e in EDGES)

    response = think_block + "\n" + weight_str

    return {
        "messages": [
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


# ── 验证 ────────────────────────────────────────────────────
def validate_sample(sample: dict) -> tuple[dict, list]:
    resp = sample["messages"][1]["content"]
    # 从 </think> 之后提取，与 app_fixed.py 推理端逻辑一致
    if "</think>" in resp:
        resp = resp.split("</think>")[-1]
    parsed = {}
    for eid, w in re.findall(r'([\w][\w\d]*)\s*:\s*(\d+(?:\.\d+)?)', resp):
        parsed[eid] = float(w)
    missing = [e for e in EDGES if e not in parsed]
    return parsed, missing


# ── Token 长度预估（防超出 2048）───────────────────────────
def estimate_tokens(sample: dict) -> int:
    total_chars = (len(sample["messages"][0]["content"]) +
                   len(sample["messages"][1]["content"]))
    return total_chars // 2  # 中文约 1.5-2 chars/token，取保守值


# ── 保存 ────────────────────────────────────────────────────
def save_parquet(samples: list, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    msgs  = [json.dumps(s["messages"], ensure_ascii=False) for s in samples]
    table = pa.table({"messages": pa.array(msgs, type=pa.string())})
    out   = os.path.join(out_dir, "data.parquet")
    pq.write_table(table, out)
    size_mb = os.path.getsize(out) / 1024**2
    print(f"  ✅ 已保存 {len(samples)} 条 → {out}  ({size_mb:.1f} MB)")


# ── 批量生成 ────────────────────────────────────────────────
def generate_dataset(n: int, label: str) -> list:
    samples   = []
    fail_cnt  = 0
    retry_tot = 0
    i = 0

    while len(samples) < n:
        force_event = (i % 3 == 0)
        sample = generate_sample(force_event=force_event)

        _, missing = validate_sample(sample)
        token_est  = estimate_tokens(sample)

        if missing:
            fail_cnt  += 1
            retry_tot += 1
            if retry_tot > MAX_RETRY * n:
                # 极端情况下强行加入（理论上不应发生）
                samples.append(sample)
                i += 1
            continue

        if token_est > 1800:
            # token 过长跳过，防止截断影响训练
            fail_cnt += 1
            continue

        samples.append(sample)
        retry_tot = 0
        i += 1

        if len(samples) % 500 == 0:
            print(f"  [{label}] {len(samples)}/{n}  跳过: {fail_cnt} 次")

    return samples


def print_stats(samples: list, label: str):
    all_weights, desc_lens, token_ests = [], [], []
    for s in samples:
        parsed, _ = validate_sample(s)
        all_weights.extend(parsed.values())
        desc_lens.append(len(s["messages"][0]["content"]))
        token_ests.append(estimate_tokens(s))

    avg_w = sum(all_weights) / len(all_weights) if all_weights else 0
    avg_t = sum(token_ests)  / len(token_ests)  if token_ests  else 0
    over  = sum(1 for t in token_ests if t > 1800)
    w_dist = {
        "畅通(<2)":  sum(1 for w in all_weights if w < 2),
        "正常(2-4)": sum(1 for w in all_weights if 2 <= w < 4),
        "拥堵(4-7)": sum(1 for w in all_weights if 4 <= w < 7),
        "严重(7+)":  sum(1 for w in all_weights if w >= 7),
    }
    total = len(all_weights) or 1
    print(f"\n  [{label}] 统计:")
    print(f"    样本数:      {len(samples)}")
    print(f"    平均权重:    {avg_w:.2f}")
    print(f"    估算 token:  均值 {avg_t:.0f}  超 1800 的: {over} 条")
    print(f"    权重分布:  " + " | ".join(f"{k}: {v/total*100:.1f}%" for k, v in w_dist.items()))


def main():
    print("=" * 65)
    print("  CoT 数据集生成器 v3 — 郑州金水区 6×5 路网 (98条路段)")
    print(f"  训练集: {TRAIN_SIZE}  验证集: {EVAL_SIZE}")
    print("=" * 65)

    print(f"\n📦 生成训练集 ({TRAIN_SIZE} 条)...")
    train = generate_dataset(TRAIN_SIZE, "TRAIN")
    save_parquet(train, f"{OUT_DIR}/train")
    print_stats(train, "训练集")

    random.seed(RANDOM_SEED + 999)
    print(f"\n📦 生成验证集 ({EVAL_SIZE} 条)...")
    ev = generate_dataset(EVAL_SIZE, "EVAL")
    save_parquet(ev, f"{OUT_DIR}/eval")
    print_stats(ev, "验证集")

    # 详细验证前3条
    print("\n🔍 样本预览（第1条）:")
    s = train[0]
    print(f"  用户: {s['messages'][0]['content'][:100]}...")
    assistant_preview = s['messages'][1]['content'][:300]
    print(f"  模型: {assistant_preview}...")
    parsed, missing = validate_sample(s)
    print(f"  解析: {len(parsed)}/98 条权重  缺失: {len(missing)}")

    print("\n" + "=" * 65)
    print("✅ CoT 数据集生成完成！")
    print(f"  训练集 → {OUT_DIR}/train/data.parquet")
    print(f"  验证集 → {OUT_DIR}/eval/data.parquet")
    print(f"\n🚀 下一步: python run_sft_lora.py --config sft_config_lora_qwen.yaml")
    print("=" * 65)


if __name__ == "__main__":
    main()
