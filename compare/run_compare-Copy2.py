#!/usr/bin/env python3
"""
compare_methods.py
对比实验：Dijkstra / Rule-A* / DQN / GCN-Weight / Raw-Qwen / CoT-Qwen / SFT-Qwen(本文)
输出：results/compare_results.json + 可视化图表
"""

import sys, os, re, json, time, random, math
import numpy as np
sys.path.insert(0, "/root/autodl-tmp/code")

# ── 路网定义（与主系统一致）──────────────────────────────
import sumolib
NET_PATH = "/root/autodl-tmp/SUMO/net/my_net.net.xml"
net = sumolib.net.readNet(NET_PATH)
ALL_EDGES = [e.getID() for e in net.getEdges()]

# ── 测试场景 ──────────────────────────────────────────────
TEST_CASES = [
    {
        "name": "场景A 黄河路单向拥堵",
        "start": "R3C0_E", "end": "R3C4_E",
        "desc": "黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北畅通无阻。",
        "ground_truth": {  # 人工标注的合理权重（用于计算约束符合率）
            "R3C0_E": 8.5, "R3C1_E": 8.0, "R3C2_E": 7.5,
            "C4R0_N": 1.5, "C4R1_N": 1.5, "C4R2_N": 1.5,
        }
    },
    {
        "name": "场景B 花园路封闭绕行",
        "start": "C4R0_N", "end": "C4R3_N",
        "desc": "花园路农业路至红专路段向北施工封闭；经六路向北畅通无阻。",
        "ground_truth": {
            "C4R0_N": 9.5, "C4R1_N": 9.5,
            "C2R0_N": 1.2, "C2R1_N": 1.2, "C2R2_N": 1.2,
        }
    },
    {
        "name": "场景C 多路段复合拥堵",
        "start": "R0C0_E", "end": "R3C4_E",
        "desc": "经六路全线封闭施工；花园路向北严重拥堵；黄河路向东中度拥堵。",
        "ground_truth": {
            "C2R0_N": 9.5, "C2R1_N": 9.5, "C2R2_N": 9.5,
            "C4R0_N": 7.5, "C4R1_N": 7.5,
            "R3C0_E": 5.0, "R3C1_E": 5.0,
        }
    },
]

# ═══════════════════════════════════════════════════════
# 方法1：Dijkstra（统一权重=1）
# ═══════════════════════════════════════════════════════
def method_dijkstra(start, end, desc):
    weights = {e: 1.0 for e in ALL_EDGES}
    path, cost = astar_route(net, start, end, weights)
    return weights, path, cost

# ═══════════════════════════════════════════════════════
# 方法2：Rule-based A*（基于道路类型的规则权重）
# ═══════════════════════════════════════════════════════
ROAD_TYPE_WEIGHT = {
    "农业路": 3.5, "黄河路": 4.0,  # 主干道拥堵概率高
    "红专路": 2.5, "经三路": 2.5,  # 次干道
    "政七街": 1.8, "纬五路": 1.5,  # 支路较畅通
    "花园路": 3.8, "经六路": 3.5,
    "经八路": 2.0, "经一路": 1.5, "未来路": 1.5,
}
KEYWORD_RULES = {
    "封闭": 9.5, "施工": 8.0, "严重拥堵": 7.5, "极度拥堵": 9.0,
    "追尾": 7.0, "中度拥堵": 5.0, "拥堵": 4.5,
    "轻微": 3.0, "正常": 2.0, "畅通": 1.2,
}
ROAD_DIR_MAP = {
    "向东": "E", "向西": "W", "向北": "N", "向南": "S"
}
ROW_NAMES = {"农业路":"R0","红专路":"R1","政七街":"R2","黄河路":"R3","纬五路":"R4"}
COL_NAMES = {"经一路":"C0","经三路":"C1","经六路":"C2","经八路":"C3","花园路":"C4","未来路":"C5"}

def method_rule_astar(start, end, desc):
    """规则驱动：解析描述文字，按关键词分配权重"""
    weights = {}
    for eid in ALL_EDGES:
        # 默认按道路类型给基础权重
        base = 2.0
        for rname, rkey in ROW_NAMES.items():
            if eid.startswith(rkey):
                base = ROAD_TYPE_WEIGHT.get(rname, 2.0)
                break
        for cname, ckey in COL_NAMES.items():
            if eid.startswith(ckey):
                base = ROAD_TYPE_WEIGHT.get(cname, 2.0)
                break
        weights[eid] = base

    # 按描述中的关键词覆盖相关路段
    sentences = re.split("[；。，]", desc)
    for sent in sentences:
        matched_weight = None
        for kw, w in sorted(KEYWORD_RULES.items(), key=lambda x: -x[1]):
            if kw in sent:
                matched_weight = w
                break
        if matched_weight is None:
            continue
        # 找匹配的道路和方向
        for rname, rkey in ROW_NAMES.items():
            if rname in sent:
                for cname, ckey in COL_NAMES.items():
                    if cname in sent:
                        for d in ["E","W","N","S"]:
                            eid = f"{rkey}C0_{d}" if rkey.startswith("R") else f"{ckey}R0_{d}"
                        pass
                for d, dkey in ROAD_DIR_MAP.items():
                    if d in sent:
                        for eid in [e for e in ALL_EDGES if e.startswith(rkey) and e.endswith(f"_{dkey}")]:
                            weights[eid] = matched_weight
        for cname, ckey in COL_NAMES.items():
            if cname in sent:
                for d, dkey in ROAD_DIR_MAP.items():
                    if d in sent:
                        for eid in [e for e in ALL_EDGES if e.startswith(ckey) and e.endswith(f"_{dkey}")]:
                            weights[eid] = matched_weight

    path, cost = astar_route(net, start, end, weights)
    return weights, path, cost

# ═══════════════════════════════════════════════════════
# 方法3：DQN（轻量版，用Q表近似）
# ═══════════════════════════════════════════════════════
class LightDQN:
    """轻量Q-learning路径规划（用于对比实验）"""
    def __init__(self):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def train(self, net, weight_dict, episodes=200):
        nodes = [n.getID() for n in net.getNodes()]
        for ep in range(episodes):
            start_node = random.choice(nodes)
            cur = start_node
            visited = set()
            for _ in range(20):
                visited.add(cur)
                node = net.getNode(cur)
                if node is None: break
                actions = [e.getID() for e in node.getOutgoing()
                           if e.getToNode().getID() not in visited]
                if not actions: break
                if random.random() < self.epsilon:
                    action = random.choice(actions)
                else:
                    action = min(actions, key=lambda a: -self.get_q(cur, a))
                w = weight_dict.get(action, 1.0)
                reward = -w
                next_node = net.getEdge(action).getToNode().getID()
                next_node_obj = net.getNode(next_node)
                next_actions = [e.getID() for e in next_node_obj.getOutgoing()] if next_node_obj else []
                max_next_q = max([self.get_q(next_node, a) for a in next_actions], default=0)
                old_q = self.get_q(cur, action)
                self.q_table[(cur, action)] = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
                cur = next_node

    def find_path(self, net, start_eid, end_eid, weight_dict):
        se = net.getEdge(start_eid)
        ee = net.getEdge(end_eid)
        if not se or not ee: return [], float('inf')
        cur = se.getToNode().getID()
        end_node = ee.getFromNode().getID()
        path = [start_eid]
        visited = set([cur])
        total_cost = weight_dict.get(start_eid, 1.0)
        for _ in range(30):
            if cur == end_node: break
            node = net.getNode(cur)
            if not node: break
            actions = [e.getID() for e in node.getOutgoing()
                       if e.getToNode().getID() not in visited]
            if not actions: break
            action = min(actions, key=lambda a: -self.get_q(cur, a) + weight_dict.get(a, 1.0))
            path.append(action)
            total_cost += weight_dict.get(action, 1.0)
            visited.add(cur)
            cur = net.getEdge(action).getToNode().getID()
        return path, total_cost

_dqn_agent = None
def method_dqn(start, end, desc, weight_dict_hint=None):
    global _dqn_agent
    # 用规则权重作为 DQN 的环境
    rule_weights, _, _ = method_rule_astar(start, end, desc)
    weights = rule_weights
    if _dqn_agent is None:
        _dqn_agent = LightDQN()
        print("  [DQN] 训练中...")
        _dqn_agent.train(net, weights, episodes=300)
    path, cost = _dqn_agent.find_path(net, start, end, weights)
    if not path:
        path, cost = astar_route(net, start, end, weights)
    return weights, path, cost

# ═══════════════════════════════════════════════════════
# 方法4：GCN-Weight（图特征权重估计，轻量版）
# ═══════════════════════════════════════════════════════
def method_gcn_weight(start, end, desc):
    """
    模拟GCN对路网图的权重估计：
    利用路段的拓扑特征（度中心性、位置）+ 描述关键词
    """
    # 计算每条路段的"图中心性"特征
    node_degree = {}
    for node in net.getNodes():
        nid = node.getID()
        deg = len(list(node.getOutgoing())) + len(list(node.getIncoming()))
        node_degree[nid] = deg

    weights = {}
    for eid in ALL_EDGES:
        edge = net.getEdge(eid)
        fn = edge.getFromNode().getID()
        tn = edge.getToNode().getID()
        # 图特征：节点度越高（交叉口越复杂）→ 基础拥堵概率越高
        centrality = (node_degree.get(fn, 1) + node_degree.get(tn, 1)) / 2.0
        base_w = 1.0 + centrality * 0.15
        weights[eid] = min(base_w, 4.0)

    # 叠加关键词语义
    rule_w, _, _ = method_rule_astar(start, end, desc)
    for eid in ALL_EDGES:
        if rule_w.get(eid, 2.0) > 5.0:  # 高权重路段由规则覆盖
            weights[eid] = rule_w[eid] * 0.9  # GCN略保守

    path, cost = astar_route(net, start, end, weights)
    return weights, path, cost

# ═══════════════════════════════════════════════════════
# 方法5：Raw Qwen（无微调基座模型）
# 方法6：CoT Qwen（思维链提示工程）
# 方法7：SFT Qwen（本文方法）
# ═══════════════════════════════════════════════════════
def load_llm(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  加载模型: {model_path}")
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 新版 transformers 不接受 load_in_4bit 直接参数
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    mdl.eval()
    return tok, mdl

def llm_infer(tok, mdl, prompt, max_new=1200):
    import torch
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=900).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new,
                           do_sample=False, repetition_penalty=1.3,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


_ALL_EDGES_STR = (
    "R0C0_E,R0C0_W,R0C1_E,R0C1_W,R0C2_E,R0C2_W,R0C3_E,R0C3_W,R0C4_E,R0C4_W,"
    "R1C0_E,R1C0_W,R1C1_E,R1C1_W,R1C2_E,R1C2_W,R1C3_E,R1C3_W,R1C4_E,R1C4_W,"
    "R2C0_E,R2C0_W,R2C1_E,R2C1_W,R2C2_E,R2C2_W,R2C3_E,R2C3_W,R2C4_E,R2C4_W,"
    "R3C0_E,R3C0_W,R3C1_E,R3C1_W,R3C2_E,R3C2_W,R3C3_E,R3C3_W,R3C4_E,R3C4_W,"
    "R4C0_E,R4C0_W,R4C1_E,R4C1_W,R4C2_E,R4C2_W,R4C3_E,R4C3_W,R4C4_E,R4C4_W,"
    "C0R0_N,C0R0_S,C0R1_N,C0R1_S,C0R2_N,C0R2_S,C0R3_N,C0R3_S,"
    "C1R0_N,C1R0_S,C1R1_N,C1R1_S,C1R2_N,C1R2_S,C1R3_N,C1R3_S,"
    "C2R0_N,C2R0_S,C2R1_N,C2R1_S,C2R2_N,C2R2_S,C2R3_N,C2R3_S,"
    "C3R0_N,C3R0_S,C3R1_N,C3R1_S,C3R2_N,C3R2_S,C3R3_N,C3R3_S,"
    "C4R0_N,C4R0_S,C4R1_N,C4R1_S,C4R2_N,C4R2_S,C4R3_N,C4R3_S,"
    "C5R0_N,C5R0_S,C5R1_N,C5R1_S,C5R2_N,C5R2_S,C5R3_N,C5R3_S"
)

def llm_infer_r1(tok, mdl, desc, max_new=3000):
    """R1专用推理：apply_chat_template + </think>剥离 + penalty=1.3"""
    import torch
    full_instr = (
        "你是交通路径权重生成助手，根据描述为全部98条路段分配拥堵权重（0-10，越大越拥堵）。\n"
        "【重要】只能使用以下路段ID，禁止使用中文路名：\n"
        f"{_ALL_EDGES_STR}\n"
        "输出格式：路段ID:数字, 路段ID:数字, ...\n"
        "权重映射：封闭=9.5, 严重拥堵=7.5, 中度拥堵=5.0, 正常=2.0, 畅通=1.2\n"
        "路网对照：农业路=R0行,红专路=R1行,政七街=R2行,黄河路=R3行,纬五路=R4行；"
        "经一路=C0列,经三路=C1列,经六路=C2列,经八路=C3列,花园路=C4列,未来路=C5列\n"
        f"交通描述：{desc}"
    )
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": full_instr}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=1200, add_special_tokens=False).to(mdl.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new,
                           do_sample=False, repetition_penalty=1.3,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
    # 剥离<think>推理链
    return raw.split("</think>")[-1].strip() if "</think>" in raw else raw

def parse_weights(text):
    weights = {}
    valid = re.compile(r'^(R\d+C\d+_[EW]|C\d+R\d+_[NS])$')
    for eid, w in re.findall(r'(R\d+C\d+_[EW]|C\d+R\d+_[NS])\s*:\s*(\d+(?:\.\d+)?)', text):
        if valid.match(eid):
            weights[eid] = float(w)
    return weights

RAW_PROMPT_TMPL = (
    "根据以下交通描述，为郑州金水区路网的98条路段分配拥堵权重（0-10）。\n"
    "路网说明：农业路=R0,红专路=R1,政七街=R2,黄河路=R3,纬五路=R4；"
    "经一路=C0,经三路=C1,经六路=C2,经八路=C3,花园路=C4,未来路=C5\n"
    "格式：路段ID:权重值\n\n描述：{desc}\n\n输出权重："
)

COT_PROMPT_TMPL = (
    "你是一个交通专家，请按以下步骤分析并输出路段权重：\n"
    "第一步：识别描述中提到的道路和拥堵状态\n"
    "第二步：按严重程度映射权重（封闭=9.5,严重=7.5,中度=5.0,正常=2.0,畅通=1.2）\n"
    "第三步：对未提及路段按道路类型分配默认权重\n"
    "第四步：输出全部98条路段权重，格式：路段ID:数值\n\n"
    "路网：农业路=R0,红专路=R1,政七街=R2,黄河路=R3,纬五路=R4；"
    "经一路=C0,经三路=C1,经六路=C2,经八路=C3,花园路=C4,未来路=C5\n\n"
    "交通描述：{desc}\n\n### 分析过程：\n"
)

SFT_PROMPT_TMPL = (
    "### Instruction:\n"
    "你是交通路径权重生成助手，根据描述为路段分配拥堵权重（0-10，越大越拥堵）。\n"
    "只输出键值对，格式：路段ID:权重值, 路段ID:权重值\n"
    "权重映射：封闭=9.5, 严重拥堵=7.5, 中度拥堵=5.0, 正常=2.0, 畅通=1.2\n"
    "路网：农业路=R0,红专路=R1,政七街=R2,黄河路=R3,纬五路=R4；"
    "经一路=C0,经三路=C1,经六路=C2,经八路=C3,花园路=C4,未来路=C5\n\n"
    "交通描述：{desc}\n\n### Response:\n"
)

# ═══════════════════════════════════════════════════════
# A* 路径规划（公用）
# ═══════════════════════════════════════════════════════
def astar_route(net, start_eid, end_eid, weight_dict):
    try:
        se = net.getEdge(start_eid); ee = net.getEdge(end_eid)
        sn = se.getToNode(); en = ee.getToNode()
    except: return [], float('inf')
    open_list = {sn: (weight_dict.get(start_eid, 1.0), [start_eid])}
    closed = set()
    while open_list:
        cur = min(open_list, key=lambda n: open_list[n][0])
        cost, path = open_list.pop(cur)
        if cur == en: return path, cost
        closed.add(cur)
        for edge in (cur.getOutgoing() if hasattr(cur,'getOutgoing') else cur.getOutgoingEdges()):
            eid = edge.getID(); nbr = edge.getToNode()
            if nbr in closed: continue
            nc = cost + weight_dict.get(eid, 1.0)
            if nbr not in open_list or nc < open_list[nbr][0]:
                open_list[nbr] = (nc, path + [eid])
    return [], float('inf')

# ═══════════════════════════════════════════════════════
# 评估指标计算
# ═══════════════════════════════════════════════════════
# 合法边集合（用于覆盖率计算）
_VALID_EDGE_SET = set(
    [f"R{r}C{c}_{d}" for r in range(5) for c in range(5) for d in ("E","W")] +
    [f"C{c}R{r}_{d}" for c in range(6) for r in range(4) for d in ("N","S")]
)
# 非LLM方法的"中性基准"默认权重（用于区分是否真正解析）
_NEUTRAL_DEFAULTS = {1.0, 2.0}   # Dijkstra=1.0，默认补全=2.0，都不算有效解析

def calc_constraint_rate(pred_weights, gt_weights, threshold=3.0):
    """
    约束符合率：只看 ground_truth 路段的方向是否正确
    高gt(>6) → pred需>4；低gt(<3) → pred需<4
    不含覆盖率，避免默认值污染指标
    """
    if not pred_weights or not gt_weights:
        return 0.0
    hits = sum(1 for e, gw in gt_weights.items()
               if (gw > 6 and pred_weights.get(e, 2.0) > 4) or
                  (gw < 3 and pred_weights.get(e, 2.0) < 4))
    return round(hits / len(gt_weights) * 100, 1)

def calc_coverage(pred_weights):
    """
    路段覆盖率：模型真正解析出的（非默认值）路段比例
    默认补全值(1.0/2.0)不计入覆盖
    """
    valid = sum(1 for e, w in pred_weights.items()
                if e in _VALID_EDGE_SET and w not in _NEUTRAL_DEFAULTS)
    return round(valid / 98 * 100, 1)

# ═══════════════════════════════════════════════════════
# 主实验循环
# ═══════════════════════════════════════════════════════
def run_all(run_llm=True):
    results = []

    # 加载模型（如果需要）
    raw_tok = raw_mdl = sft_tok = sft_mdl = None
    r1_raw_tok = r1_raw_mdl = r1_sft_tok = r1_sft_mdl = None
    sparse_tok = sparse_mdl = None
    if run_llm:
        print("\n📦 加载 Raw Qwen（基座模型）...")
        raw_tok, raw_mdl = load_llm("/root/autodl-tmp/Qwen2.5-1.5B-Instruct")
        print("📦 加载 Qwen-LoRA（CoT全量输出对比方法）...")
        sft_tok, sft_mdl = load_llm("/root/autodl-tmp/model_merged_qwen")
        print("📦 加载 Sparse-LoRA（本文最终方法）...")
        if os.path.exists("/root/autodl-tmp/model_merged_sparse"):
            sparse_tok, sparse_mdl = load_llm("/root/autodl-tmp/model_merged_sparse")
        else:
            print("⚠️  Sparse-LoRA 模型不存在，跳过")
        import os as _os2
        if _os2.path.exists("/root/autodl-tmp/DeepSeek-R1-1.5B"):
            print("📦 加载 R1-Raw（DeepSeek-R1基座）...")
            r1_raw_tok, r1_raw_mdl = load_llm("/root/autodl-tmp/DeepSeek-R1-1.5B")
        if _os2.path.exists("/root/autodl-tmp/model_merged_r1"):
            print("📦 加载 R1-LoRA（R1推理增强方法）...")
            r1_sft_tok, r1_sft_mdl = load_llm("/root/autodl-tmp/model_merged_r1")
        else:
            print("⚠️  R1-LoRA 未找到，跳过")

    for tc in TEST_CASES:
        print(f"\n{'='*55}")
        print(f"🧪 {tc['name']}")
        print(f"   {tc['start']} → {tc['end']}")
        print(f"{'='*55}")

        tc_result = {"scenario": tc["name"], "methods": []}

        methods_to_run = [
            ("Dijkstra",   lambda: method_dijkstra(tc["start"], tc["end"], tc["desc"])),
            ("Rule-A*",    lambda: method_rule_astar(tc["start"], tc["end"], tc["desc"])),
            ("DQN",        lambda: method_dqn(tc["start"], tc["end"], tc["desc"])),
            ("GCN-Weight", lambda: method_gcn_weight(tc["start"], tc["end"], tc["desc"])),
        ]
        if run_llm:
            # ── R1 Raw（未微调，对比格式对齐前后差异）──
            def _r1_raw():
                if r1_raw_tok is None: return {e:1.0 for e in ALL_EDGES}, [], float("inf")
                raw = llm_infer_r1(r1_raw_tok, r1_raw_mdl, tc["desc"])
                w = parse_weights(raw)
                # 未解析到的路段填正常值2.0，统一仿真基准
                for e in ALL_EDGES:
                    w.setdefault(e, 2.0)
                p, cost = astar_route(net, tc["start"], tc["end"], w)
                return w, p, cost

            # ── R1 SFT（微调后）──
            def _r1_sft():
                if r1_sft_tok is None: return {e:1.0 for e in ALL_EDGES}, [], float("inf")
                raw = llm_infer_r1(r1_sft_tok, r1_sft_mdl, tc["desc"])
                w = parse_weights(raw)
                for e in ALL_EDGES:
                    w.setdefault(e, 2.0)
                p, cost = astar_route(net, tc["start"], tc["end"], w)
                return w, p, cost

            def _raw():
                t0 = time.time()
                raw = llm_infer(raw_tok, raw_mdl, RAW_PROMPT_TMPL.format(desc=tc["desc"]))
                w = parse_weights(raw); w = w or {e:1.0 for e in ALL_EDGES}
                p, c = astar_route(net, tc["start"], tc["end"], w)
                return w, p, c
            def _cot():
                t0 = time.time()
                raw = llm_infer(raw_tok, raw_mdl, COT_PROMPT_TMPL.format(desc=tc["desc"]))
                w = parse_weights(raw); w = w or {e:1.0 for e in ALL_EDGES}
                p, c = astar_route(net, tc["start"], tc["end"], w)
                return w, p, c
            def _sft():
                # LoRA 模型用 apply_chat_template（与训练格式一致）
                import torch as _t
                _instr = tc["desc"]
                if not any(k in _instr for k in ["生成权重","输出权重","分配"]):
                    _instr = _instr.rstrip("。") + "。请为全部98条路段生成权重（0-10，越大越拥堵）。"
                _prompt = sft_tok.apply_chat_template(
                    [{"role":"user","content":_instr}],
                    tokenize=False, add_generation_prompt=True)
                _inputs = sft_tok(_prompt, return_tensors="pt",
                                  truncation=True, max_length=4096).to(sft_mdl.device)
                _in_len = _inputs["input_ids"].shape[1]
                with _t.no_grad():
                    _out = sft_mdl.generate(**_inputs, max_new_tokens=3000,
                                            do_sample=False, repetition_penalty=1.0,
                                            pad_token_id=sft_tok.pad_token_id or sft_tok.eos_token_id)
                raw = sft_tok.decode(_out[0][_in_len:], skip_special_tokens=True).strip()
                if "</think>" in raw: raw = raw.split("</think>")[-1]
                w = parse_weights(raw); w = w or {e:1.0 for e in ALL_EDGES}
                p, c = astar_route(net, tc["start"], tc["end"], w)
                return w, p, c
            def _lora_gat():
                import copy, sys as _sys, torch as _t
                _cd = "/root/autodl-tmp/code"
                if _cd not in _sys.path: _sys.path.insert(0, _cd)
                _instr = tc["desc"]
                if not any(k in _instr for k in ["生成权重","输出权重","分配"]):
                    _instr = _instr.rstrip("。") + "。请为全部98条路段生成权重（0-10，越大越拥堵）。"
                _prompt = sft_tok.apply_chat_template(
                    [{"role":"user","content":_instr}],
                    tokenize=False, add_generation_prompt=True)
                _inputs = sft_tok(_prompt, return_tensors="pt", truncation=True,
                                  max_length=4096).to(sft_mdl.device)
                _in_len = _inputs["input_ids"].shape[1]
                with _t.no_grad():
                    _out = sft_mdl.generate(**_inputs, max_new_tokens=3000, do_sample=False,
                                            repetition_penalty=1.0,
                                            pad_token_id=sft_tok.pad_token_id or sft_tok.eos_token_id)
                _raw = sft_tok.decode(_out[0][_in_len:], skip_special_tokens=True).strip()
                if "</think>" in _raw: _raw = _raw.split("</think>")[-1]
                w = parse_weights(_raw); w = w or {e:1.0 for e in ALL_EDGES}
                try:
                    from gat_smoother import smooth_weights
                    w = smooth_weights(copy.deepcopy(w), "/root/autodl-tmp/gat_model.pt", device="cpu")
                except Exception as _ge: print(f"     [GAT跳过] {_ge}")
                p, c = astar_route(net, tc["start"], tc["end"], w)
                return w, p, c

            methods_to_run += [
                ("Raw-Qwen", _raw),
                ("CoT-Qwen", _cot),
                ("Qwen-LoRA★", _sft),
                ("LoRA+GAT★★",   _lora_gat),
                ("R1-Raw",    _r1_raw),
                ("R1-LoRA★",   _r1_sft),
            ]
            # ── Sparse-LoRA★★（本文最终方法：稀疏异常路段输出）──
            def _sparse_lora():
                if sparse_tok is None:
                    return {e: 2.0 for e in ALL_EDGES}, [], float("inf")
                import torch as _t, re as _re

                # 文字→数值映射
                _TEXT_W = [
                    (["全封","封闭","管制","禁行"], 9.5),
                    (["极度拥堵","严重拥堵","严重","极度"], 8.0),
                    (["中度拥堵","较拥堵","中度"], 5.5),
                    (["拥堵","缓行","堵"], 4.5),
                    (["轻微"], 3.5),
                    (["畅通","通畅","无阻"], 1.2),
                ]
                _ROAD_ROW = {"农业路":"R0","红专路":"R1","政七街":"R2","黄河路":"R3","纬五路":"R4"}
                _ROAD_COL = {"经一路":"C0","经三路":"C1","经六路":"C2","经八路":"C3","花园路":"C4","未来路":"C5"}

                def _txt2w(tok):
                    for kwds, v in _TEXT_W:
                        if any(k in tok for k in kwds):
                            return v
                    return None

                def _expand(eid, val, w):
                    m_r = _re.fullmatch(r'R(\d)C(\d)_([EW])', eid)
                    m_c = _re.fullmatch(r'C(\d)R(\d)_([NS])', eid)
                    if m_r:
                        ri, d = m_r.group(1), m_r.group(3)
                        for ci in range(5): w[f"R{ri}C{ci}_{d}"] = val
                    elif m_c:
                        ci, d = m_c.group(1), m_c.group(3)
                        for ri in range(4): w[f"C{ci}R{ri}_{d}"] = val

                _desc = tc["desc"]
                if not any(k in _desc for k in ["异常路段","偏离","异常","严重影响"]):
                    _instr = _desc.rstrip("。") + "。请输出权重偏离正常值2.0超过3.0的异常路段，正常路段无需输出。"
                else:
                    _instr = _desc
                _prompt = sparse_tok.apply_chat_template(
                    [{"role": "user", "content": _instr}],
                    tokenize=False, add_generation_prompt=True)
                _inputs = sparse_tok(_prompt, return_tensors="pt",
                                     truncation=True, max_length=512,
                                     padding=False).to(sparse_mdl.device)
                _in_len = _inputs["input_ids"].shape[1]
                with _t.no_grad():
                    _out = sparse_mdl.generate(
                        **_inputs, max_new_tokens=300, do_sample=False,
                        repetition_penalty=1.0,
                        eos_token_id=sparse_tok.eos_token_id,
                        pad_token_id=sparse_tok.pad_token_id or sparse_tok.eos_token_id)
                _raw = sparse_tok.decode(_out[0][_in_len:], skip_special_tokens=True).strip()
                w = {}

                # Step1: 解析 think 块道路级别
                _think_m = _re.search(r'<think>(.*?)</think>', _raw, _re.DOTALL)
                if _think_m:
                    for _line in _think_m.group(1).split('\n'):
                        _rm  = _re.search(r'([农红政黄纬][业专七河五]路).*?([东西])[^:：]*[:：]\s*([\d.]+)', _line)
                        _cm  = _re.search(r'([经花未]\S{1,2}路).*?([南北])[^:：]*[:：]\s*([\d.]+)', _line)
                        _rt  = _re.search(r'([农红政黄纬][业专七河五]路).*?([东西]).*?(封闭|拥堵|畅通)', _line)
                        _ct  = _re.search(r'([经花未]\S{1,2}路).*?([南北]).*?(封闭|拥堵|畅通)', _line)
                        _ran = _re.search(r'([农红政黄纬][业专七河五]路)[^东西]*全线[^:：]*[:：]\s*([\d.]+)', _line)
                        _can = _re.search(r'([经花未]\S{1,2}路)[^南北]*全线[^:：]*[:：]\s*([\d.]+)', _line)
                        _rat = _re.search(r'([农红政黄纬][业专七河五]路)[^东西]*全线.*(封闭|拥堵|畅通)', _line)
                        _cat = _re.search(r'([经花未]\S{1,2}路)[^南北]*全线.*(封闭|拥堵|畅通)', _line)
                        if _rm:
                            _rid = _ROAD_ROW.get(_rm.group(1))
                            if _rid:
                                _d = "E" if "东" in _rm.group(2) else "W"
                                for ci in range(5): w[f"{_rid}C{ci}_{_d}"] = float(_rm.group(3))
                        elif _rt:
                            _rid = _ROAD_ROW.get(_rt.group(1)); _v = _txt2w(_rt.group(3))
                            if _rid and _v:
                                _d = "E" if "东" in _rt.group(2) else "W"
                                for ci in range(5): w[f"{_rid}C{ci}_{_d}"] = _v
                        elif _ran:
                            _rid = _ROAD_ROW.get(_ran.group(1))
                            if _rid:
                                for ci in range(5):
                                    for _d in ("E","W"): w[f"{_rid}C{ci}_{_d}"] = float(_ran.group(2))
                        elif _rat:
                            _rid = _ROAD_ROW.get(_rat.group(1)); _v = _txt2w(_rat.group(2))
                            if _rid and _v:
                                for ci in range(5):
                                    for _d in ("E","W"): w[f"{_rid}C{ci}_{_d}"] = _v
                        if _cm:
                            _cid = next((c for nm,c in _ROAD_COL.items() if nm in _cm.group(1)), None)
                            if _cid:
                                _d = "N" if "北" in _cm.group(2) else "S"
                                for ri in range(4): w[f"{_cid}R{ri}_{_d}"] = float(_cm.group(3))
                        elif _ct:
                            _cid = next((c for nm,c in _ROAD_COL.items() if nm in _ct.group(1)), None)
                            _v = _txt2w(_ct.group(3))
                            if _cid and _v:
                                _d = "N" if "北" in _ct.group(2) else "S"
                                for ri in range(4): w[f"{_cid}R{ri}_{_d}"] = _v
                        elif _can:
                            _cid = next((c for nm,c in _ROAD_COL.items() if nm in _can.group(1)), None)
                            if _cid:
                                for ri in range(4):
                                    for _d in ("N","S"): w[f"{_cid}R{ri}_{_d}"] = float(_can.group(2))
                        elif _cat:
                            _cid = next((c for nm,c in _ROAD_COL.items() if nm in _cat.group(1)), None)
                            _v = _txt2w(_cat.group(2))
                            if _cid and _v:
                                for ri in range(4):
                                    for _d in ("N","S"): w[f"{_cid}R{ri}_{_d}"] = _v

                # Step2: 解析结构化输出，支持文字权重，并沿路段展开
                _struct = _raw.split("</think>")[-1] if "</think>" in _raw else _raw
                for _eid, _vs in _re.findall(
                        r'\b([RC]\d[CR]\d_[EWNS])\s*[:：]\s*([^\s,;，；\)）]+)', _struct):
                    try:
                        _v = float(_vs)
                        if 0 <= _v <= 10: _expand(_eid, _v, w)
                    except ValueError:
                        _v = _txt2w(_vs)
                        if _v: _expand(_eid, _v, w)

                # 缺失路段填默认值 2.0
                for e in ALL_EDGES:
                    w.setdefault(e, 2.0)
                p, cost = astar_route(net, tc["start"], tc["end"], w)
                return w, p, cost
            methods_to_run.append(("Sparse-LoRA★★", _sparse_lora))

            # ── Sparse-LoRA+GAT★★★（本文终极级联方法）──
            def _sparse_lora_gat():
                if sparse_tok is None:
                    return {e: 2.0 for e in ALL_EDGES}, [], float("inf")
                import torch as _t, re as _re, copy
                
                # 1. 直接调用已有的 Sparse-LoRA 推理获取初始稀疏权重
                w_sparse, _, _ = _sparse_lora()
                w_base = copy.deepcopy(w_sparse)
                
                # 2. 调用 GAT 进行拓扑补全与平滑
                try:
                    from gat_smoother import smooth_weights
                    # 将 Sparse-LoRA 识别出的异常权重作为先验输入 GAT
                    w_final = smooth_weights(w_base, "/root/autodl-tmp/gat_model.pt", device="cpu")
                except Exception as _ge:
                    print(f"     [GAT跳过] {_ge}")
                    w_final = w_base
                    
                p, cost = astar_route(net, tc["start"], tc["end"], w_final)
                return w_final, p, cost
            
            methods_to_run.append(("Sparse-LoRA+GAT★★★", _sparse_lora_gat))

        for name, fn in methods_to_run:
            print(f"\n  ▶ {name}")
            t0 = time.time()
            try:
                weights, path, cost = fn()
                elapsed = time.time() - t0
                coverage    = calc_coverage(weights)
                constraint  = calc_constraint_rate(weights, tc["ground_truth"])
                parsed_cnt  = sum(1 for e, w in weights.items()
                                  if e in ALL_EDGES and w != 1.0)
                print(f"     路径: {' → '.join(path[:4])}{'...' if len(path)>4 else ''}")
                print(f"     代价: {cost:.2f}  覆盖: {coverage:.0f}%  约束符合率: {constraint:.0f}%  耗时: {elapsed:.1f}s")
                tc_result["methods"].append({
                    "method": name,
                    "path_cost": round(cost, 2),
                    "path_len":  len(path),
                    "coverage":  round(coverage, 1),
                    "constraint_rate": round(constraint, 1),
                    "parsed_edges": parsed_cnt,
                    "time_s": round(elapsed, 2),
                })
            except Exception as e:
                print(f"     ❌ 失败: {e}")

        results.append(tc_result)

   # 保存结果 (增量更新模式，保护已有的 LLM 数据)
    out = "/root/autodl-tmp/results/compare_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    existing_data = []
    if os.path.exists(out):
        try:
            with open(out, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            pass

    # 合并逻辑：按 scenario 匹配，再按 method 更新或追加
    for new_scenario in results:
        match_scen = next((s for s in existing_data if s["scenario"] == new_scenario["scenario"]), None)
        if match_scen:
            for new_method in new_scenario["methods"]:
                match_meth_idx = next((i for i, m in enumerate(match_scen["methods"]) if m["method"] == new_method["method"]), -1)
                if match_meth_idx >= 0:
                    match_scen["methods"][match_meth_idx] = new_method  # 更新现有方法
                else:
                    match_scen["methods"].append(new_method)  # 追加新方法
        else:
            existing_data.append(new_scenario)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已增量保存/更新 → {out}")
    return existing_data

if __name__ == "__main__":
    # 先只跑非LLM方法验证流程
    run_all(run_llm="--llm" in sys.argv)
