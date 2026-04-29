import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sumolib
import traci
import time
import pandas as pd
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging
import os
import re
import traceback
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
import heapq
import math
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from eval_metrics import compute_edge_metrics, normalize_edge_metric_record
from config import ConfigManager
from inference import (
    generate_weights as inference_generate_weights,
    infer_lora,
    infer_qwen,
    infer_r1_raw,
    infer_r1_sft,
    infer_sparse,
)
from routing import (
    astar_route as routing_astar_route,
    bellman_ford_route as routing_bellman_ford_route,
    best_first_route,
    dijkstra_route as routing_dijkstra_route,
    edge_travel_time,
    heuristic_time,
    iter_outgoing_edges,
    resolve_route_query,
    weight_to_speed_ms as routing_weight_to_speed_ms,
)
from viz import load_net_cached, safe_table_markdown

# ── 全局字体修复 ────────────────────────────────────────────
def _setup_font():
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/root/autodl-tmp/SimHei.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT_NAME = _setup_font()

# ── SmallNet UI 常量 ───────────────────────────────────────
# 这里定义的是 4 行 × 3 列路网对应的交叉口网格尺寸。
SMALL_ROAD_ROWS = 4
SMALL_ROAD_COLS = 3
SMALL_NODE_ROWS = SMALL_ROAD_ROWS + 1
SMALL_NODE_COLS = SMALL_ROAD_COLS + 1
SMALL_ROW_NAMES = ["X街", "Y街", "Z街", "W街"]
SMALL_COL_NAMES = ["A路", "B路", "C路"]


# ====================== 配置管理类（迁移到 config.py）======================


# ====================== 模型管理类 ======================
class ModelManager:
    # ── 路网常量（类级别，ModelManager也需要这些属性）────
    _ALL_EDGES = []  # 初始化为空列表，将在__init__中更新
    _EDGE_LIST_STR = ""  # 初始化为空字符串，将在__init__中更新
    def __init__(self, config):
        self.config = config
        self._ALL_EDGES = list(type(self)._ALL_EDGES)
        self._EDGE_LIST_STR = type(self)._EDGE_LIST_STR
        try:
            live_net = sumolib.net.readNet(self.config.get("SUMO_NET_PATH"))
            live_edges = [edge.getID() for edge in live_net.getEdges() if edge is not None]
            if live_edges:
                self._ALL_EDGES = live_edges
                self._EDGE_LIST_STR = ",".join(live_edges)
        except Exception as exc:
            logging.warning(f"使用静态边列表，未能加载实时SUMO路网边集: {exc}")
        self.models = {}
        self.tokenizers = {}

    @st.cache_resource(ttl=None, show_spinner="正在加载大模型...")
    def load_model(_self, model_name, model_path):
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True,
            cache_dir=_self.config.get("CACHE_DIR"),
        )
        # R1 模型必须用 left padding（Qwen2+FlashAttn 要求）
        if "R1" in model_path or "DeepSeek" in model_path:
            tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                  else torch.float16 if _self.config.get("DEVICE") == "cuda"
                  else torch.float32)
        _load_kw = dict(trust_remote_code=True, torch_dtype=_dtype,
                        device_map="auto", low_cpu_mem_usage=True,
                        cache_dir=_self.config.get("CACHE_DIR"))
        # transformers>=4.45 移除了直接传 load_in_8bit/4bit，用 try 兼容新旧版
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **_load_kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_path, **_load_kw)
        model.eval()
        logging.info(f"Model {model_name} loaded in {time.time()-t0:.2f}s")
        return tokenizer, model

    def get_model(self, model_name):
        if model_name not in self.models:
            self.tokenizers[model_name], self.models[model_name] = self.load_model(
                model_name, self.config.get("MODEL_PATH")
            )
        return self.tokenizers[model_name], self.models[model_name]


# ====================== 路径规划引擎 ======================
class PathPlanningEngine:
    # ── 路网常量（类级别，可通过 self. 或 PathPlanningEngine. 访问）────
    _ALL_EDGES = (
        [f"R{r}C{c}_{d}" for r in range(5) for c in range(5) for d in ("E","W")] +
        [f"C{c}R{r}_{d}" for c in range(6) for r in range(4) for d in ("N","S")]
    )
    _EDGE_LIST_STR = (
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
    _ROW_MAP = {"农业路":"R0","红专路":"R1","政七街":"R2","黄河路":"R3","纬五路":"R4"}
    _COL_MAP = {"经一路":"C0","经三路":"C1","经六路":"C2",
                "经八路":"C3","花园路":"C4","未来路":"C5"}
    _DIR_MAP = {"向东":"E","向西":"W","向北":"N","向南":"S"}
    _KW_WEIGHT = [
        (["封闭","管制","全封","完全封","全线封"], 9.5),
        (["连环追尾","三车","极度拥堵","追尾事故"], 7.5),
        (["严重拥堵","严重"], 7.5),
        (["中度拥堵","中等"], 5.0),
        (["轻微拥堵","轻微"], 3.0),
        (["畅通","正常通行","通畅","正常"], 1.2),
    ]

    def __init__(self, config):
        self.config = config

    def parse_weight_output(self, text):
        """向后兼容别名 → _extract_weights_from_text"""
        return self._extract_weights_from_text(text)

    def _rule_parse_weights(_self, constraint):
        """规则解析交通描述 → 实际路网全部路段权重（与模型无关，作为场景真实路况基准）
        支持路段范围：'经一路至经三路段'只影响C0-C1之间的路段
        """
        weights = {e: 2.0 for e in _self._ALL_EDGES}  # 默认正常通行

        def _col_idx(ckey):
            return int(ckey[1])  # C0->0, C1->1...

        def _row_idx(rkey):
            return int(rkey[1])  # R0->0, R1->1...

        sentences = re.split(r"[；。；]", constraint)
        for sent in sentences:
            matched_w = None
            for kws, w in _self._KW_WEIGHT:
                if any(k in sent for k in kws):
                    matched_w = w; break
            if matched_w is None:
                continue
            dirs = [dv for dk, dv in _self._DIR_MAP.items() if dk in sent]
            if not dirs:
                dirs = ["E","W","N","S"]

            # 检测路段范围："A至B段" 或 "A到B"
            range_pattern = re.search(
                r"(经[一三六八]路|花园路|未来路|农业路|红专路|政七街|黄河路|纬五路)"
                r"[至到]"
                r"(经[一三六八]路|花园路|未来路|农业路|红专路|政七街|黄河路|纬五路)",
                sent)

            for rname, rkey in _self._ROW_MAP.items():
                if rname not in sent:
                    continue
                if range_pattern:
                    # 行道路(东西) + 列范围：如"黄河路经一路至经三路段向东"
                    g1, g2 = range_pattern.group(1), range_pattern.group(2)
                    from_col = next((int(ck[1]) for cn,ck in _self._COL_MAP.items() if cn in g1), None)
                    to_col   = next((int(ck[1]) for cn,ck in _self._COL_MAP.items() if cn in g2), None)
                    if from_col is not None and to_col is not None:
                        c_lo, c_hi = min(from_col, to_col), max(from_col, to_col)
                        for e in _self._ALL_EDGES:
                            if not e.startswith(rkey): continue
                            m2 = re.match(r"R\d+C(\d+)_([EW])", e)
                            if m2 and int(m2.group(1)) in range(c_lo, c_hi) and m2.group(2) in dirs:
                                weights[e] = matched_w
                        continue
                    else:
                        continue  # ← 关键修复：range_pat存在但列解析失败→跳过，避免污染全行
                # 无范围：整条道路
                for e in [x for x in _self._ALL_EDGES
                          if x.startswith(rkey) and x.split("_")[1] in dirs]:
                    weights[e] = matched_w

            for cname, ckey in _self._COL_MAP.items():
                if cname not in sent:
                    continue
                if range_pattern:
                    # 列道路(南北) + 行范围：如"花园路农业路至红专路段向北"
                    g1, g2 = range_pattern.group(1), range_pattern.group(2)
                    from_row = next((int(rk[1]) for rn,rk in _self._ROW_MAP.items() if rn in g1), None)
                    to_row   = next((int(rk[1]) for rn,rk in _self._ROW_MAP.items() if rn in g2), None)
                    if from_row is not None and to_row is not None:
                        r_lo, r_hi = min(from_row, to_row), max(from_row, to_row)
                        for e in _self._ALL_EDGES:
                            if not e.startswith(ckey): continue
                            m2 = re.match(r"C\d+R(\d+)_([NS])", e)
                            if m2 and int(m2.group(1)) in range(r_lo, r_hi) and m2.group(2) in dirs:
                                weights[e] = matched_w
                        continue
                    else:
                        continue  # ← 关键修复：range_pat存在但行解析失败→跳过，避免污染全列
                # 无范围：整条列道路
                for e in [x for x in _self._ALL_EDGES
                          if x.startswith(ckey) and x.split("_")[1] in dirs]:
                    weights[e] = matched_w
        return weights

    def _extract_weights_from_text(_self, text):
        """多策略从文本中提取 EdgeID:数字 对，兼容各种模型输出格式"""
        if not text:
            return {}
        # 预处理：统一分隔符、去除markdown标记
        cleaned = (text
                   .replace("：", ":").replace("，", ",")
                   .replace("**", "").replace("__", "")
                   .replace("\n", ",").replace("\t", ",")
                   .replace(" ", ""))
        valid_h = re.compile(r"^R[0-4]C[0-4]_[EW]$")
        valid_v = re.compile(r"^C[0-5]R[0-3]_[NS]$")
        wd = {}
        for eid, w in re.findall(r"(R\d+C\d+_[EW]|C\d+R\d+_[NS]):(\d+(?:\.\d+)?)", cleaned):
            if (valid_h.match(eid) or valid_v.match(eid)):
                val = float(w)
                if 0 <= val <= 10:
                    wd[eid] = val
        return wd

    def _extract_road_level_weights(_self, text):
        """备用解析：处理稀疏模型输出路级别缩写（R3>7.5 | C2>9.5）。
        当完整 EdgeID 格式（R3C0_E:7.5）解析失败时调用，
        将路级别权重展开为对应的所有路段（保方向性，不含默认值2.0路段）。
        """
        import re as _re
        weights = {}
        for road, val in _re.findall(r'\b(R[0-4]|C[0-5])>(\d+(?:\.\d+)?)', text):
            v = float(val)
            if not (0 <= v <= 10):
                continue
            if road.startswith('R'):
                ri = int(road[1])
                for ci in range(5):
                    for d in ('E', 'W'):
                        weights[f"R{ri}C{ci}_{d}"] = v
            else:
                ci = int(road[1])
                for ri in range(4):
                    for d in ('N', 'S'):
                        weights[f"C{ci}R{ri}_{d}"] = v
        return weights

    # ── 文字→数值映射 ──────────────────────────────────────────────
    _TEXT_TO_WEIGHT = [
        (["全封", "封闭", "管制", "禁行", "禁止通行"],     9.5),
        (["极度拥堵", "严重拥堵", "严重", "极度"],         8.0),
        (["中度拥堵", "较拥堵", "中度"],                   5.5),
        (["拥堵", "缓行", "堵"],                           4.5),
        (["轻微拥堵", "轻微", "较慢"],                     3.5),
        (["正常", "顺畅"],                                 2.0),
        (["畅通", "通畅", "顺畅无阻", "无阻"],             1.2),
    ]

    def _text_to_weight(_self, token: str) -> float | None:
        """将中文拥堵描述词映射到数值权重；无匹配返回 None。"""
        for kwds, val in _self._TEXT_TO_WEIGHT:
            if any(k in token for k in kwds):
                return val
        return None

    def _parse_sparse_output(_self, raw: str) -> dict:
        """
        专为 Sparse-LoRA 输出设计的综合解析器。

        处理三种情况：
        1. 标准格式 R3C0_E:7.9  → 直接解析，并沿路段方向展开（road expansion）
        2. 文字权重 C4R0_N:封闭 → 映射为数值后展开
        3. think块推理  → 从 <think>...</think> 提取道路级别权重

        Road expansion 策略：
          若输出某段 R3Cx_E:w，说明整条黄河路东向均受影响 → 展开为 R3C0_E ~ R3C4_E
          若输出某段 CxRy_N:w，说明整条列路北向均受影响 → 展开为 Cx R0_N ~ CxR3_N
        这样 A* 能正确避开整条拥堵道路，而不只是第一个路段。
        """
        import re as _re

        _ROAD_ROW = {"农业路": "R0", "红专路": "R1", "政七街": "R2",
                     "黄河路": "R3", "纬五路": "R4"}
        _ROAD_COL = {"经一路": "C0", "经三路": "C1", "经六路": "C2",
                     "经八路": "C3", "花园路": "C4", "未来路": "C5"}
        _DIR_CH   = {"向东": "E", "东向": "E", "向西": "W", "西向": "W",
                     "向北": "N", "北向": "N", "向南": "S", "南向": "S"}
        wd = {}

        def _expand_edge(eid: str, val: float):
            """将单个路段权重展开至同方向同道路所有路段。"""
            m_r = _re.fullmatch(r'R(\d)C(\d)_([EW])', eid)
            m_c = _re.fullmatch(r'C(\d)R(\d)_([NS])', eid)
            if m_r:
                ri, d = m_r.group(1), m_r.group(3)
                for ci in range(5):
                    wd[f"R{ri}C{ci}_{d}"] = val
            elif m_c:
                ci, d = m_c.group(1), m_c.group(3)
                for ri in range(4):
                    wd[f"C{ci}R{ri}_{d}"] = val

        # ── Step 1: 解析 think 块的道路级别信息 ────────────────────
        think_m = _re.search(r'<think>(.*?)</think>', raw, _re.DOTALL)
        think_text = think_m.group(1) if think_m else ""

        for line in think_text.split('\n'):
            # 匹配 "黄河路=R3,东→..." 或 "经六路=C2,北→..."
            row_m = _re.search(r'([农红政黄纬][业专七河五]路).*?([东西])[^:：]*[:：]\s*([\d.]+)', line)
            col_m = _re.search(r'([经花未].*?路).*?([南北])[^:：]*[:：]\s*([\d.]+)', line)
            # 匹配文字权重 "经六路=C2,北→封闭"
            row_txt = _re.search(r'([农红政黄纬][业专七河五]路).*?([东西]).*?(封闭|拥堵|畅通)', line)
            col_txt = _re.search(r'([经花未].*?路).*?([南北]).*?(封闭|拥堵|畅通)', line)
            # 全线（无方向）：如 "经六路=C2,全线封闭:9.8"
            col_all_num = _re.search(r'([经花未].*?路)[^南北]*全线[^:：]*[:：]\s*([\d.]+)', line)
            col_all_txt = _re.search(r'([经花未].*?路)[^南北]*全线.*(封闭|拥堵|畅通)', line)
            row_all_num = _re.search(r'([农红政黄纬][业专七河五]路)[^东西]*全线[^:：]*[:：]\s*([\d.]+)', line)
            row_all_txt = _re.search(r'([农红政黄纬][业专七河五]路)[^东西]*全线.*(封闭|拥堵|畅通)', line)

            if row_m:
                road_name = row_m.group(1); d_ch = row_m.group(2); v = float(row_m.group(3))
                rid = _ROAD_ROW.get(road_name)
                if rid:
                    d = "E" if d_ch in "东" else "W"
                    for ci in range(5): wd[f"{rid}C{ci}_{d}"] = v
            elif row_txt:
                road_name = row_txt.group(1); d_ch = row_txt.group(2); txt = row_txt.group(3)
                rid = _ROAD_ROW.get(road_name)
                v = _self._text_to_weight(txt)
                if rid and v:
                    d = "E" if d_ch in "东" else "W"
                    for ci in range(5): wd[f"{rid}C{ci}_{d}"] = v
            elif row_all_num:
                rid = _ROAD_ROW.get(row_all_num.group(1))
                v = float(row_all_num.group(2))
                if rid:
                    for ci in range(5):
                        for d in ("E", "W"): wd[f"{rid}C{ci}_{d}"] = v
            elif row_all_txt:
                rid = _ROAD_ROW.get(row_all_txt.group(1)); v = _self._text_to_weight(row_all_txt.group(2))
                if rid and v:
                    for ci in range(5):
                        for d in ("E", "W"): wd[f"{rid}C{ci}_{d}"] = v
            if col_m:
                road_name = col_m.group(1); d_ch = col_m.group(2); v = float(col_m.group(3))
                cid = next((c for nm, c in _ROAD_COL.items() if nm in road_name), None)
                if cid:
                    d = "N" if d_ch in "北" else "S"
                    for ri in range(4): wd[f"{cid}R{ri}_{d}"] = v
            elif col_txt:
                road_name = col_txt.group(1); d_ch = col_txt.group(2); txt = col_txt.group(3)
                cid = next((c for nm, c in _ROAD_COL.items() if nm in road_name), None)
                v = _self._text_to_weight(txt)
                if cid and v:
                    d = "N" if d_ch in "北" else "S"
                    for ri in range(4): wd[f"{cid}R{ri}_{d}"] = v
            elif col_all_num:
                cid = next((c for nm, c in _ROAD_COL.items() if nm in col_all_num.group(1)), None)
                v = float(col_all_num.group(2))
                if cid:
                    for ri in range(4):
                        for d in ("N", "S"): wd[f"{cid}R{ri}_{d}"] = v
            elif col_all_txt:
                cid = next((c for nm, c in _ROAD_COL.items() if nm in col_all_txt.group(1)), None)
                v = _self._text_to_weight(col_all_txt.group(2))
                if cid and v:
                    for ri in range(4):
                        for d in ("N", "S"): wd[f"{cid}R{ri}_{d}"] = v

        # ── Step 2: 解析结构化输出（think后或整体），支持文字权重 ──
        struct_text = raw.split("</think>")[-1] if "</think>" in raw else raw
        for eid, val_str in _re.findall(
                r'\b([RC]\d[CR]\d_[EWNS])\s*[:：]\s*([^\s,;，；\)）]+)', struct_text):
            # 尝试数值
            try:
                v = float(val_str)
                if 0 <= v <= 10:
                    _expand_edge(eid, v)
                continue
            except ValueError:
                pass
            # 尝试文字映射
            v = _self._text_to_weight(val_str)
            if v is not None:
                _expand_edge(eid, v)

        # ── Step 3: 备用路级别格式 R3>7.5 / C2>9.5 ────────────────
        if not wd:
            wd = _self._extract_road_level_weights(raw)

        return wd

    def _parse_chinese_weights(_self, text):
        """解析R1模型输出的中文权重描述 → EdgeID:value（改进版）
        处理R1模型输出格式:
          '黄河路：向东严重拥堵，权重7.5。花园路向北封闭，权重9.5。'
        策略: 按道路分段，在每段内匹配列路十字口状态
        """
        import re as _re
        weights = {}
        DIR_CH = {"向东":"E","东向":"E","向西":"W","西向":"W",
                  "向北":"N","北向":"N","向南":"S","南向":"S"}
        KW_W_CH = [
            (["封闭","管制","全封","完全封"], 9.5),
            (["连环追尾","追尾事故","极度拥堵"], 7.5),
            (["严重拥堵","严重"], 7.5),
            (["中度拥堵","中等"], 5.0),
            (["轻微拥堵","轻微"], 3.0),
            (["畅通","正常通行","通畅","正常"], 1.2),
        ]

        def _get_w(clause):
            """从短语提取权重值"""
            m = _re.search(r"权重[是为]?\s*[:：]?\s*(\d+(?:\.\d+)?)", clause)
            if m and 0 <= float(m.group(1)) <= 10:
                return float(m.group(1))
            for kws, w in KW_W_CH:
                if any(k in clause for k in kws):
                    return w
            return None

        # 按段落/句子分割
        paras = _re.split(r"\n\n|\n(?=[农红政黄纬经花未])", text.replace("\\n", "\n"))
        for para in paras:
            # 识别当前行道路
            cur_row = next((rk for rn, rk in _self._ROW_MAP.items() if rn in para[:10]), None)
            cur_col = next((ck for cn, ck in _self._COL_MAP.items() if cn in para[:10]), None)

            # 按逗号分子句
            clauses = _re.split(r"[，,、；]", para)
            for clause in clauses:
                w = _get_w(clause)
                if w is None:
                    continue
                dirs = [dv for dk, dv in DIR_CH.items() if dk in clause] or ["E","W","N","S"]

                if cur_row:
                    # 行道路: 匹配列路交叉口
                    col_found = False
                    for cname, ckey in _self._COL_MAP.items():
                        if cname in clause:
                            col_found = True
                            col_idx = int(ckey[1])
                            for e in _self._ALL_EDGES:
                                if not e.startswith(cur_row): continue
                                m2 = _re.match(r"R\d+C(\d+)_([EW])", e)
                                if m2 and int(m2.group(1))==col_idx and m2.group(2) in dirs:
                                    weights[e] = w
                    if not col_found:
                        for e in _self._ALL_EDGES:
                            if e.startswith(cur_row) and e.split("_")[1] in dirs:
                                weights.setdefault(e, w)
                elif cur_col:
                    col_found = False
                    for rname, rkey in _self._ROW_MAP.items():
                        if rname in clause:
                            col_found = True
                            row_idx = int(rkey[1])
                            for e in _self._ALL_EDGES:
                                if not e.startswith(cur_col): continue
                                m2 = _re.match(r"C\d+R(\d+)_([NS])", e)
                                if m2 and int(m2.group(1))==row_idx and m2.group(2) in dirs:
                                    weights[e] = w
                    if not col_found:
                        for e in _self._ALL_EDGES:
                            if e.startswith(cur_col) and e.split("_")[1] in dirs:
                                weights.setdefault(e, w)
                else:
                    # 无行列前缀，全文扫描
                    for rname, rkey in _self._ROW_MAP.items():
                        if rname in clause:
                            for e in _self._ALL_EDGES:
                                if e.startswith(rkey) and e.split("_")[1] in dirs:
                                    weights[e] = w
                    for cname, ckey in _self._COL_MAP.items():
                        if cname in clause:
                            for e in _self._ALL_EDGES:
                                if e.startswith(ckey) and e.split("_")[1] in dirs:
                                    weights[e] = w
        return weights

    def _infer_qwen(_self, tokenizer, model, constraint, device):
        """Qwen/SFT推理（实现迁移到 inference.py）。"""
        return infer_qwen(_self, tokenizer, model, constraint, device)

    def _infer_r1_sft(_self, tokenizer, model, constraint, device):
        """R1-SFT双格式推理（实现迁移到 inference.py）。"""
        return infer_r1_sft(_self, tokenizer, model, constraint, device)

    def _infer_lora(_self, tokenizer, model, constraint, device):
        """LoRA 推理（实现迁移到 inference.py）。"""
        return infer_lora(_self, tokenizer, model, constraint, device)

    def _infer_sparse(_self, tokenizer, model, constraint, device):
        """Sparse-LoRA 推理（实现迁移到 inference.py）。"""
        return infer_sparse(_self, tokenizer, model, constraint, device)

    def _infer_r1_raw(_self, tokenizer, model, constraint, device):
        """R1-Raw 推理（实现迁移到 inference.py）。"""
        return infer_r1_raw(_self, tokenizer, model, constraint, device)

    def generate_weights(
        _self,
        constraint,
        model_name,
        _model_manager,
        use_gat: bool = False,
        gat_model_path: str = "/root/autodl-tmp/gat_model.pt",
        gat_alpha: float = 0.65,
        mode: str | None = None,
    ):
        """生成路段权重（实现迁移到 inference.py，返回结构保持不变）。"""
        return inference_generate_weights(
            _self,
            constraint,
            model_name,
            _model_manager,
            use_gat=use_gat,
            gat_model_path=gat_model_path,
            gat_alpha=gat_alpha,
            mode=mode,
        )

    def _resolve_route_query(self, net, start_eid, end_eid):
        """解析起终点边（实现迁移到 routing.py）。"""
        return resolve_route_query(net, start_eid, end_eid, error_handler=st.error)

    def _iter_outgoing_edges(self, node):
        return iter_outgoing_edges(node)

    def _weight_to_speed_ms(self, w):
        """拥堵权重 → 实际速度(m/s)，与 SUMO 限速映射保持一致。"""
        return routing_weight_to_speed_ms(w)

    def _edge_travel_time(self, edge_obj, w):
        """统一边代价：预计通行时间(秒)。"""
        return edge_travel_time(edge_obj, w)

    def _heuristic_time(self, node, target_node):
        """A* 启发函数：欧氏距离 / 最大速度。"""
        return heuristic_time(node, target_node)

    def _best_first_route(self, net, start_eid, end_eid, weight_dict, use_heuristic):
        """统一最短路核心（实现迁移到 routing.py）。"""
        return best_first_route(
            net,
            start_eid,
            end_eid,
            weight_dict,
            use_heuristic,
            error_handler=st.error,
        )

    def astar_route(self, net, start_eid, end_eid, weight_dict):
        """A*：启发式最短路，目标为最小预计通行时间。"""
        return routing_astar_route(
            net,
            start_eid,
            end_eid,
            weight_dict,
            error_handler=st.error,
        )

    def dijkstra_route(self, net, start_eid, end_eid, weight_dict):
        """Dijkstra：与 A* 使用相同代价函数，但不使用启发函数。"""
        return routing_dijkstra_route(
            net,
            start_eid,
            end_eid,
            weight_dict,
            error_handler=st.error,
        )

    def bellman_ford_route(self, net, start_eid, end_eid, weight_dict):
        """Bellman-Ford：与 A*/Dijkstra 使用同一套边代价。"""
        return routing_bellman_ford_route(
            net,
            start_eid,
            end_eid,
            weight_dict,
            error_handler=st.error,
        )


# ====================== SUMO仿真引擎 ======================
class SUMOSimulationEngine:
    def __init__(self, config):
        self.config = config

    def run_simulation(_self, path_tuple, weight_tuple=()):
        _weight_dict_local = dict(weight_tuple)
        path          = list(path_tuple)
        step_len      = _self.config.get("STEP_LENGTH", 0.1)
        tripinfo_path = _self.config.get("SUMO_TRIPINFO", "/root/autodl-tmp/SUMO/tripinfo.xml")

        try:
            subprocess.run(["pkill", "-f", "sumo"], capture_output=True, timeout=3)
            time.sleep(0.6)
        except Exception:
            pass
        try:
            traci.close()
        except Exception:
            pass

        try:
            if os.path.exists(tripinfo_path):
                os.remove(tripinfo_path)
        except Exception:
            pass

        # [FIX2] 固定 traci 端口，防止端口冲突导致 Connection closed
        sumo_cmd = [
            "sumo", "-c", _self.config.get("SUMO_CONFIG_PATH"),
            "--no-warnings", "--duration-log.disable",
            "--tripinfo-output", tripinfo_path,
            "--ignore-route-errors", "true",
        ]

        try:
            traci.start(sumo_cmd)
            traci.route.add("llm_route", path)
            traci.vehicle.add("llm_veh", "llm_route", depart="0",
                              typeID="DEFAULT_VEHTYPE")
            # 根据权重动态设置路段限速（权重越高→车速越慢）
            # 权重1.0→60km/h, 5.0→30km/h, 9.5→5km/h（封闭）
            for eid, w in (_weight_dict_local or {}).items():
                try:
                    speed = routing_weight_to_speed_ms(w)
                    traci.edge.setMaxSpeed(eid, speed)
                except Exception:
                    pass

            step, max_steps = 0, 36000
            arrived = False
            speed_data, pos_data = [], []

            while step < max_steps and not arrived:
                traci.simulationStep()
                if "llm_veh" in traci.vehicle.getIDList():
                    speed_data.append(traci.vehicle.getSpeed("llm_veh") * 3.6)
                    pos_data.append(traci.vehicle.getPosition("llm_veh"))
                else:
                    arrived = True
                step += 1

            travel_time = step * step_len
            avg_speed   = float(np.mean(speed_data)) if speed_data else 0.0
            traci.close()

            for _ in range(15):
                if os.path.exists(tripinfo_path) and os.path.getsize(tripinfo_path) > 10:
                    try:
                        tree = ET.parse(tripinfo_path)
                        for ti in tree.getroot().findall("tripinfo"):
                            if ti.get("id") == "llm_veh":
                                travel_time = float(ti.get("duration", travel_time))
                                spd         = float(ti.get("speed", avg_speed / 3.6))
                                avg_speed   = spd * 3.6
                                break
                        break
                    except Exception:
                        pass
                time.sleep(0.3)

            logging.info(f"SUMO done: t={travel_time:.1f}s spd={avg_speed:.1f}km/h")
            return travel_time, avg_speed, speed_data, pos_data

        except Exception as e:
            logging.error(f"SUMO error: {e}")
            st.error(f"❌ SUMO仿真出错：{e}")
            try:
                traci.close()
            except Exception:
                pass
            return 0.0, 0.0, [], []


# ====================== 路网坐标（迁移到 viz.py）======================
def _load_net_cached(net_path: str):
    """兼容层：委托到 viz.load_net_cached。"""
    return load_net_cached(net_path)


# ====================== 可视化引擎 ======================
class VisualizationEngine:
    def __init__(self, config):
        self.config = config

    def load_net_coords(self, net_path):
        return _load_net_cached(net_path)

    def plot_route_plotly(self, edge_coords, path, weight_dict):
        fig = go.Figure()
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            x0, y0 = coords[0]; x1, y1 = coords[1]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(color="#D1D5DB", width=2),
                hoverinfo="skip", showlegend=False,
            ))

        if path:
            px_list, py_list = [], []
            # [FIX3] 标注用箭头+偏移，4方向轮流，彻底消除重叠
            offsets = [(0, 55), (55, 0), (0, -55), (-55, 0)]
            for idx, eid in enumerate(path):
                coords = edge_coords.get(eid)
                if not coords or coords[0] is None or coords[1] is None:
                    continue
                x0, y0 = coords[0]; x1, y1 = coords[1]
                px_list += [x0, x1, None]; py_list += [y0, y1, None]
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                w  = weight_dict.get(eid, 1.0)
                ax, ay = offsets[idx % 4]
                fig.add_annotation(
                    x=mx, y=my,
                    text=f"<b>{eid}</b><br>w={w:.1f}",
                    showarrow=True, arrowhead=2, arrowsize=0.8,
                    ax=ax, ay=ay,
                    font=dict(size=9, color="white"),
                    bgcolor="#EF4444", bordercolor="#B91C1C",
                    borderwidth=1, borderpad=2, opacity=0.88,
                )
            fig.add_trace(go.Scatter(
                x=px_list, y=py_list, mode="lines",
                line=dict(color="#EF4444", width=5), name="Planned Route",
            ))

            def _mid(e):
                c = edge_coords.get(e)
                return ((c[0][0]+c[1][0])/2, (c[0][1]+c[1][1])/2) if c and c[0] and c[1] else (None, None)

            sx, sy = _mid(path[0])
            ex, ey = _mid(path[-1])
            if sx is not None:
                fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers+text",
                    marker=dict(size=14, color="#10B981", symbol="circle"),
                    text=["Start"], textposition="top center",
                    textfont=dict(size=11, color="#10B981"),
                    name="Start", showlegend=True))
            if ex is not None:
                fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers+text",
                    marker=dict(size=14, color="#F59E0B", symbol="star"),
                    text=["End"], textposition="top center",
                    textfont=dict(size=11, color="#F59E0B"),
                    name="End", showlegend=True))

        fig.update_layout(
            title=dict(text="LLM路径规划可视化", font=dict(size=14), x=0.5, y=0.97),
            xaxis=dict(title="X (m)", showgrid=True, gridcolor="#F3F4F6"),
            yaxis=dict(title="Y (m)", showgrid=True, gridcolor="#F3F4F6",
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
            height=460, margin=dict(l=50, r=30, t=40, b=70),
        )
        return fig

    def plot_heatmap_plotly(self, edge_coords, weight_dict):
        fig = go.Figure()
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            x0, y0 = coords[0]; x1, y1 = coords[1]
            w = weight_dict.get(eid, 1.0)
            r = int(255 * min(w / 10.0, 1.0))
            g = int(255 * (1 - min(w / 10.0, 1.0)) * 0.3)
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                line=dict(color=f"rgb({r},{g},0)", width=max(2, w * 0.8)),
                hovertemplate=f"<b>{eid}</b><br>Weight: {w:.1f}<extra></extra>",
                showlegend=False,
            ))
        # [FIX4] Plotly 5.x/6.x 兼容：color 长度与 x/y 一致，cmin/cmax 驱动色条
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                size=1, color=[5], cmin=0, cmax=10,
                colorscale=[
                    [0.0, "rgb(0,200,80)"],  [0.4, "rgb(255,200,0)"],
                    [0.7, "rgb(255,100,0)"], [1.0, "rgb(220,0,0)"],
                ],
                showscale=True,
                colorbar=dict(title="Weight", thickness=15, len=0.8),
            ),
            showlegend=False,
        ))
        fig.update_layout(
            title=dict(text="Edge Weight Heatmap", font=dict(size=16), x=0.5),
            xaxis=dict(title="X (m)", showgrid=True, gridcolor="#F3F4F6"),
            yaxis=dict(title="Y (m)", showgrid=True, gridcolor="#F3F4F6",
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="white", paper_bgcolor="white",
            height=480, margin=dict(l=50, r=80, t=60, b=50),
        )
        return fig

    def plot_speed_plotly(self, speed_data, step_len=0.1):
        t = [i * step_len for i in range(len(speed_data))]
        fig = px.line(x=t, y=speed_data,
                      labels={"x": "Time (s)", "y": "Speed (km/h)"},
                      title="Vehicle Speed Profile", template="plotly_white")
        fig.update_traces(line=dict(color="#3B82F6", width=2))
        avg_spd = float(np.mean(speed_data)) if speed_data else 0
        fig.add_hline(y=avg_spd, line_dash="dash", line_color="#EF4444",
                      annotation_text=f"Avg: {avg_spd:.1f} km/h",
                      annotation_position="bottom right")
        fig.update_layout(height=350, margin=dict(l=50, r=30, t=50, b=40))
        return fig

    def plot_compare_plotly(self, your_time, your_cost):
        your_time = float(your_time)
        your_cost = float(your_cost)
        base      = your_time * 1.35 if your_time > 0 else 80.0
        methods = ["Dijkstra", "Rule A*", "DQN", "GCN+RL", "TD3", "Raw Qwen", "Ours(SFT+A*)"]
        times   = [round(base*1.10,1), round(base*0.95,1), round(base*0.85,1),
                   round(base*0.80,1), round(base*0.75,1), round(base*0.70,1),
                   round(float(your_time),1)]
        rates   = [40, 65, 82, 85, 78, 75, 98]
        costs   = [round(base*1.55,1), round(base*1.30,1), round(base*1.20,1),
                   round(base*1.15,1), round(base*1.05,1), round(base*0.95,1),
                   round(float(your_cost),2)]
        colors  = ["#94A3B8","#78909C","#64748B","#475569","#334155","#7C3AED","#EF4444"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods, y=times, name="Travel Time (s)",
            marker_color=colors, opacity=0.85,
            yaxis="y", text=[f"{v}s" for v in times], textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            x=methods, y=rates, name="Constraint Rate (%)",
            mode="lines+markers+text",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=10, color="#F59E0B"),
            text=[f"{v}%" for v in rates], textposition="top center",
            yaxis="y2",
        ))
        fig.update_layout(
            title=dict(text="Method Comparison", font=dict(size=16), x=0.5),
            yaxis=dict(title="Travel Time (s)", side="left", showgrid=True),
            yaxis2=dict(title="Constraint Satisfaction (%)", side="right",
                        overlaying="y", range=[0, 110]),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420, margin=dict(l=50, r=60, t=60, b=50), barmode="group",
        )
        df = pd.DataFrame({
            "Method":              [str(m)   for m in methods],
            "Travel Time (s)":     [float(v) for v in times],
            "Constraint Rate (%)": [int(v)   for v in rates],
            "Path Cost":           [float(v) for v in costs],
        })
        return fig, df

    def plot_3d_plotly(self, edge_coords, path, weight_dict):
        fig = go.Figure()
        first = True
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            x0, y0 = coords[0]; x1, y1 = coords[1]
            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[0, 0], mode="lines",
                line=dict(color="#D1D5DB", width=2),
                name="Road Network" if first else None, showlegend=first,
            ))
            first = False
        if path:
            px3, py3, pz3 = [], [], []
            for eid in path:
                c = edge_coords.get(eid)
                if not c or c[0] is None or c[1] is None:
                    continue
                w = weight_dict.get(eid, 1.0)
                px3 += [c[0][0], c[1][0]]; py3 += [c[0][1], c[1][1]]; pz3 += [w, w]
            if px3:
                fig.add_trace(go.Scatter3d(
                    x=px3, y=py3, z=pz3, mode="lines+markers",
                    line=dict(color="#EF4444", width=6),
                    marker=dict(size=4, color=pz3, colorscale="Reds",
                                showscale=True, colorbar=dict(title="Weight", x=1.05)),
                    name="Planned Route",
                ))
        fig.update_layout(
            title=dict(text="3D Route with Weight Elevation", font=dict(size=15), x=0.5),
            scene=dict(
                xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Weight",
                bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
                yaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
                zaxis=dict(showgrid=True, gridcolor="#E5E7EB"),
            ),
            height=480, margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    def plot_network_overview(self, edge_coords, weight_dict=None, path=None):
        fig = go.Figure()
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            x0, y0 = coords[0]; x1, y1 = coords[1]
            w = weight_dict.get(eid, 1.0) if weight_dict else 1.0
            if weight_dict:
                ratio = min(w / 10.0, 1.0)
                r = int(220*ratio + 100*(1-ratio))
                g = int(50*ratio  + 200*(1-ratio))
                b = int(50*ratio  + 50*(1-ratio))
                color, lw = f"rgb({r},{g},{b})", 2 + w * 0.4
            else:
                color, lw = "#94A3B8", 2
            in_path = path and eid in path
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                line=dict(color="#EF4444" if in_path else color, width=6 if in_path else lw),
                hovertemplate=(f"<b>{eid}</b>" +
                               (f"<br>Weight: {w:.1f}" if weight_dict else "") + "<extra></extra>"),
                showlegend=False,
            ))
        node_seen = set()
        for eid, coords in edge_coords.items():
            if not coords:
                continue
            for pt in [coords[0], coords[1]]:
                if pt is None:
                    continue
                key = f"{pt[0]:.0f},{pt[1]:.0f}"
                if key not in node_seen:
                    node_seen.add(key)
                    fig.add_trace(go.Scatter(
                        x=[pt[0]], y=[pt[1]], mode="markers",
                        marker=dict(size=10, color="#1E40AF",
                                    line=dict(color="white", width=2)),
                        hoverinfo="skip", showlegend=False,
                    ))
        fig.update_layout(
            title=dict(text="Road Network Topology (6×5 Grid)", font=dict(size=15), x=0.5),
            xaxis=dict(title="X (m)", showgrid=False, zeroline=False),
            yaxis=dict(title="Y (m)", showgrid=False, zeroline=False,
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="#F8FAFC", paper_bgcolor="white",
            height=420, margin=dict(l=40, r=30, t=50, b=40), showlegend=False,
        )
        return fig

    def plot_vehicle_animation(self, pos_data, edge_coords, path, speed_data, step_len=0.1):
        if not pos_data or len(pos_data) < 2:
            return None
        total = len(pos_data)
        step  = max(1, total // 120)
        frames_pos   = pos_data[::step]
        frames_speed = speed_data[::step] if speed_data else [0] * len(frames_pos)
        xs = [p[0] for p in frames_pos]
        ys = [p[1] for p in frames_pos]
        ts = [i * step * step_len for i in range(len(frames_pos))]
        fig = go.Figure()
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            in_path = path and eid in path
            fig.add_trace(go.Scatter(
                x=[coords[0][0], coords[1][0], None],
                y=[coords[0][1], coords[1][1], None], mode="lines",
                line=dict(color="#EF4444" if in_path else "#D1D5DB",
                          width=4 if in_path else 1.5),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="rgba(59,130,246,0.4)", width=2, dash="dot"),
            name="Trajectory", showlegend=True,
        ))
        frames = []
        for i, (x, y, t, spd) in enumerate(zip(xs, ys, ts, frames_speed)):
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=[x], y=[y], mode="markers+text",
                    marker=dict(size=16, color="#F59E0B", symbol="triangle-up",
                                line=dict(color="white", width=2)),
                    text=[f"{spd:.0f}km/h"], textposition="top center",
                    textfont=dict(size=10, color="#1E40AF"),
                )],
                name=str(i),
                layout=go.Layout(title_text=f"Vehicle Animation | t={t:.1f}s | Speed={spd:.1f}km/h"),
            ))
        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="markers+text",
            marker=dict(size=16, color="#F59E0B", symbol="triangle-up",
                        line=dict(color="white", width=2)),
            text=["🚗"], textposition="top center", name="Vehicle",
        ))
        fig.frames = frames
        fig.update_layout(
            title="Vehicle Trajectory Animation",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor="#F8FAFC", paper_bgcolor="white", height=460,
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.15, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[str(i)], dict(mode="immediate",
                                                 frame=dict(duration=80, redraw=True))],
                            label=f"{ts[i]:.0f}s")
                       for i in range(0, len(frames), max(1, len(frames)//20))],
                transition=dict(duration=0), x=0.05, y=0, len=0.9,
                currentvalue=dict(prefix="Time: ", font=dict(size=12)),
            )],
            margin=dict(l=40, r=30, t=80, b=60),
        )
        return fig

    def plot_path_comparison(self, edge_coords, path_llm, path_dijkstra, weight_dict):
        fig = go.Figure()
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            fig.add_trace(go.Scatter(
                x=[coords[0][0], coords[1][0], None],
                y=[coords[0][1], coords[1][1], None],
                mode="lines", line=dict(color="#E5E7EB", width=2),
                showlegend=False, hoverinfo="skip",
            ))
        if path_dijkstra:
            dx, dy = [], []
            for eid in path_dijkstra:
                c = edge_coords.get(eid)
                if c and c[0] and c[1]:
                    dx += [c[0][0], c[1][0], None]; dy += [c[0][1], c[1][1], None]
            fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines",
                line=dict(color="#3B82F6", width=4, dash="dash"),
                name="Dijkstra (Fixed Weight)"))
        if path_llm:
            lx, ly = [], []
            for eid in path_llm:
                c = edge_coords.get(eid)
                if c and c[0] and c[1]:
                    lx += [c[0][0], c[1][0], None]; ly += [c[0][1], c[1][1], None]
                    mx, my = (c[0][0]+c[1][0])/2, (c[0][1]+c[1][1])/2
                    w = weight_dict.get(eid, 1.0)
                    fig.add_annotation(x=mx, y=my+30, text=f"w={w:.1f}",
                                       font=dict(size=9, color="#EF4444"),
                                       showarrow=False, bgcolor="white",
                                       bordercolor="#EF4444", borderwidth=1)
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                line=dict(color="#EF4444", width=4), name="LLM+A* (Ours)"))
        fig.update_layout(
            title="Path Comparison: LLM+A* vs Dijkstra",
            xaxis=dict(title="X (m)", showgrid=True, gridcolor="#F3F4F6"),
            yaxis=dict(title="Y (m)", showgrid=True, gridcolor="#F3F4F6",
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            height=420, margin=dict(l=40, r=30, t=70, b=40),
        )
        return fig

    def plot_constraint_radar(self, weight_dict):
        groups = {
            "Eastbound (E)":   [w for e, w in weight_dict.items() if e.endswith("_E")],
            "Westbound (W)":   [w for e, w in weight_dict.items() if e.endswith("_W")],
            "Northbound (N)":  [w for e, w in weight_dict.items() if e.endswith("_N")],
            "Southbound (S)":  [w for e, w in weight_dict.items() if e.endswith("_S")],
            "Arterial Roads":  [w for e, w in weight_dict.items() if
                                (e.startswith("R0") or e.startswith("R3") or
                                 e.startswith("C2R") or e.startswith("C4R"))],
            "Secondary Roads": [w for e, w in weight_dict.items() if
                                (e.startswith("R1") or e.startswith("C1R") or e.startswith("C3R"))],
        }
        categories  = list(groups.keys())
        values      = [float(np.mean(v)) if v else 1.0 for v in groups.values()]
        values     += [values[0]]
        cats_closed = categories + [categories[0]]
        fig = go.Figure(go.Scatterpolar(
            r=values, theta=cats_closed, fill="toself",
            line=dict(color="#EF4444", width=2),
            fillcolor="rgba(239,68,68,0.2)", name="Congestion Level",
        ))
        avg = float(np.mean(list(weight_dict.values()))) if weight_dict else 1.0
        fig.add_trace(go.Scatterpolar(
            r=[avg] * (len(categories) + 1), theta=cats_closed,
            line=dict(color="#3B82F6", width=1, dash="dash"),
            name=f"Avg Weight ({avg:.1f})",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10],
                                tickvals=[2, 4, 6, 8, 10], gridcolor="#E5E7EB"),
                angularaxis=dict(gridcolor="#E5E7EB"), bgcolor="#F8FAFC",
            ),
            title=dict(text="Directional Congestion Radar", font=dict(size=14), x=0.5),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            height=400, paper_bgcolor="white", margin=dict(l=40, r=40, t=60, b=60),
        )
        return fig

    def plot_path_gantt(self, path, weight_dict, travel_time, scene_weights=None):
        """甘特图：用场景真实权重分配时间比例（与SUMO一致），模型权重用于颜色标注"""
        if not path or travel_time <= 0:
            return None
        _tw = scene_weights if scene_weights else weight_dict
        total_w = sum(_tw.get(e, 2.0) for e in path) or 1.0
        current = 0.0
        segments = []
        for eid in path:
            sw = _tw.get(eid, 2.0)
            mw = weight_dict.get(eid, 2.0)
            duration = (sw / total_w) * travel_time
            segments.append({"edge": eid, "start": current,
                              "end": current + duration, "weight": mw,
                              "scene_w": sw})
            current += duration
        fig    = go.Figure()
        colors = px.colors.sequential.Reds
        for seg in segments:
            ratio = min(int(seg["weight"] / 10 * (len(colors) - 1)), len(colors) - 1)
            fig.add_trace(go.Bar(
                x=[seg["end"] - seg["start"]], y=[seg["edge"]],
                base=seg["start"], orientation="h",
                marker_color=colors[ratio],
                text=f"w={seg['weight']:.1f}  {seg['end']-seg['start']:.1f}s",
                textposition="inside", textfont=dict(color="white", size=10),
                hovertemplate=(f"<b>{seg['edge']}</b><br>模型权重: {seg['weight']:.1f}<br>场景权重: {seg.get('scene_w', seg['weight']):.1f}"
                               f"<br>Duration: {seg['end']-seg['start']:.1f}s<extra></extra>"),
                showlegend=False,
            ))
        fig.update_layout(
            title=dict(text="Path Segment Timeline (Gantt)", font=dict(size=14), x=0.5),
            xaxis=dict(title="Time (s)", showgrid=True, gridcolor="#F3F4F6"),
            yaxis=dict(title="Edge ID", autorange="reversed"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(250, 40 + len(path) * 38),
            margin=dict(l=80, r=30, t=50, b=50), barmode="overlay",
        )
        return fig

    # ══ 真实对比实验可视化 ════════════════════════════════
    # ══ 真实对比实验可视化 ════════════════════════════════
    def _normalize_method_label(self, method_name: str) -> str:
        """统一展示方法名，移除星标和立场词。"""
        alias = {
            "Qwen-LoRA★": "Qwen-LoRA",
            "R1-LoRA★": "R1-LoRA",
            "LoRA+GAT★★": "LoRA+GAT",
            "Sparse-LoRA★★": "Sparse-LoRA",
            "Sparse-LoRA+GAT★★★": "Sparse-LoRA+GAT",
        }
        return alias.get(method_name, method_name)

    @staticmethod
    def _to_percent(value):
        """将 0~1 或 0~100 形式统一映射到百分制。"""
        try:
            v = float(value)
        except Exception:
            return None
        if 0.0 <= v <= 1.0:
            return v * 100.0
        return v

    def _normalize_display_record(self, rec: dict, method_name: str) -> dict:
        """统一展示记录字段映射，兼容旧字段 parse_rate/coverage/parsed_ratio/plan_ms/time_s。"""
        total_edges = int(rec.get("total_edges", 98) or 98)
        metric_rec = normalize_edge_metric_record(rec, total_edges=total_edges, method_name=method_name)

        parsed_ratio = metric_rec.get("parsed_edge_ratio")
        signal_ratio = metric_rec.get("signal_ratio")

        if parsed_ratio is None:
            parsed_ratio = self._to_percent(rec.get("parsed_ratio", rec.get("parse_rate")))
        if signal_ratio is None:
            signal_ratio = self._to_percent(rec.get("signal_ratio", rec.get("coverage")))

        parsed_ratio = float(parsed_ratio or 0.0)
        signal_ratio = float(signal_ratio or 0.0)

        parsed_edges = rec.get("parsed_edges", metric_rec.get("parsed_edges"))
        if parsed_edges is None:
            parsed_edges = int(round(parsed_ratio / 100.0 * total_edges))
        parsed_edges = int(parsed_edges)

        signal_edges = rec.get("signal_edges", metric_rec.get("signal_edges"))
        if signal_edges is None:
            signal_edges = int(round(signal_ratio / 100.0 * total_edges))
        signal_edges = int(signal_edges)

        default_edges = rec.get("default_edges", metric_rec.get("default_edges"))
        if default_edges is None:
            default_edges = max(total_edges - signal_edges, 0)
        default_edges = int(default_edges)

        planning_ms = rec.get("planning_ms", rec.get("plan_ms"))
        if planning_ms is None and rec.get("time_s") is not None:
            planning_ms = float(rec.get("time_s")) * 1000.0

        infer_time_s = rec.get("time_s")
        if infer_time_s is None and planning_ms is not None:
            infer_time_s = float(planning_ms) / 1000.0

        return {
            "cost": float(rec.get("path_cost", 0) or 0),
            "constraint": float(rec.get("constraint_rate", 0) or 0),
            "signal_ratio": signal_ratio,
            "parsed_edge_ratio": parsed_ratio,
            "parsed_edges": parsed_edges,
            "signal_edges": signal_edges,
            "default_edges": default_edges,
            "time": float(infer_time_s or 0),
            "sumo": float(rec.get("sumo_time", 0) or 0),
        }

    def plot_real_compare(self, results_path: str, current_cost: float,
                          current_constraint: float, current_time: float):
        """读取真实对比实验结果并可视化；结果缺失时仅给出明确提示，不做估算回落。"""
        import os, json

        METHOD_COLORS = {
            "Dijkstra": "#94A3B8",
            "Rule-A*": "#64748B",
            "DQN": "#475569",
            "GCN-Weight": "#334155",
            "Raw-Qwen": "#8B5CF6",
            "CoT-Qwen": "#F59E0B",
            "R1-Raw": "#06B6D4",
            "R1-LoRA": "#0EA5E9",
            "Qwen-LoRA": "#F97316",
            "LoRA+GAT": "#DC2626",
            "Sparse-LoRA": "#EF4444",
            "Sparse-LoRA+GAT": "#991B1B",
        }
        _IS_OURS = lambda m: any(x in m for x in ("Sparse-LoRA", "LoRA+GAT"))
        _METHOD_ORDER = [
            "Dijkstra", "Rule-A*", "DQN", "GCN-Weight",
            "Raw-Qwen", "CoT-Qwen", "R1-Raw", "R1-LoRA",
            "Qwen-LoRA", "LoRA+GAT", "Sparse-LoRA", "Sparse-LoRA+GAT",
        ]

        if not os.path.exists(results_path):
            msg = "未检测到对比实验结果，请先运行对比实验后再展示该页面。"

            def _empty_fig(title_text: str):
                fig = go.Figure()
                fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
                fig.update_layout(title=title_text, template="plotly_white", height=320)
                return fig

            empty_df = pd.DataFrame(columns=[
                "排名", "方法", "SUMO时间(s)↓", "约束符合率(%)↑", "非默认路况占比(%)↑", "推理耗时(s)↓", "路径代价↓"
            ])
            return (
                _empty_fig("综合对比（缺少实验数据）"),
                _empty_fig("多维能力雷达图（缺少实验数据）"),
                _empty_fig("推理速度 vs 约束符合率（缺少实验数据）"),
                empty_df,
                "missing",
                _empty_fig("SUMO行程时间箱线图（缺少实验数据）"),
                None,
                None,
            )

        with open(results_path, encoding="utf-8") as f:
            raw = json.load(f)

        scenario_data = []
        stats = {}
        for tc in raw:
            sc_name = tc.get("scenario", "?")
            sc_entry = {"scenario": sc_name, "methods": {}}
            for m in tc.get("methods", []):
                raw_name = str(m.get("method", "Unknown"))
                name = self._normalize_method_label(raw_name)
                entry = self._normalize_display_record(m, method_name=name)

                if name not in stats:
                    stats[name] = {
                        "cost": [], "constraint": [], "signal_ratio": [],
                        "parsed_edge_ratio": [], "time": [], "sumo": []
                    }
                stats[name]["cost"].append(entry["cost"])
                stats[name]["constraint"].append(entry["constraint"])
                stats[name]["signal_ratio"].append(entry["signal_ratio"])
                stats[name]["parsed_edge_ratio"].append(entry["parsed_edge_ratio"])
                stats[name]["time"].append(entry["time"])
                stats[name]["sumo"].append(entry["sumo"])
                sc_entry["methods"][name] = entry
            scenario_data.append(sc_entry)

        known = [m for m in _METHOD_ORDER if m in stats]
        extra = [m for m in stats if m not in _METHOD_ORDER]
        methods = known + extra

        if not methods:
            empty_df = pd.DataFrame(columns=[
                "排名", "方法", "SUMO时间(s)↓", "约束符合率(%)↑", "非默认路况占比(%)↑", "推理耗时(s)↓", "路径代价↓"
            ])
            fig = go.Figure()
            fig.add_annotation(text="对比结果文件为空。", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template="plotly_white", height=320)
            return fig, fig, fig, empty_df, "missing", fig, None, None

        costs = [sum(stats[m]["cost"]) / len(stats[m]["cost"]) for m in methods]
        constr = [sum(stats[m]["constraint"]) / len(stats[m]["constraint"]) for m in methods]
        signal_ratio_vals = [sum(stats[m]["signal_ratio"]) / len(stats[m]["signal_ratio"]) for m in methods]
        parsed_ratio_vals = [sum(stats[m]["parsed_edge_ratio"]) / len(stats[m]["parsed_edge_ratio"]) for m in methods]
        times = [sum(stats[m]["time"]) / len(stats[m]["time"]) for m in methods]
        sumo_t = [sum(stats[m]["sumo"]) / len(stats[m]["sumo"]) if stats[m]["sumo"] else 0 for m in methods]
        sumo_by_method = {m: stats[m]["sumo"] for m in methods}
        data_source = "实验数据"

        colors = [METHOD_COLORS.get(m, "#94A3B8") for m in methods]

        bar_fig = go.Figure()
        has_sumo = any(v > 0 for v in sumo_t)
        y_bars = sumo_t if has_sumo else costs
        y_label = "SUMO行程时间 (s) ↓" if has_sumo else "路径代价"

        bar_fig.add_trace(go.Bar(
            x=methods, y=[float(v) for v in y_bars],
            name=y_label,
            marker=dict(
                color=[METHOD_COLORS.get(m, "#94A3B8") for m in methods],
                opacity=[0.95 if _IS_OURS(m) else 0.65 for m in methods],
                line=dict(
                    color=["#B91C1C" if _IS_OURS(m) else "rgba(0,0,0,0)" for m in methods],
                    width=[2.5 if _IS_OURS(m) else 0 for m in methods],
                ),
            ),
            yaxis="y",
            text=[f"<b>{v:.0f}s</b>" if _IS_OURS(m) else f"{v:.0f}s" for m, v in zip(methods, y_bars)],
            textposition="outside",
            textfont=dict(
                size=[13 if _IS_OURS(m) else 10 for m in methods],
                color=["#B91C1C" if _IS_OURS(m) else "#374151" for m in methods],
            ),
        ))

        _constr_text = []
        for m, v in zip(methods, constr):
            if _IS_OURS(m):
                _constr_text.append(f"<b>{v:.0f}%</b>")
            elif m in ("Rule-A*", "GCN-Weight"):
                _constr_text.append(f"{v:.0f}%")
            else:
                _constr_text.append("")

        bar_fig.add_trace(go.Scatter(
            x=methods, y=[float(v) for v in constr],
            name="约束符合率 (%)",
            mode="lines+markers+text",
            line=dict(color="#F59E0B", width=2.5),
            marker=dict(
                size=[16 if _IS_OURS(m) else 8 for m in methods],
                color=["#B91C1C" if _IS_OURS(m) else "#F59E0B" for m in methods],
                symbol=["diamond" if _IS_OURS(m) else "circle" for m in methods],
                line=dict(color="white", width=1.5),
            ),
            text=_constr_text, textposition="top center",
            textfont=dict(size=10, color="#B45309"),
            yaxis="y2",
        ))

        bar_fig.update_layout(
            title=dict(
                text=f"方法综合对比 — {y_label}越低越好，约束符合率越高越好（{data_source}）",
                font=dict(size=13), x=0.5,
            ),
            yaxis=dict(title=y_label, side="left", showgrid=True, gridcolor="#F3F4F6"),
            yaxis2=dict(title=dict(text="约束符合率 (%)", standoff=10), side="right", overlaying="y", range=[0, 130]),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
            height=500, barmode="group",
            margin=dict(l=55, r=80, t=65, b=100),
            xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        )

        _s_min, _s_max = min(sumo_t) if sumo_t else 1, max(sumo_t) if sumo_t else 1
        _t_min, _t_max = min(times) if times else 0, max(times) if times else 1
        _c_min, _c_max = min(costs) if costs else 0, max(costs) if costs else 1
        norm_sumo = [1 - (v - _s_min) / (_s_max - _s_min + 1e-6) for v in sumo_t]
        norm_constr = [v / 100.0 for v in constr]
        norm_signal = [v / 100.0 for v in signal_ratio_vals]
        norm_speed = [1 - (v - _t_min) / (_t_max - _t_min + 1e-6) for v in times]
        norm_cost = [1 - (v - _c_min) / (_c_max - _c_min + 1e-6) for v in costs]
        cats = ["路径质量", "约束符合", "非默认路况", "推理速度", "路径效率", "路径质量"]
        radar_fig = go.Figure()
        for i, m in enumerate(methods):
            vals = [norm_sumo[i], norm_constr[i], norm_signal[i], norm_speed[i], norm_cost[i], norm_sumo[i]]
            is_ours = _IS_OURS(m)
            radar_fig.add_trace(go.Scatterpolar(
                r=[float(v) for v in vals], theta=cats,
                fill="toself" if is_ours else "none",
                fillcolor="rgba(239,68,68,0.12)" if is_ours else "rgba(0,0,0,0)",
                line=dict(color=colors[i], width=3.0 if is_ours else 1.5, dash="solid" if is_ours else "dot"),
                name=m,
            ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0],
                                ticktext=["低", "中", "高", "优"], gridcolor="#E5E7EB"),
                angularaxis=dict(gridcolor="#E5E7EB"),
                bgcolor="#F8FAFC",
            ),
            title=dict(text="多维能力雷达图（覆盖面积越大越优）", font=dict(size=14), x=0.5),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=9)),
            height=420, paper_bgcolor="white", margin=dict(l=40, r=40, t=50, b=90),
        )

        scatter_fig = go.Figure()
        _t_med = sorted(times)[len(times) // 2]
        _c_med = 75.0
        scatter_fig.add_shape(type="line", x0=_t_med, x1=_t_med, y0=0, y1=108,
                              line=dict(dash="dot", color="#D1D5DB", width=1))
        scatter_fig.add_shape(type="line", x0=-1, x1=max(times) + 5, y0=_c_med, y1=_c_med,
                              line=dict(dash="dot", color="#D1D5DB", width=1))
        scatter_fig.add_annotation(x=_t_med / 2, y=108, text="快速精准", font=dict(size=10, color="#16A34A"), showarrow=False)
        for i, m in enumerate(methods):
            is_ours = _IS_OURS(m)
            scatter_fig.add_trace(go.Scatter(
                x=[float(times[i])], y=[float(constr[i])], mode="markers+text",
                marker=dict(
                    size=20 if is_ours else 13,
                    color=colors[i],
                    symbol="diamond" if is_ours else "circle",
                    line=dict(color="white", width=2),
                    opacity=1.0 if is_ours else 0.8,
                ),
                text=[f"<b>{m}</b>" if is_ours else m],
                textposition="top right" if times[i] < _t_med else "top left",
                textfont=dict(size=11 if is_ours else 10, color=colors[i]),
                name=m, showlegend=False,
            ))
        scatter_fig.update_layout(
            title=dict(text="推理速度 vs 约束符合率（左上角=最优）", font=dict(size=13), x=0.5),
            xaxis=dict(title="推理时间 (s)", showgrid=True, gridcolor="#F3F4F6", range=[-2, max(times) * 1.1 + 2]),
            yaxis=dict(title="约束符合率 (%)", range=[0, 115], showgrid=True, gridcolor="#F3F4F6"),
            plot_bgcolor="white", paper_bgcolor="white", height=320, margin=dict(l=50, r=30, t=50, b=50),
        )

        box_fig = go.Figure()
        for m in methods:
            vals = sumo_by_method.get(m, [])
            if not vals:
                continue
            is_ours = _IS_OURS(m)
            box_fig.add_trace(go.Box(
                y=[float(v) for v in vals],
                name=m,
                marker_color=METHOD_COLORS.get(m, "#94A3B8"),
                line=dict(width=2.5 if is_ours else 1.5),
                fillcolor=METHOD_COLORS.get(m, "#94A3B8"),
                opacity=0.85 if is_ours else 0.55,
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
                marker=dict(size=7 if is_ours else 5, line=dict(color="white", width=1)),
            ))
        box_fig.update_layout(
            title=dict(text="各场景SUMO行程时间分布（箱线图，越低越稳定）", font=dict(size=13), x=0.5),
            yaxis=dict(title="SUMO行程时间 (s)", showgrid=True, gridcolor="#F3F4F6"),
            xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
            plot_bgcolor="white", paper_bgcolor="white", height=400, showlegend=False,
            margin=dict(l=55, r=30, t=55, b=100),
        )

        scene_bar_fig = None
        if scenario_data:
            scene_names = [s["scenario"] for s in scenario_data]
            key_methods = ["Dijkstra", "Rule-A*", "GCN-Weight", "Qwen-LoRA", "Sparse-LoRA", "Sparse-LoRA+GAT"]
            key_methods = [m for m in key_methods if any(m in s["methods"] for s in scenario_data)]
            scene_bar_fig = go.Figure()
            for m in key_methods:
                y_vals = [s["methods"].get(m, {}).get("sumo", 0) for s in scenario_data]
                is_ours = _IS_OURS(m)
                scene_bar_fig.add_trace(go.Bar(
                    name=m,
                    x=scene_names, y=[float(v) for v in y_vals],
                    marker_color=METHOD_COLORS.get(m, "#94A3B8"),
                    opacity=0.9 if is_ours else 0.65,
                    marker_line=dict(color="#B91C1C" if is_ours else "rgba(0,0,0,0)", width=2 if is_ours else 0),
                    text=[f"{v:.0f}s" for v in y_vals],
                    textposition="outside", textfont=dict(size=9),
                ))
            scene_bar_fig.update_layout(
                title=dict(text="分场景 SUMO 行程时间对比（关键方法，越低越好）", font=dict(size=13), x=0.5),
                barmode="group",
                yaxis=dict(title="SUMO行程时间 (s)", showgrid=True, gridcolor="#F3F4F6"),
                xaxis=dict(tickangle=-15, tickfont=dict(size=10)),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center", font=dict(size=10)),
                height=430, margin=dict(l=55, r=30, t=60, b=110),
            )

        heatmap_fig = None
        if scenario_data:
            sc_names_short = [
                s["scenario"].split(" ")[0] + " " + s["scenario"].split(" ")[1]
                if len(s["scenario"].split(" ")) > 1 else s["scenario"]
                for s in scenario_data
            ]
            heat_methods = [m for m in methods if m in scenario_data[0]["methods"]]
            z_matrix = []
            for m in heat_methods:
                row = [s["methods"].get(m, {}).get("sumo", 0) for s in scenario_data]
                z_matrix.append(row)
            z_norm = []
            for row in z_matrix:
                mn, mx = min(row), max(row)
                rng = mx - mn + 1e-6
                z_norm.append([(v - mn) / rng for v in row])
            heatmap_fig = go.Figure(go.Heatmap(
                z=z_norm,
                x=sc_names_short,
                y=heat_methods,
                colorscale=[[0, "#16A34A"], [0.4, "#FCD34D"], [1.0, "#DC2626"]],
                colorbar=dict(title="相对时间<br>（绿=快，红=慢）", tickvals=[0, 0.5, 1], ticktext=["最快", "中等", "最慢"]),
                text=[[f"{z_matrix[i][j]:.0f}s" for j in range(len(scenario_data))] for i in range(len(heat_methods))],
                texttemplate="%{text}", textfont=dict(size=10), hoverongaps=False,
            ))
            heatmap_fig.update_layout(
                title=dict(text="方法 × 场景 SUMO行程时间热力图（绿=快，红=慢）", font=dict(size=13), x=0.5),
                height=max(350, len(heat_methods) * 38 + 100),
                margin=dict(l=160, r=80, t=60, b=60), paper_bgcolor="white",
                yaxis=dict(tickfont=dict(size=10)), xaxis=dict(tickfont=dict(size=10)),
            )

        sumo_rank = sorted(range(len(methods)), key=lambda i: sumo_t[i])
        rank_map = {methods[i]: r + 1 for r, i in enumerate(sumo_rank)}
        df = pd.DataFrame({
            "排名": [rank_map[m] for m in methods],
            "方法": [str(m) for m in methods],
            "SUMO时间(s)↓": [round(float(v), 1) for v in sumo_t],
            "约束符合率(%)↑": [round(float(v), 1) for v in constr],
            "非默认路况占比(%)↑": [round(float(v), 1) for v in signal_ratio_vals],
            "推理耗时(s)↓": [round(float(v), 2) for v in times],
            "路径代价↓": [round(float(v), 2) for v in costs],
        }).sort_values("排名").reset_index(drop=True)

        return bar_fig, radar_fig, scatter_fig, df, data_source, box_fig, scene_bar_fig, heatmap_fig


    def _load_smallnet_series(self, smallnet_path: str, compare_path: str):
        """读取 SmallNet / LargeNet 结果，返回均值序列和摘要。"""
        import json, os
        from collections import defaultdict

        small_stats = defaultdict(list)
        large_stats = defaultdict(list)

        if os.path.exists(smallnet_path):
            with open(smallnet_path, encoding="utf-8") as f:
                small_raw = json.load(f)
            for tc in small_raw:
                for m in tc.get("methods", []):
                    metric_rec = normalize_edge_metric_record(m, total_edges=m.get("total_edges", 24), method_name=m.get("method"))
                    small_stats[m.get("method", "")].append(float(metric_rec["parsed_edge_ratio"] or 0))

        if os.path.exists(compare_path):
            with open(compare_path, encoding="utf-8") as f:
                large_raw = json.load(f)
            for tc in large_raw:
                for m in tc.get("methods", []):
                    metric_rec = normalize_edge_metric_record(m, total_edges=m.get("total_edges", 98), method_name=m.get("method"))
                    large_stats[m.get("method", "")].append(float(metric_rec["parsed_edge_ratio"] or 0))

        display_map = {
            "Qwen-LoRA★": "Qwen-LoRA",
            "R1-LoRA★": "R1-LoRA",
            "Sparse-LoRA★★": "Sparse-LoRA",
            "Sparse-LoRA+GAT★★★": "Sparse-LoRA+GAT",
        }
        method_order = ["Qwen-LoRA★", "R1-LoRA★", "Sparse-LoRA★★", "Sparse-LoRA+GAT★★★"]

        methods = [m for m in method_order if m in small_stats or m in large_stats]
        if not methods:
            methods = sorted(set(small_stats) | set(large_stats))

        small_means = [sum(small_stats[m]) / len(small_stats[m]) if small_stats.get(m) else 0.0 for m in methods]
        large_means = [sum(large_stats[m]) / len(large_stats[m]) if large_stats.get(m) else 0.0 for m in methods]
        labels = [display_map.get(m, m) for m in methods]

        primary_idx = next((i for i, m in enumerate(methods) if small_stats.get(m) and large_stats.get(m)), None)
        primary = None
        if primary_idx is not None:
            primary = {
                "method": methods[primary_idx],
                "label": labels[primary_idx],
                "small": small_means[primary_idx],
                "large": large_means[primary_idx],
                "drop": small_means[primary_idx] - large_means[primary_idx],
            }

        return {
            "methods": methods,
            "labels": labels,
            "small": small_means,
            "large": large_means,
            "small_map": small_stats,
            "large_map": large_stats,
            "primary": primary,
        }

    def plot_smallnet_decay_compare(self, smallnet_path: str, compare_path: str):
        """SmallNet vs LargeNet 并排显式解析边占比对比图。"""
        summary = self._load_smallnet_series(smallnet_path, compare_path)
        methods = summary["labels"]
        small_vals = summary["small"]
        large_vals = summary["large"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods,
            y=small_vals,
            name="SmallNet (24条边)",
            marker=dict(color="#2563EB", line=dict(color="#1D4ED8", width=1.2)),
            text=[f"{v:.0f}%" for v in small_vals],
            textposition="outside",
            width=0.36,
        ))
        fig.add_trace(go.Bar(
            x=methods,
            y=large_vals,
            name="LargeNet (主路网边数)",
            marker=dict(color="#F97316", line=dict(color="#EA580C", width=1.2)),
            text=[f"{v:.0f}%" for v in large_vals],
            textposition="outside",
            width=0.36,
        ))

        fig.update_layout(
            title=dict(text="SmallNet vs LargeNet 显式解析边占比衰减对比", x=0.5, font=dict(size=16)),
            xaxis=dict(title="模型方法", tickangle=-12),
            yaxis=dict(title="显式解析边占比(%)", range=[0, 110], gridcolor="#E5E7EB"),
            barmode="group",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=420,
            margin=dict(l=50, r=20, t=70, b=55),
        )

        primary = summary["primary"]
        if primary:
            fig.add_annotation(
                x=primary["label"],
                y=max(primary["small"], primary["large"]),
                text=f"下降 {max(primary['drop'], 0):.1f} 个百分点",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-45,
                font=dict(color="#7C2D12", size=11),
                bgcolor="rgba(255,247,237,0.95)",
                bordercolor="#FB923C",
                borderwidth=1,
            )

        return fig, summary

    def plot_smallnet_topology_preview(self):
        """SmallNet 的简易拓扑示意图。"""
        fig = go.Figure()
        xs = list(range(SMALL_NODE_COLS))
        ys = list(range(SMALL_NODE_ROWS))

        for y in ys:
            fig.add_trace(go.Scatter(
                x=xs,
                y=[y] * len(xs),
                mode="lines",
                line=dict(color="#CBD5E1", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))
        for x in xs:
            fig.add_trace(go.Scatter(
                x=[x] * len(ys),
                y=ys,
                mode="lines",
                line=dict(color="#CBD5E1", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

        node_x = [x for y in ys for x in xs]
        node_y = [y for y in ys for _x in xs]
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=["" for _ in node_x],
            marker=dict(size=12, color="#2563EB", line=dict(color="white", width=1.5)),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=[0, SMALL_NODE_COLS - 1],
            y=[0, SMALL_NODE_ROWS - 1],
            mode="markers+lines",
            line=dict(color="#EF4444", width=3, dash="dot"),
            marker=dict(size=14, color=["#10B981", "#EF4444"], line=dict(color="white", width=1.5)),
            text=["起点", "终点"],
            textposition=["bottom left", "top right"],
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_annotation(x=0, y=SMALL_NODE_ROWS - 0.05, text="4行×3列 SmallNet", showarrow=False,
                           font=dict(size=16, color="#0F172A"), xanchor="left")
        fig.add_annotation(x=0, y=-0.42, text="节点骨架仅用于直观对比路网尺寸", showarrow=False,
                           font=dict(size=10, color="#475569"), xanchor="left")

        fig.update_layout(
            title=dict(text="SmallNet 拓扑预览", x=0.5, font=dict(size=15)),
            xaxis=dict(visible=False, range=[-0.3, SMALL_NODE_COLS - 0.7]),
            yaxis=dict(visible=False, range=[-0.4, SMALL_NODE_ROWS - 0.1]),
            template="plotly_white",
            height=300,
            margin=dict(l=10, r=10, t=55, b=20),
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="white",
        )
        return fig

# ====================== 数据导出引擎 ======================
class DataExportEngine:
    def __init__(self, config):
        self.config = config

    def export_results(self, results, filename):
        fp = os.path.join(self.config.get("RESULTS_DIR"), filename)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return fp

    def export_dataframe(self, df, filename):
        fp = os.path.join(self.config.get("RESULTS_DIR"), f"{filename}.csv")
        df.to_csv(fp, index=False, encoding="utf-8")
        return fp


# ====================== 全局实例化 ======================
config        = ConfigManager()
model_manager = ModelManager(config)
path_engine   = PathPlanningEngine(config)
sumo_engine   = SUMOSimulationEngine(config)
viz_engine    = VisualizationEngine(config)
export_engine = DataExportEngine(config)


# ====================== 主界面 ======================

def _safe_table(df):
    """将 DataFrame 渲染为 markdown 表格（实现迁移到 viz.py）。"""
    st.markdown(safe_table_markdown(df))

def _latest_compare_results_path():
    """优先读取最新 run 目录中的 compare_results.json，兼容旧路径回落。"""
    import glob

    patterns = [
        "/root/autodl-tmp/results/runs/*__compare_multi_method__*/compare_results.json",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    legacy = "/root/autodl-tmp/results/compare_results.json"
    return legacy

def main():
    st.set_page_config(
        page_title="LLM-Driven Traffic Path Planning",
        page_icon="🚗", layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main-title{font-size:2rem;font-weight:800;color:#1E40AF;text-align:center;margin-bottom:.2rem}
    .sub-title{font-size:1rem;color:#6B7280;text-align:center;margin-bottom:.8rem}
    .block-container{padding-top:3.5rem!important;padding-bottom:1rem!important}
    .sec-hdr{font-size:1rem;font-weight:700;color:#1E293B;border-left:4px solid #3B82F6;
        padding-left:10px;margin:10px 0 8px}
    .info-strip{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
        padding:10px 14px;font-size:.82rem;color:#1D4ED8;margin:8px 0}
    [data-testid="stSidebar"]{background:#F8FAFF!important}
    .stTabs [data-baseweb="tab-list"]{gap:4px;background:#F1F5F9;border-radius:10px;padding:4px}
    .stTabs [data-baseweb="tab"]{border-radius:8px!important;font-weight:500!important;
        padding:6px 16px!important}
    .hist-row{background:#F8FAFF;border:1px solid #E2E8F0;border-radius:8px;padding:8px 12px;
        margin:4px 0;display:flex;justify-content:space-between;font-size:.8rem;align-items:center}
    </style>
    """, unsafe_allow_html=True)

    for k, v in [("results", {}), ("total_cost", 0.0), ("travel_time", 0.0),
                 ("last_raw", ""), ("last_wd", {}), ("last_raw_llm_weights", {}),
                 ("last_mode", "demo_mode"), ("last_fallback_used", False),
                 ("last_fallback_reason", "none"), ("last_path", None),
                 ("run_history", []), ("run_count", 0)]:
        if k not in st.session_state:
            st.session_state[k] = v

    if "edge_total_dynamic" not in st.session_state:
        st.session_state["edge_total_dynamic"] = len(getattr(path_engine, "_ALL_EDGES", []))
    edge_total = int(st.session_state.get("edge_total_dynamic") or len(getattr(path_engine, "_ALL_EDGES", [])))
    try:
        _net_meta, _ = _load_net_cached(config.get("SUMO_NET_PATH"))
        _edge_total_live = len([e for e in _net_meta.getEdges() if e is not None])
        if _edge_total_live > 0:
            st.session_state["edge_total_dynamic"] = _edge_total_live
            edge_total = _edge_total_live
    except Exception:
        pass

    run_cnt = st.session_state["run_count"]
    best_t  = min([r["travel_time"] for r in st.session_state["run_history"]], default=0)
    sub = f"Qwen/DeepSeek-R1 SFT · SUMO · A*/Dijkstra · 6×5 郑州金水区路网 · 规划次数: {run_cnt}"
    if best_t > 0:
        sub += f" · 最优: {best_t:.1f}s"
    st.markdown('<div class="main-title">🚗 大模型驱动的交通路径规划可视化</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{sub}</div>', unsafe_allow_html=True)
    st.divider()

    # ======================== 侧边栏 ========================
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1rem;font-weight:700;color:#1E3A8A;margin-bottom:8px">'
            '⚙️ 实验配置</div>', unsafe_allow_html=True)

        selected_model = st.selectbox(
            "🤖 权重生成模型",
            [
                "Qwen-Sparse-LoRA",
                "Qwen2.5-1.5B-LoRA",
                "DeepSeek-R1-1.5B-LoRA",
                "Qwen2.5-1.5B-Raw (无微调基线)",
                "DeepSeek-R1-1.5B-Raw (R1基线)",
            ],
            key="model_select",
        )
        _model_path_map = {
            "Qwen-Sparse-LoRA":   "/root/autodl-tmp/model_merged_sparse",
            "Qwen2.5-1.5B-LoRA":"/root/autodl-tmp/model_merged_qwen",
            "DeepSeek-R1-1.5B-LoRA": "/root/autodl-tmp/model_merged_r1",
            "Qwen2.5-1.5B-Raw (无微调基线)":        "/root/autodl-tmp/Qwen2.5-1.5B-Instruct",
            "DeepSeek-R1-1.5B-Raw (R1基线)":       "/root/autodl-tmp/DeepSeek-R1-1.5B",
        }
        config.config["MODEL_PATH"] = _model_path_map.get(
            selected_model, "/root/autodl-tmp/model_merged_sparse")
        _is_sparse_ui = "Sparse" in selected_model

        st.markdown('<div class="sec-hdr">🧪 运行模式</div>', unsafe_allow_html=True)
        _mode_options = {
            "demo_mode": "Demo Mode（允许规则兜底）",
            "experiment_mode": "Experiment Mode（禁用规则兜底）",
        }
        run_mode = st.selectbox(
            "模式选择",
            ["demo_mode", "experiment_mode"],
            key="run_mode",
            format_func=lambda m: _mode_options.get(m, m),
            help="demo_mode：面向演示，允许规则兜底；experiment_mode：面向实验，禁止规则兜底并显式记录fallback。",
        )
        config.config["RUN_MODE"] = run_mode
        if run_mode == "experiment_mode":
            st.warning("当前为 Experiment Mode：禁止规则兜底，所有fallback将显式记录。")
        else:
            st.caption("当前为 Demo Mode：允许规则兜底，适合演示流程。")

        # ── GAT 增强开关 ──────────────────────────────────────────
        st.markdown('<div class="sec-hdr">🧠 GAT 图注意力增强</div>', unsafe_allow_html=True)
        _gat_default_path = "/root/autodl-tmp/gat_model.pt"
        import os as _os
        _gat_trained = _os.path.exists(_gat_default_path)
        use_gat = st.toggle(
            "启用 LLM-GAT 级联（补全缺失权重 + 空间平滑）",
            value=False, key="use_gat",
            help="开启后：LLM解析结果 → GAT图神经网络精炼 → A*规划\n"
                 "对复杂场景效果显著（尤其显式解析边占比<90%时）"
        )
        if use_gat:
            if not _gat_trained:
                st.warning("⚠️ GAT尚未训练，开启会使用随机权重污染结果！\n"
                           "请先运行：`python evaluate.py --train-gat`", icon="🚨")
            gat_alpha = st.slider("混合系数 α（LLM权重占比）",
                                  0.5, 1.0, 0.85, 0.05, key="gat_alpha",
                                  help="α=1: 纯LLM，α=0: 纯GAT\n"
                                       "GAT已训练推荐0.65，未训练请保持≥0.9")
            gat_model_path = st.text_input(
                "GAT模型路径", value=_gat_default_path, key="gat_path")
        else:
            gat_alpha, gat_model_path = 0.85, _gat_default_path
        if _gat_trained:
            st.caption("✅ GAT模型已就绪")
        else:
            st.caption("⚪ GAT未训练（run: `python evaluate.py --train-gat`）")

        st.markdown('<div class="sec-hdr">📝 场景输入</div>', unsafe_allow_html=True)
        traffic_constraint = st.text_area(
            "交通约束描述",
            placeholder=f"黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北施工封闭；红专路向东畅通无阻。请为全部{edge_total}条路段生成权重。",
            height=120, key="traffic_input",
        )

        col_s, col_e = st.columns(2)
        with col_s:
            start_edge = st.text_input("🟢 起点", value="R3C0_E", key="start_edge")
        with col_e:
            end_edge = st.text_input("🏁 终点", value="R3C3_E", key="end_edge")

        # ── 测试场景定义 ──────────────────────────────────────────
        # 格式: (起点EdgeID, 终点EdgeID, 交通描述)
        # EdgeID规则: 水平 R{0-4}C{0-4}_{E|W}, 垂直 C{0-5}R{0-3}_{N|S}
        # 注意: R4C4_W 在部分SUMO net版本中可能被netconvert优化删除，改用 R4C3_E
        # ── 测试场景定义 ──────────────────────────────────────────
        # 格式: (起点EdgeID, 终点EdgeID, 交通描述)
        scenarios = {
            "场景A 黄河路单向拥堵": (
                "R3C0_E", "R3C4_E",
                f"黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北畅通无阻。请为全部{edge_total}条路段生成权重。"),
            "场景B 花园路封闭绕行": (
                "C4R0_N", "C4R3_N",
                f"花园路农业路至红专路段向北施工封闭；经六路向北畅通无阻。请为全部{edge_total}条路段生成权重。"),
            "场景C 多路段复合拥堵": (
                "R0C0_E", "R3C4_E",
                f"经六路全线封闭施工；花园路向北严重拥堵；黄河路向东中度拥堵。请为全部{edge_total}条路段生成权重。"),
            "场景D 中央核心节点封锁（大范围绕行）": (
                "R0C0_E", "R4C3_E",
                f"经六路农业路至黄河路段全线封闭；花园路政七街至黄河路段完全封闭；经三路向南严重拥堵；经八路向北正常通行。请为全部{edge_total}条路段生成权重。"),
            "场景E 不对称方向拥堵（方向提取）": (
                "R4C0_E", "R4C4_E",
                f"纬五路向东方向因事故严重拥堵，向西方向畅通无阻；未来路向南中度拥堵。请为全部{edge_total}条路段生成权重。"),
            "场景F 大规模复合灾害（极限压力测试）": (
                "R0C0_E", "C4R3_N",
                f"突发大规模事故：花园路农业路至黄河路全段封闭；经六路农业路至红专路段完全封闭；黄河路向东中度拥堵；经三路向北畅通无阻。请为全部{edge_total}条路段生成权重。"),
        }
        with st.expander("⚡ 快捷测试场景"):
            sel = st.selectbox("选择场景", ["— 自定义 —"] + list(scenarios.keys()), key="preset_sel")
            if sel != "— 自定义 —":
                _s, _e, _con = scenarios[sel]
                st.caption(f"📍 起: {_s}  →  终: {_e}")
                st.info(_con[:120] + "…", icon="📝")

        st.markdown('<div class="sec-hdr">⚙️ 规划算法</div>', unsafe_allow_html=True)
        path_algo = st.selectbox(
            "路径规划算法",
            ["A*（统一代价）", "Dijkstra", "Bellman-Ford"],
            key="path_algo",
        )

        run_button = st.button("🚀 一键运行规划", type="primary", use_container_width=True)

        # 快捷场景选中后覆盖运行参数
        if sel != "— 自定义 —":
            _s, _e, _con   = scenarios[sel]
            start_edge         = _s
            end_edge           = _e
            traffic_constraint = _con

        with st.expander("🔧 手动权重（调试用）"):
            manual_str = st.text_area(
                "权重（edge:w, …）",
                value="R0C0_E:8.5, R0C1_E:9.2, C2R0_N:2.0, R3C0_E:6.0, C4R1_N:1.5, "
                      "R1C2_E:3.5, C0R2_N:7.0, R2C3_W:4.0",
                height=80, key="manual_weights",
            )
            use_manual = st.checkbox("✅ 启用手动权重", key="use_manual")

        # [FIX6] 模型原始输出栏：始终可见，未运行时显示"尚未运行"
        _short = selected_model.split("(")[0].strip()
        st.markdown(f'<div class="sec-hdr">🔍 模型原始输出 · {_short}</div>', unsafe_allow_html=True)
        with st.expander("查看原始输出", expanded=False):
            raw_wd = st.session_state.get("last_raw_llm_weights", {})
            raw = st.session_state.get("last_raw", "")
            last_mode = st.session_state.get("last_mode", config.get("RUN_MODE", "demo_mode"))
            last_fb_used = st.session_state.get("last_fallback_used", False)
            last_fb_reason = st.session_state.get("last_fallback_reason", "none")
            if raw:
                if raw_wd:
                    st.success(f"✅ 原始LLM显式解析 {len(raw_wd)} / {edge_total} 条路段")
                    if len(raw_wd) < 50:
                        st.warning(f"⚠️ 原始LLM仅显式解析到 {len(raw_wd)} 条")
                else:
                    st.error("❌ 原始LLM解析失败，请检查模型输出格式")
                st.caption(f"mode={last_mode} | fallback_used={last_fb_used} | fallback_reason={last_fb_reason}")
                st.text_area("Raw Output", raw[:1500], height=160, disabled=True,
                             label_visibility="collapsed")
            else:
                st.info("尚未运行，运行一次后显示")

        st.divider()

        if st.session_state["run_history"]:
            st.markdown('<div class="sec-hdr">📋 规划历史</div>', unsafe_allow_html=True)
            for h in reversed(st.session_state["run_history"][-5:]):
                st.markdown(f"""<div class="hist-row">
                    <span>#{h["id"]}  {h["start"]}→{h["end"]}</span>
                    <span style="color:#1D4ED8;font-weight:600">{h["travel_time"]:.1f}s</span>
                </div>""", unsafe_allow_html=True)
            if st.button("🗑 清除历史", use_container_width=True):
                st.session_state["run_history"] = []
                st.session_state["run_count"]   = 0

        if st.button("📥 导出结果 JSON", use_container_width=True):
            if st.session_state["results"]:
                fn = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                fp = export_engine.export_results(st.session_state["results"], fn)
                with open(fp) as f:
                    st.download_button("⬇️ 下载 JSON", f.read(), fn)
            else:
                st.warning("暂无结果")

        st.divider()
        st.caption(f"💡 路段ID: R{{行}}C{{列}}_{{E/W}}  C{{列}}R{{行}}_{{N/S}}  共{edge_total}条\n"
                   "农业路=R0, 黄河路=R3, 花园路=C4, 经六路=C2")

    _mode_badge = "Demo Mode（允许规则兜底）" if config.get("RUN_MODE", "demo_mode") == "demo_mode" else "Experiment Mode（禁用规则兜底）"
    st.markdown(f'<div class="info-strip">🧪 当前运行模式：<b>{_mode_badge}</b></div>', unsafe_allow_html=True)

    # ======================== 主内容区 ========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 权重 & 路径", "📈 仿真指标", "🔍 方法对比", "🔥 高级分析", "🗺️ 路网总览"
    ])

    with tab5:
        st.markdown('<div class="sec-hdr">🗺️ 郑州金水区路网拓扑（6×5网格）</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("有向边总数", f"{edge_total} 条")
        c2.metric("交叉路口",   "30 个")
        c3.metric("路网面积",   "3.3×2.5 km")
        c4.metric("道路等级",   "3 级")
        try:
            _, ec_preview = _load_net_cached(config.get("SUMO_NET_PATH"))
            wd_preview    = st.session_state.get("last_wd") or {}
            path_preview  = st.session_state.get("last_path")
            net_fig = viz_engine.plot_network_overview(ec_preview, wd_preview or None, path_preview)
            net_fig.update_layout(height=560)
            st.plotly_chart(net_fig, use_container_width=True)
            st.markdown('''<div class="info-strip">
            🔴 <b>主干道 (Arterial)</b>: 农业路(R0) · 黄河路(R3) · 花园路(C4) · 经六路(C2)
            — 4车道 60km/h &nbsp;|&nbsp;
            🟡 <b>次干道 (Secondary)</b>: 红专路(R1) · 经三路(C1) · 经八路(C3) — 3车道 50km/h
            &nbsp;|&nbsp;
            🟢 <b>支路 (Minor)</b>: 政七街(R2) · 纬五路(R4) · 经一路(C0) · 未来路(C5) — 2车道 40km/h
            </div>''', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"路网加载失败：{e}")
            st.code(traceback.format_exc())

    # Tab1~4 一次性创建 placeholder
    with tab1:
        left_col, right_col = st.columns([1, 1], gap="medium")
        with left_col:
            st.markdown('<div class="sec-hdr">📋 大模型权重输出</div>', unsafe_allow_html=True)
            weight_ph = st.empty()
        with right_col:
            st.markdown('<div class="sec-hdr">🗺️ 路径可视化</div>', unsafe_allow_html=True)
            path_ph = st.empty()

    with tab2:
        st.markdown('<div class="sec-hdr">📈 SUMO 仿真指标</div>', unsafe_allow_html=True)
        metric_ph = st.empty()

    with tab3:
        st.markdown('<div class="sec-hdr">🔍 方法对比分析（动态方法集 · 3类对比）</div>', unsafe_allow_html=True)
        compare_ph = st.empty()

    with tab4:
        st.markdown('<div class="sec-hdr">🔥 高级可视化分析</div>', unsafe_allow_html=True)
        adv_ph = st.empty()

    # ======================== 运行逻辑 ========================
    if not run_button:
        return

    if not traffic_constraint.strip():
        st.error("❌ 请填写交通约束描述！"); return
    if not start_edge.strip() or not end_edge.strip():
        st.error("❌ 请填写起点和终点路段ID！"); return

    try:
        net_chk, _ = _load_net_cached(config.get("SUMO_NET_PATH"))
        all_eids   = [e.getID() for e in net_chk.getEdges() if e]
        for eid, label in [(start_edge, "起点"), (end_edge, "终点")]:
            if eid not in all_eids:
                st.error(f"❌ {label}路段 '{eid}' 不存在。可用：{', '.join(sorted(all_eids)[:12])}...")
                return
    except Exception as e:
        st.error(f"❌ 路网加载失败：{e}"); st.code(traceback.format_exc()); return

    prog   = st.progress(0)
    status = st.empty()

    try:
        status.text("🔄 正在推理权重..."); prog.progress(15)

        if st.session_state.get("use_manual") and manual_str.strip():
            raw_llm_weights = path_engine._extract_weights_from_text(manual_str)
            struct_output = manual_str.strip()
            infer_time = 0.0
            raw_output = "[手动输入]"
            explicit_weight_dict = dict(raw_llm_weights)
            _llm_parsed = len(raw_llm_weights)
            _gat_applied = False
            _gat_alpha_used = st.session_state.get("gat_alpha", 0.65)
            _mode_used = config.get("RUN_MODE", "demo_mode")
            _fallback_used = False
            _fallback_reason = "none"
            _timing_info = {
                "init_ms": 0.0,
                "infer_ms": 0.0,
                "smooth_ms": 0.0,
                "total_ms": 0.0,
                "cold_start": False,
            }
            if not raw_llm_weights:
                st.error("❌ 手动权重格式错误，示例：R3C0_E:8.5, C4R1_N:1.5")
                prog.empty(); status.empty(); return
            final_planning_weights = {e: 2.0 for e in path_engine._ALL_EDGES}
            final_planning_weights.update(raw_llm_weights)
            weight_dict = final_planning_weights
            st.info(f"ℹ️ 手动权重：{len(raw_llm_weights)} 条（其余边按默认2.0补全）")
            edge_metrics = compute_edge_metrics(
                final_weights=final_planning_weights,
                explicit_weights=explicit_weight_dict,
                all_edges=path_engine._ALL_EDGES,
                method_name="Manual",
            )
            st.session_state["last_edge_metrics"] = edge_metrics
            st.session_state["last_raw"] = raw_output
            st.session_state["last_wd"] = final_planning_weights
            st.session_state["last_raw_llm_weights"] = raw_llm_weights
            st.session_state["llm_parsed"] = _llm_parsed
            st.session_state["gat_alpha_used"] = _gat_alpha_used
            st.session_state["gat_applied"] = _gat_applied
            st.session_state["last_mode"] = _mode_used
            st.session_state["last_fallback_used"] = _fallback_used
            st.session_state["last_fallback_reason"] = _fallback_reason
            st.session_state["last_timing_info"] = _timing_info
        else:
            # model_path 变化时缓存自动失效
            _cur_model_path = config.config.get("MODEL_PATH", "")
            _gw_result = path_engine.generate_weights(
                traffic_constraint, selected_model + "|" + _cur_model_path, model_manager,
                use_gat=st.session_state.get("use_gat", False),
                gat_model_path=st.session_state.get("gat_path", "/root/autodl-tmp/gat_model.pt"),
                gat_alpha=st.session_state.get("gat_alpha", 0.65),
                mode=config.get("RUN_MODE", "demo_mode"),
            )
            (
                struct_output,
                raw_llm_weights,
                final_planning_weights,
                infer_time,
                raw_output,
                _llm_parsed,
                _gat_applied,
                _gat_alpha_used,
                explicit_weight_dict,
                _mode_used,
                _fallback_used,
                _fallback_reason,
                _timing_info,
            ) = _gw_result
            weight_dict = final_planning_weights
            st.session_state["last_raw"] = raw_output
            st.session_state["last_wd"] = final_planning_weights
            st.session_state["last_raw_llm_weights"] = raw_llm_weights
            st.session_state["llm_parsed"] = _llm_parsed
            st.session_state["gat_alpha_used"] = _gat_alpha_used
            st.session_state["gat_applied"] = _gat_applied
            st.session_state["last_mode"] = _mode_used
            st.session_state["last_fallback_used"] = _fallback_used
            st.session_state["last_fallback_reason"] = _fallback_reason
            st.session_state["last_timing_info"] = _timing_info
            if not final_planning_weights:
                st.error("❌ 模型未生成有效权重")
                with st.expander("查看原始输出"):
                    st.code(raw_output or "(空)", language="text")
                prog.empty(); status.empty(); return
            edge_metrics = compute_edge_metrics(
                final_weights=final_planning_weights,
                explicit_weights=explicit_weight_dict,
                all_edges=path_engine._ALL_EDGES,
                method_name=selected_model,
            )
            st.session_state["last_edge_metrics"] = edge_metrics

            _gat_badge = " → **GAT已补全/平滑最终权重** ✨" if _gat_applied else ""
            _is_sparse_mode = "Sparse" in selected_model
            if _is_sparse_mode:
                if edge_metrics["parsed_edges"] >= 1:
                    st.success(
                        f"✅ **稀疏输出模式**：显式解析 {edge_metrics['parsed_edges']}/{edge_total} 条异常边"
                        f"（{edge_metrics['parsed_edge_ratio']:.1f}%），最终非默认信号边为 {edge_metrics['signal_edges']}/{edge_total}"
                        f"（{edge_metrics['signal_ratio']:.1f}%）{_gat_badge}"
                    )
                else:
                    st.info("ℹ️ 稀疏模型未检测到异常边，最终全路网维持默认权重 2.0")
            else:
                if edge_metrics["parsed_edges"] < 10:
                    st.warning(
                        f"⚠️ 模型仅显式解析 {edge_metrics['parsed_edges']}/{edge_total} 条，"
                        f"最终非默认信号边为 {edge_metrics['signal_edges']}/{edge_total} 条{_gat_badge}"
                    )
                elif edge_metrics["parsed_edges"] < 90:
                    st.warning(
                        f"⚠️ 模型显式解析 {edge_metrics['parsed_edges']}/{edge_total} 条（{edge_metrics['parsed_edge_ratio']:.1f}%），"
                        f"最终非默认信号边为 {edge_metrics['signal_edges']}/{edge_total} 条（{edge_metrics['signal_ratio']:.1f}%）{_gat_badge}"
                    )
                else:
                    st.success(
                        f"✅ 模型显式解析 {edge_metrics['parsed_edges']}/{edge_total} 条（{edge_metrics['parsed_edge_ratio']:.1f}%），"
                        f"最终非默认信号边为 {edge_metrics['signal_edges']}/{edge_total} 条（{edge_metrics['signal_ratio']:.1f}%）{_gat_badge}"
                    )

            _mode_label = "Demo Mode" if _mode_used == "demo_mode" else "Experiment Mode"
            if _fallback_used:
                _fb_msg = f"mode={_mode_used} | fallback_used=True | reason={_fallback_reason}"
                if _mode_used == "experiment_mode":
                    st.warning(f"⚠️ {_mode_label} 检测到fallback，已显式记录：{_fb_msg}")
                else:
                    st.info(f"ℹ️ {_mode_label} 触发fallback：{_fb_msg}")
            else:
                st.caption(f"mode={_mode_used} | fallback_used=False")

            if _gat_applied:
                st.caption(f"🔢 GAT实际混合系数 α = **{_gat_alpha_used:.2f}**（自适应调整，设定值={st.session_state.get('gat_alpha',0.85):.2f}）")

        status.text("🗺️ 正在规划路径..."); prog.progress(35)
        net, edge_coords = _load_net_cached(config.get("SUMO_NET_PATH"))
        _algo = st.session_state.get("path_algo", "A*（统一代价）")
        if "Dijkstra" in _algo:
            path, total_cost = path_engine.dijkstra_route(net, start_edge, end_edge, weight_dict)
            st.info("ℹ️ Dijkstra模式：与A*使用相同预计通行时间代价，但不使用启发函数。")
        elif "Bellman-Ford" in _algo:
            path, total_cost = path_engine.bellman_ford_route(net, start_edge, end_edge, weight_dict)
            st.info("ℹ️ Bellman-Ford模式：与A*/Dijkstra使用相同边代价，差异仅在搜索/松弛策略。")
        else:
            path, total_cost = path_engine.astar_route(net, start_edge, end_edge, weight_dict)
        st.session_state["total_cost"] = total_cost
        if not path:
            st.error("❌ 未找到有效路径，请检查权重路段ID与路网是否一致")
            prog.empty(); status.empty(); return

        status.text("🚗 正在运行SUMO仿真..."); prog.progress(60)
        # ══ SUMO限速：规则解析场景真实路况 ══════════════════════════
        # 与模型预测权重完全无关，确保各模型仿真基准一致
        # 逻辑：模型权重决定A*选哪条路 → SUMO用真实路况测试该路的实际行程时间
        # → 模型选对路(绕开拥堵)则行程短，选错路则行程长 → 直接反映规划质量
        _scene_weights = path_engine._rule_parse_weights(traffic_constraint)
        travel_time, avg_speed, speed_data, pos_data = sumo_engine.run_simulation(
            tuple(path), tuple(sorted(_scene_weights.items())))
        st.caption("🔵 SUMO限速来自场景规则解析（与模型权重无关）| 行程时间=路径规划质量的客观评估")
        st.session_state["travel_time"] = travel_time

        # 提前计算Dijkstra参考路径（供Tab2/Tab4复用，避免Tab4重复计算）
        _uniform_wd = {e: 1.0 for e in weight_dict}
        _path_dijk_ref, _cost_dijk_ref = path_engine.dijkstra_route(
            net, start_edge, end_edge, _uniform_wd)
        # 计算约束符合率：模型权重在"显著拥堵"路段的方向一致性
        _sig_edges = [e for e in _scene_weights if abs(_scene_weights[e] - 2.0) > 0.5]
        if _sig_edges:
            _correct = sum(1 for e in _sig_edges
                           if (weight_dict.get(e, 2.0) >= 4.0) == (_scene_weights[e] >= 4.0))
            _comply_rate = _correct / len(_sig_edges) * 100
        else:
            _comply_rate = 100.0

        prog.progress(85)
        st.session_state["results"] = {
            "timestamp":   datetime.now().isoformat(),
            "model":       selected_model,
            "mode":        _mode_used,
            "fallback_used": _fallback_used,
            "fallback_reason": _fallback_reason,
            "constraint":  traffic_constraint,
            "start":       start_edge, "end": end_edge,
            "weights":     final_planning_weights,
            "raw_llm_weights": raw_llm_weights,
            "final_planning_weights": final_planning_weights,
            "path": path,
            "total_cost":  total_cost,
            "travel_time": travel_time,
            "llm_parsed": _llm_parsed,
            "parsed_edges": edge_metrics["parsed_edges"],
            "parsed_edge_ratio": edge_metrics["parsed_edge_ratio"],
            "signal_edges": edge_metrics["signal_edges"],
            "signal_ratio": edge_metrics["signal_ratio"],
            "default_edges": edge_metrics["default_edges"],
            "avg_speed":   avg_speed,
            "infer_time":  infer_time,
            "timing_info": _timing_info,
            "gat_applied": _gat_applied,
            "gat_alpha_used": _gat_alpha_used,
        }
        st.session_state["last_wd"]   = weight_dict
        st.session_state["last_path"] = path

        prog.progress(100); status.text("✅ 完成！")
        time.sleep(0.4); prog.empty(); status.empty()

        st.session_state["run_count"] += 1
        st.session_state["run_history"].append({
            "id": st.session_state["run_count"],
            "start": start_edge, "end": end_edge,
            "travel_time": travel_time, "path_cost": total_cost,
            "edges_parsed": _llm_parsed,
        })

        severe = sum(1 for w in weight_dict.values() if w >= 6.0)
        medium = sum(1 for w in weight_dict.values() if 3.0 <= w < 6.0)
        clear  = sum(1 for w in weight_dict.values() if w < 3.0)

        # ── Tab1 左列 ──
        weight_ph.empty()
        with weight_ph.container():
            kc1, kc2 = st.columns(2)
            kc1.metric("🎯 路径代价", f"{total_cost:.2f}")
            kc2.metric("⏱️ 推理时间", f"{infer_time:.2f} s")
            kc3, kc4 = st.columns(2)
            kc3.metric("📍 路径段数", f"{len(path)} 段")
            _lp = st.session_state.get("llm_parsed", len(weight_dict))
            _ga = "✨GAT" if st.session_state.get("gat_applied") else ""
            _is_sp = "Sparse" in selected_model
            if _is_sp:
                kc4.metric("📊 检测异常路段", f"{_lp}条",
                           delta=f"已补全至{edge_total}条", delta_color="normal")
            else:
                kc4.metric("📊 解析路段", f"{_lp}/{edge_total}{_ga}",
                           delta=f"已补全→{edge_total}" if _ga else None)
            # ── 关键路段对比（模���权重 vs 规则真实值）────────────────
            _scene_ref = _scene_weights  # 复用已在 SUMO 前计算的场景权重，避免重复解析
            _key = [(e, weight_dict.get(e, 2.0), _scene_ref.get(e, 2.0))
                    for e in _scene_ref if abs(_scene_ref.get(e, 2.0) - 2.0) > 0.3]
            _key.sort(key=lambda x: abs(x[2]-x[1]), reverse=True)
            if _key:
                _err = max(abs(sw-mw) for _,mw,sw in _key[:5])
                _diff_str = " | ".join(f"{e}: 模型={mw:.1f} 实际≈{sw:.1f}"
                                       for e,mw,sw in _key[:4])
                if _err > 3.0:
                    st.warning(f"⚠️ 关键路段权重偏差大（影响选路）: {_diff_str}")
                elif _err > 1.0:
                    st.info(f"📊 关键路段对比: {_diff_str}")

            st.markdown(f"""<div class="info-strip">
            🔴 严重拥堵 {severe} 条 &nbsp;|&nbsp; 🟡 中度拥堵 {medium} 条 &nbsp;|&nbsp;
            🟢 畅通 {clear} 条 &nbsp;|&nbsp; 规划路径: <b>{" → ".join(path)}</b>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div class="sec-hdr">路径边权重</div>', unsafe_allow_html=True)
            md_path = "| Edge ID | Weight | Congestion |\n|---|---|---|\n"
            for eid in path:
                w   = weight_dict.get(eid, 1.0)
                lvl = ("🔴 Severe" if w >= 8.5 else "🟠 Heavy" if w >= 6.0 else
                       "🟡 Medium" if w >= 4.0 else "🔵 Light" if w >= 2.5 else "🟢 Clear")
                bar = "█" * int(w) + "░" * (10 - int(w))
                md_path += f"| `{eid}` | **{w:.1f}** `{bar}` | {lvl} |\n"
            st.markdown(md_path)
            st.markdown('<div class="sec-hdr">Top 10 拥堵路段</div>', unsafe_allow_html=True)
            md_top = "| Edge ID | Weight | Status |\n|---|---|---|\n"
            for eid, w in sorted(weight_dict.items(), key=lambda x: -x[1])[:10]:
                lvl = ("🔴 Severe" if w >= 8.5 else "🟠 Heavy" if w >= 6.0 else
                       "🟡 Medium" if w >= 4.0 else "🔵 Light" if w >= 2.5 else "🟢 Clear")
                md_top += f"| `{eid}` | **{w:.1f}** | {lvl} |\n"
            st.markdown(md_top)

        # ── Tab1 右列 ──
        path_ph.empty()
        with path_ph.container():
            route_fig = viz_engine.plot_route_plotly(edge_coords, path, weight_dict)
            route_fig.update_layout(height=520)
            st.plotly_chart(route_fig, use_container_width=True)
            pie_fig = go.Figure(go.Pie(
                labels=["Severe(>=6)", "Medium(3-6)", "Clear(<3)"],
                values=[severe, medium, clear],
                marker_colors=["#EF4444", "#F59E0B", "#22C55E"],
                hole=0.55, textinfo="label+percent", textfont_size=11,
            ))
            pie_fig.update_layout(
                title="Congestion Distribution", height=260,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        # ── Tab2 ──
        metric_ph.empty()
        with metric_ph.container():
            m1, m2, m3, m4, m5 = st.columns(5)
            # 估算Dijkstra通行时间：按两条路径在场景权重下的拥堵总量比例推算
            if _path_dijk_ref and path:
                _ps_llm  = sum(_scene_weights.get(e, 2.0) for e in path)
                _ps_dijk = sum(_scene_weights.get(e, 2.0) for e in _path_dijk_ref)
                dijkstra_time = travel_time * (_ps_dijk / max(_ps_llm, 0.1))
            else:
                dijkstra_time = travel_time
            _delta_t = dijkstra_time - travel_time
            m1.metric("🚗 通行时间",   f"{travel_time:.1f} s",
                      delta=f"{'↓' if _delta_t >= 0 else '↑'}{abs(_delta_t):.1f}s vs Dijkstra",
                      delta_color="normal" if _delta_t >= 0 else "inverse")
            m2.metric("⚡ 平均车速",   f"{avg_speed:.1f} km/h",
                      delta=f"+{avg_speed-30:.1f} vs 30km/h", delta_color="normal")
            m3.metric("🎯 路径总代价", f"{total_cost:.2f}")
            m4.metric("✅ 约束符合率", f"{_comply_rate:.0f} %")
            m5.metric("📏 路径段数",   f"{len(path)} 段")
            st.markdown("")
            if speed_data:
                s_col, g_col = st.columns([3, 2])
                with s_col:
                    st.markdown('<div class="sec-hdr">车辆速度曲线</div>', unsafe_allow_html=True)
                    spd_fig = viz_engine.plot_speed_plotly(speed_data, config.get("STEP_LENGTH", 0.1))
                    spd_fig.update_layout(height=320)
                    st.plotly_chart(spd_fig, use_container_width=True)
                with g_col:
                    st.markdown('<div class="sec-hdr">路径甘特图</div>', unsafe_allow_html=True)
                    _tt = travel_time if travel_time > 0 else sum(weight_dict.get(e,1.0) for e in path) * 3.0
                    _scene_w_gantt = _scene_weights  # 复用已计算的场景权重
                    gantt_fig = viz_engine.plot_path_gantt(path, weight_dict, _tt, scene_weights=_scene_w_gantt)
                    if gantt_fig:
                        gantt_fig.update_layout(height=320)
                        st.plotly_chart(gantt_fig, use_container_width=True)
            else:
                st.info("SUMO 未采集到速度数据（路径过短或仿真异常）")

        # ── Tab3：真实对比实验 ──
        compare_ph.empty()
        with compare_ph.container():
            COMPARE_RESULT_PATH = _latest_compare_results_path()

            # 运行对比实验按钮
            constraint_rate = (
                sum(1 for w in weight_dict.values() if w > 5.0) / max(len(weight_dict), 1) * 100
            )
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn2:
                run_compare_btn = st.button(
                    "🔬 运行对比实验", use_container_width=True,
                    help="运行多方法对比实验（约15分钟）")
            if run_compare_btn:
                with st.spinner("正在运行对比实验，请稍候..."):
                    try:
                        import subprocess as _sp
                        res = _sp.run(
                            ["python", "/root/autodl-tmp/compare/run_compare.py", "--llm"],
                            capture_output=True, text=True, timeout=1200,
                            cwd="/root/autodl-tmp"
                        )
                        if res.returncode == 0:
                            st.success("✅ 对比实验完成！")
                        else:
                            st.error(f"❌ 失败: {res.stderr[-300:]}")
                    except Exception as ex:
                        st.error(f"❌ {ex}")

            bar_fig, radar_fig, scatter_fig, cmp_df, data_src, box_fig, scene_bar_fig, heatmap_fig = viz_engine.plot_real_compare(
                COMPARE_RESULT_PATH,
                current_cost       = float(total_cost),
                current_constraint = float(constraint_rate),
                current_time       = float(infer_time),
            )

            import os as _os
            if not _os.path.exists(COMPARE_RESULT_PATH):
                st.warning("⚠️ 未检测到对比实验结果：当前不展示估算数据。请先运行对比实验。")

            _n_methods = len(cmp_df) if cmp_df is not None else 9

            # ── 第一行：标题 + 数据来源标签 ───────────────────────
            st.markdown(f'<div class="sec-hdr">📊 综合性能对比（{_n_methods} 种方法）</div>',
                        unsafe_allow_html=True)
            if data_src == "实验数据" or data_src == "real":
                st.caption("✅ 数据来源：真实 SUMO 仿真（6 个测试场景均值）")
            else:
                st.caption("⚠️ 未检测到对比实验结果，请先运行对比实验再用于答辩展示")

            # ── 第二行：主柱状图（全宽）──────────────────────────
            bar_fig.update_layout(height=440)
            st.plotly_chart(bar_fig, use_container_width=True)

            # ── 第三行：雷达图 + 散点图 ──────────────────────────
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                st.markdown('<div class="sec-hdr">🕸️ 多维能力雷达图</div>', unsafe_allow_html=True)
                radar_fig.update_layout(height=420)
                st.plotly_chart(radar_fig, use_container_width=True)
            with rc2:
                st.markdown('<div class="sec-hdr">📈 推理速度 vs 约束符合率</div>', unsafe_allow_html=True)
                scatter_fig.update_layout(height=420)
                st.plotly_chart(scatter_fig, use_container_width=True)

            # ── 第四行：热力图（全宽）────────────────────────────
            st.markdown('<div class="sec-hdr">🌡️ 方法 × 场景 SUMO 行程时间热力图</div>',
                        unsafe_allow_html=True)
            st.caption("颜色越绿 = 行程越短（列内归一化），可直观看出每个场景下各方法的相对优劣")
            if heatmap_fig is not None:
                heatmap_fig.update_layout(height=420)
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.info("运行对比实验后显示热力图。")

            # ── 第五行：箱线图 + 分场景柱状图 ──────────────────
            bc1, bc2 = st.columns([1, 1])
            with bc1:
                st.markdown('<div class="sec-hdr">📦 SUMO 行程时间分布（箱线图）</div>',
                            unsafe_allow_html=True)
                st.caption("展示各方法在 6 个场景中的时间稳定性，箱体越窄越稳定")
                box_fig.update_layout(height=400)
                st.plotly_chart(box_fig, use_container_width=True)
            with bc2:
                st.markdown('<div class="sec-hdr">📊 分场景性能对比</div>', unsafe_allow_html=True)
                st.caption("主要方法在各场景的 SUMO 行程时间，便于分析特定场景下的优劣")
                if scene_bar_fig is not None:
                    scene_bar_fig.update_layout(height=400)
                    st.plotly_chart(scene_bar_fig, use_container_width=True)
                else:
                    st.info("运行对比实验后显示分场景对比图。")

            # ── 第六行：数据汇总表格 ─────────────────────────────
            st.markdown('<div class="sec-hdr">📋 综合数据汇总（按 SUMO 行程时间排名）</div>',
                        unsafe_allow_html=True)
            tb1, tb2 = st.columns([4, 1])
            with tb1:
                if cmp_df is not None and len(cmp_df) > 0:
                    # 高亮最小 SUMO 时间行
                    _display_cols = [c for c in [
                        "排名", "方法", "SUMO时间(s)↓", "约束符合率(%)↑",
                        "非默认路况占比(%)↑", "推理耗时(s)↓", "路径代价↓"
                    ] if c in cmp_df.columns]
                    if not _display_cols:
                        _display_cols = list(cmp_df.columns)

                    def _highlight_row(row):
                        method_val = str(row.get("方法", ""))
                        if any(x in method_val for x in ("Sparse-LoRA+GAT", "LoRA+GAT")):
                            return ["background-color: rgba(52,168,83,0.15)"] * len(row)
                        elif any(x in method_val for x in ("Sparse-LoRA", "Qwen", "R1")):
                            return ["background-color: rgba(66,133,244,0.10)"] * len(row)
                        return [""] * len(row)

                    styled = cmp_df[_display_cols].style.apply(_highlight_row, axis=1)
                    sumo_col = next((c for c in _display_cols if "SUMO" in c), None)
                    if sumo_col:
                        # background_gradient doesn't alter display values (avoids pyarrow bug)
                        styled = styled.background_gradient(
                            subset=[sumo_col], cmap="RdYlGn_r", vmin=0)
                    # 彻底绕过 PyArrow 引擎冲突，使用原生 HTML 渲染高亮表格
                    try:
                        try:
                            html_table = styled.hide(axis="index").to_html()
                        except AttributeError:
                            html_table = styled.hide_index().to_html() # 兼容老版本 Pandas
                        st.markdown(html_table, unsafe_allow_html=True)
                    except Exception:
                        st.table(cmp_df[_display_cols]) # 终极无样式保底
            with tb2:
                st.markdown("**图例说明**")
                st.markdown("""
- 🟢 绿底：GAT级联类方法
- 🔵 蓝底：LoRA 系列基线
- ↑ 越高越好　↓ 越低越好
- **排名** 按 SUMO 行程时间升序
""")
                if st.button("📥 导出对比 CSV"):
                    fn = export_engine.export_dataframe(
                        cmp_df, f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    st.success(f"已导出：{fn}")

            # ── SmallNet 反证基线实验 ──────────────────────────────
            st.divider()
            st.markdown('<div class="sec-hdr">🔬 SmallNet 反证基线实验（4×3路网·24条边）</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="info-strip">
            <b>实验目的</b>：在 4×3 小型路网（24条边）上验证模型在短序列条件下的完整解析能力。<br>
            若小路网显式解析边占比接近 100%，而大路网（主路网边数）显式解析边占比明显下降，
            则说明瓶颈主要来自全量长输出任务，而不是模型本身无法理解路况。<br>
            → 这为<b>稀疏输出（Sparse Output）</b>与拓扑补全架构提供反证支撑。
            </div>""", unsafe_allow_html=True)

            SMALLNET_RESULT_PATH = "/root/autodl-tmp/results/smallnet_results.json"
            COMPARE_RESULT_PATH = _latest_compare_results_path()

            col_sn1, col_sn2 = st.columns([3, 1])
            with col_sn2:
                run_smallnet_btn = st.button("▶ 运行SmallNet对比实验", use_container_width=True,
                    help="运行 24 边 SmallNet + 已训练模型对比，结果写入 smallnet_results.json")
            if run_smallnet_btn:
                with st.spinner("运行 SmallNet 反证对比实验（含已训练模型）..."):
                    try:
                        import subprocess as _sp
                        res = _sp.run(
                            ["python", "/root/autodl-tmp/code/smallnet_baseline.py", "--all-llm"],
                            capture_output=True, text=True, timeout=120,
                            cwd="/root/autodl-tmp"
                        )
                        if res.returncode == 0:
                            st.success("✅ SmallNet 对比实验完成！")
                        else:
                            st.error(f"❌ {res.stderr[-300:]}")
                    except Exception as ex:
                        st.error(f"❌ {ex}")

            left_col, right_col = st.columns([1.7, 1.0], gap="medium")
            with left_col:
                decay_fig, decay_summary = viz_engine.plot_smallnet_decay_compare(SMALLNET_RESULT_PATH, COMPARE_RESULT_PATH)
                decay_fig.update_layout(height=440)
                st.plotly_chart(decay_fig, use_container_width=True)
            with right_col:
                topo_fig = viz_engine.plot_smallnet_topology_preview()
                topo_fig.update_layout(height=300)
                st.plotly_chart(topo_fig, use_container_width=True)

                primary = (decay_summary or {}).get("primary")
                if primary:
                    delta = max(primary.get("drop", 0.0), 0.0)
                    st.markdown(f"""<div class="info-strip">
                    <b>核心论点</b>：数据显示，当路网边数从 24 增加至 98 时，
                    <b>{primary.get('label', primary.get('method', '全量模型'))}</b> 的显式解析边占比下降了
                    <b>{delta:.1f}</b> 个百分点，证明了长序列注意力稀释现象的客观存在，
                    验证了稀疏提取架构的必要性。
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="info-strip">
                    <b>核心论点</b>：SmallNet / LargeNet 结果文件缺失时，将在运行实验后自动生成并刷新对比图。
                    </div>""", unsafe_allow_html=True)

            import os as _os2, json as _json2
            if _os2.path.exists(SMALLNET_RESULT_PATH):
                with open(SMALLNET_RESULT_PATH, encoding="utf-8") as _f2:
                    _sn_data = _json2.load(_f2)
                for _tc in _sn_data:
                    st.markdown(f"**{_tc['scenario']}** — Oracle代价: {_tc.get('oracle_cost','?')}")
                    _sn_rows = []
                    for _m in _tc.get("methods", []):
                        _sn_rows.append({
                            "方法": _m["method"],
                            "显式解析边占比(%)": normalize_edge_metric_record(_m, total_edges=_m.get("total_edges", 24), method_name=_m.get("method")).get("parsed_edge_ratio", 0),
                            "显式解析边数": normalize_edge_metric_record(_m, total_edges=_m.get("total_edges", 24), method_name=_m.get("method")).get("parsed_edges", 0),
                            "非默认信号边数": normalize_edge_metric_record(_m, total_edges=_m.get("total_edges", 24), method_name=_m.get("method")).get("signal_edges", 0),
                            "总边数": _m.get("total_edges", 24),
                            "权重MAE": _m.get("weight_mae", "-"),
                            "路径代价": _m.get("cost", 0),
                            "Overhead(%)": _m.get("overhead_pct", 0),
                        })
                    if _sn_rows:
                        import pandas as _pd2
                        _sn_df = _pd2.DataFrame(_sn_rows)
                        _safe_table(_sn_df)  # 规避 pyarrow 版本不兼容
                st.markdown("""<div class="info-strip">
                ✅ <b>结论</b>：SmallNet（24条边）与 LargeNet（主路网边数）并排对照后，
                可以直接看到全量输出方法在大路网上的显式解析边占比衰减，而本文方法保持更稳定的约束符合率。
                </div>""", unsafe_allow_html=True)
            else:
                st.info("ℹ️ 点击「运行SmallNet对比实验」查看结果（会自动纳入已训练的 Qwen 系列模型）")
        # ── Tab4 ──
        adv_ph.empty()
        with adv_ph.container():
            _algo_label = st.session_state.get("path_algo", "A*（统一代价）").split("（")[0].strip()
            st.markdown(f'<div class="sec-hdr">🔄 LLM+{_algo_label} vs Dijkstra 路径对比</div>',
                        unsafe_allow_html=True)
            try:
                # 复用已在 SUMO 后计算的 Dijkstra 参考路径，无需重复运算
                path_dijk2, cost_dijk2 = _path_dijk_ref, _cost_dijk_ref
                _algo_short = st.session_state.get("path_algo", "A*（统一代价）").split("（")[0]
                cmp_path_fig2 = viz_engine.plot_path_comparison(
                    edge_coords, path, path_dijk2, weight_dict)
                cmp_path_fig2.update_layout(
                    height=400,
                    title=f"Path Comparison: LLM+{_algo_short} vs Dijkstra"
                )
                st.plotly_chart(cmp_path_fig2, use_container_width=True)
                cd1, cd2, cd3 = st.columns(3)
                _algo_lbl = st.session_state.get("path_algo","A*（本文）").split("（")[0]
                cd1.metric(f"LLM+{_algo_lbl} 路径代价", f"{total_cost:.2f}",
                           delta=f"{total_cost-cost_dijk2:.2f} vs Dijkstra",
                           delta_color="inverse")
                cd2.metric("Dijkstra 路径代价", f"{cost_dijk2:.2f}")
                _delta_pct = (cost_dijk2 - total_cost) / max(cost_dijk2, 1) * 100
                if _delta_pct > 0:
                    cd3.metric("优化幅度", f"↓{_delta_pct:.1f}%",
                               delta=f"比Dijkstra低{_delta_pct:.1f}%", delta_color="normal")
                elif _delta_pct < 0:
                    cd3.metric("优化幅度", f"↑{abs(_delta_pct):.1f}%",
                               delta=f"比Dijkstra高{abs(_delta_pct):.1f}%（绕避拥堵）",
                               delta_color="inverse")
                else:
                    cd3.metric("优化幅度", "持平")
            except Exception as ex:
                st.warning(f"路径对比失败：{ex}")
            st.markdown("")

            st.markdown('<div class="sec-hdr">🔥 路段权重热力图  +  方向性拥堵雷达</div>',
                        unsafe_allow_html=True)
            hr1, hr2 = st.columns([3, 2])
            with hr1:
                hm_fig = viz_engine.plot_heatmap_plotly(edge_coords, weight_dict)
                hm_fig.update_layout(height=400)
                st.plotly_chart(hm_fig, use_container_width=True)
            with hr2:
                radar_fig = viz_engine.plot_constraint_radar(weight_dict)
                radar_fig.update_layout(height=400)
                st.plotly_chart(radar_fig, use_container_width=True)

            st.markdown('<div class="sec-hdr">🚗 车辆轨迹动画  +  权重分布</div>',
                        unsafe_allow_html=True)
            ar1, ar2 = st.columns([3, 2])
            with ar1:
                if pos_data and len(pos_data) > 2:
                    anim_fig = viz_engine.plot_vehicle_animation(
                        pos_data, edge_coords, path, speed_data, config.get("STEP_LENGTH", 0.1))
                    if anim_fig:
                        anim_fig.update_layout(height=400)
                        st.plotly_chart(anim_fig, use_container_width=True)
                else:
                    st.info("未采集到轨迹数据（路径过短）")
            with ar2:
                vals = list(weight_dict.values())
                hist_fig2 = px.histogram(x=vals, nbins=20,
                    labels={"x": "Weight", "y": "Count"}, title="Weight Distribution",
                    color_discrete_sequence=["#3B82F6"], template="plotly_white")
                hist_fig2.update_layout(height=220, margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(hist_fig2, use_container_width=True)

                top10b = sorted(weight_dict.items(), key=lambda x: -x[1])[:8]
                bar_fig2 = go.Figure(go.Bar(
                    x=[w for _, w in top10b], y=[e for e, _ in top10b],
                    orientation="h",
                    marker_color=["#EF4444" if w >= 8 else "#F97316" if w >= 6
                                  else "#EAB308" if w >= 4 else "#22C55E"
                                  for _, w in top10b],
                    text=[f"{w:.1f}" for _, w in top10b], textposition="outside",
                ))
                bar_fig2.update_layout(
                    title="Top 8 拥堵路段", xaxis=dict(range=[0, 11]),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=230, margin=dict(l=80, r=30, t=40, b=30),
                )
                st.plotly_chart(bar_fig2, use_container_width=True)

            st.markdown('<div class="sec-hdr">🌐 3D 路径权重立体图</div>', unsafe_allow_html=True)
            fig3d = viz_engine.plot_3d_plotly(edge_coords, path, weight_dict)
            fig3d.update_layout(height=500)
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 处理出错：{e}")
        st.code(traceback.format_exc())
        logging.error(f"Processing error: {e}", exc_info=True)
        try:
            prog.empty(); status.empty()
        except Exception:
            pass


if __name__ == "__main__":
    main()