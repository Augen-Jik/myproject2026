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
import html
import traceback
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
import heapq
import math
from typing import Any
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components

from eval_metrics import compute_edge_metrics, normalize_edge_metric_record
from utils_df_safe import make_arrow_safe
from ui_debug_panel import render_debug_panel
from ui_schema import (
    build_debug_table,
    build_model_parsed_constraints,
    build_route_gain_summary,
    build_scene_runtime_constraints,
)
from config import ConfigManager
from demo_scene_selector import (
    DEFAULT_PRODUCTION_SHOWCASE_NAME,
    PRODUCTION_SHOWCASE_NAMES,
    list_calibration_showcase_presets,
    select_best_showcase_scenario,
    select_demo_scenarios,
    select_showcase_scenarios,
)
from inference import (
    generate_weights as inference_generate_weights,
    infer_lora,
    infer_qwen,
    infer_r1_raw,
    infer_r1_sft,
    infer_sparse,
)
from roadnet_meta import edge_road_name as shared_edge_road_name
from roadnet_meta import COL_NAMES, ROW_NAMES, expand_anchor_edges, live_edge_ids, load_roadnet_meta
from routing import (
    astar_route as routing_astar_route,
    bellman_ford_route as routing_bellman_ford_route,
    best_first_route,
    dijkstra_route as routing_dijkstra_route,
    edge_freeflow_time,
    edge_travel_time,
    free_flow_route as routing_free_flow_route,
    heuristic_time,
    iter_outgoing_edges,
    resolve_route_query,
    weight_to_speed_ms as routing_weight_to_speed_ms,
)
from viz import (
    build_folium_map,
    clear_viz_cache,
    export_folium_map,
    load_display_geo_bundle,
    remap_logical_path_to_display_path,
    load_net_cached,
    load_roadnet_visual_meta,
    render_folium_html,
    safe_table_markdown,
    sanitize_component_args,
)
from sumo_runner import build_safe_vehicle_route
from apply_scene_profile_to_sumo import SUMOSceneProfileApplier
from sim_eval_protocol import (
    build_sim_eval_protocol,
    normalize_eval_time_fields,
    protocol_to_dict,
)
from scenarios import classify_scene_type, get_ui_scenarios
from sim_scene_profiles import (
    build_environment_weight_layers,
    build_environment_weights,
    build_reroute_policy,
    get_scene_profile,
    list_scene_profiles,
    resolve_scene_profile,
    resolve_scene_profile_name,
    scene_profile_to_dict,
)
from sparse_utils import parse_sparse_output_bundle as shared_parse_sparse_output_bundle
from final_table_loader import build_ablation_summary, frozen_table_cache_signature, load_frozen_table_bundle

LIVE_EDGE_IDS = tuple(live_edge_ids())
LIVE_EDGE_LIST_STR = ",".join(LIVE_EDGE_IDS)
TRUNCATION_SUMMARY_PATH = (
    "/root/autodl-tmp/results/runs/"
    "20260407T124637Z__experiment_mode__truncation_experiment__llm_length_ablation/"
    "truncation_summary.csv"
)
STRICT_PROTOCOL_SUMMARY_PATHS = {
    "stage4_fix": "/root/autodl-tmp/results/sparse_v2_validation_strict_stage4_fix_summary.csv",
    "stage4c_fix": "/root/autodl-tmp/results/sparse_v2_validation_strict_stage4c_fix_summary.csv",
    "stage4d_micro": "/root/autodl-tmp/results/sparse_v2_validation_strict_stage4d_micro_summary.csv",
}
STRICT_PROTOCOL_DISPLAY_NAMES = {
    "stage4_fix": "Sparse-LoRA-v2 / stage4_fix",
    "stage4c_fix": "stage4c_fix (patch, hold)",
    "stage4d_micro": "stage4d_micro (patch, stopped)",
}
STRICT_PROTOCOL_SLICE_LABELS = {
    "overall": "Overall",
    "simple_local": "simple_local",
    "anti_truncation": "anti_truncation",
}

# 全局字体修复
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

# 统一方法列表、别名和颜色定义
MAINLINE_METHOD_NAME = "Sparse-LoRA-v2"
MAINLINE_MODEL_PATH = "/root/autodl-tmp/model_merged_sparse_v2_stage4_fix"

ALL_METHODS = [
    "Dijkstra", "Rule-A*", "DQN-RouteSelector", "GCN-Weight", "PPO (quick baseline, untuned)",
    "Raw-Qwen", "CoT-Qwen", "Qwen-LoRA", "Qwen-LoRA + GAT (exploratory)",
    "R1-Raw", "R1-LoRA", MAINLINE_METHOD_NAME, "Sparse-LoRA-v2 + GAT (exploratory)"
]

METHOD_NAME_ALIASES = {
    "Rule A*": "Rule-A*",
    "DQN": "DQN-RouteSelector",
    "GCN+RL": "GCN-Weight",
    "Raw Qwen": "Raw-Qwen",
    "CoT Qwen": "CoT-Qwen",
    "Qwen-LoRA★": "Qwen-LoRA",
    "Qwen-LoRA": "Qwen-LoRA",
    "LoRA+GAT★★": "Qwen-LoRA + GAT (exploratory)",
    "LoRA+GAT": "Qwen-LoRA + GAT (exploratory)",
    "R1-LoRA★": "R1-LoRA",
    "R1-LoRA": "R1-LoRA",
    "Sparse-LoRA★★": MAINLINE_METHOD_NAME,
    "Sparse-LoRA": MAINLINE_METHOD_NAME,
    "Sparse-LoRA+GAT★★★": "Sparse-LoRA-v2 + GAT (exploratory)",
    "Sparse-LoRA+GAT": "Sparse-LoRA-v2 + GAT (exploratory)",
}

METHOD_COLORS = {
    "Dijkstra": "#94A3B8",
    "Rule-A*": "#475569",
    "DQN-RouteSelector": "#2563EB",
    "GCN-Weight": "#0F766E",
    "PPO (quick baseline, untuned)": "#64748B",
    "Raw-Qwen": "#8B5CF6",
    "CoT-Qwen": "#A855F7",
    "Qwen-LoRA": "#F59E0B",
    "Qwen-LoRA + GAT (exploratory)": "#FCD34D",
    "R1-Raw": "#06B6D4",
    "R1-LoRA": "#0284C7",
    MAINLINE_METHOD_NAME: "#16A34A",
    "Sparse-LoRA-v2 + GAT (exploratory)": "#F59E0B",
}
METHOD_COLORS.update({
    alias: METHOD_COLORS[target]
    for alias, target in METHOD_NAME_ALIASES.items()
    if target in METHOD_COLORS
})


def canonical_method_name(method_name: str) -> str:
    """将结果文件中的方法别名统一为界面展示名。"""
    name = str(method_name or "").strip()
    return METHOD_NAME_ALIASES.get(name, name)


def _IS_OURS(method_name):
    """仅高亮默认主链路方法，不把 exploratory 方法当成主方法。"""
    return canonical_method_name(method_name) == MAINLINE_METHOD_NAME


def _IS_EXPLORATORY_METHOD(method_name):
    """标记附录/探索性方法，避免在主链路里被误读为默认主方法。"""
    name = canonical_method_name(method_name)
    return (
        "GAT" in name
        or "stage4b" in name.lower()
        or "stage4c" in name.lower()
        or "stage4d" in name.lower()
        or "patch" in name.lower()
    )

# SmallNet UI 常量
# 这里定义的是 4 行 × 3 列路网对应的交叉口网格尺寸。
SMALL_ROAD_ROWS = 4
SMALL_ROAD_COLS = 3
SMALL_NODE_ROWS = SMALL_ROAD_ROWS + 1
SMALL_NODE_COLS = SMALL_ROAD_COLS + 1
SMALL_ROW_NAMES = ["X街", "Y街", "Z街", "W街"]
SMALL_COL_NAMES = ["A路", "B路", "C路"]


# 配置管理类（迁移到 config.py）


# 模型管理类
class ModelManager:
    _ALL_EDGES = LIVE_EDGE_IDS
    _EDGE_LIST_STR = LIVE_EDGE_LIST_STR

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
        # R1 模型需要用 left padding（Qwen2+FlashAttn 要求）
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


# 路径规划引擎
class PathPlanningEngine:
    # 路网常量（类级别，可通过 self. 或 PathPlanningEngine. 访问）
    _ALL_EDGES = LIVE_EDGE_IDS
    _EDGE_LIST_STR = LIVE_EDGE_LIST_STR
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
        # 预处理：统一分隔符、去除Markdown标记
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

    # 文字→数值映射
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

    def _legacy_parse_sparse_output(_self, raw: str) -> dict:
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

        # 1. 解析 think 块的道路级别信息
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

        # 2. 解析结构化输出（think 后或整体），支持文字权重
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

        # 3. 备用路级别格式 R3>7.5 / C2>9.5
        if not wd:
            wd = _self._extract_road_level_weights(raw)

        return wd

    def _weight_to_level(_self, value: float) -> str:
        if value >= 9.2:
            return "封闭"
        if value >= 8.5:
            return "极度拥堵"
        if value >= 7.0:
            return "严重拥堵"
        if value >= 5.0:
            return "中度拥堵"
        if value >= 3.0:
            return "轻度拥堵"
        if value <= 1.3:
            return "畅通"
        return "正常"

    def _normalize_dir_label(_self, token: str | None) -> str:
        if not token:
            return "未指明"
        mapping = {
            "向东": "向东", "东向": "向东",
            "向西": "向西", "西向": "向西",
            "向北": "向北", "北向": "向北",
            "向南": "向南", "南向": "向南",
            "双向": "双向", "全向": "双向",
        }
        return mapping.get(str(token).strip(), str(token).strip() or "未指明")

    def _extract_sparse_range_label(_self, clause: str) -> str:
        if any(token in clause for token in ("全线", "全段", "全程")):
            return "全线"
        all_roads = list(_self._ROW_MAP.keys()) + list(_self._COL_MAP.keys())
        road_pattern = "|".join(re.escape(name) for name in all_roads)
        match = re.search(rf"({road_pattern})[至到]({road_pattern})", clause)
        if match:
            return f"{match.group(1)}至{match.group(2)}"
        return "局部"

    def _infer_sparse_clause_anchor(_self, clause: str) -> dict | None:
        clause = str(clause or "").strip()
        if not clause:
            return None
        dir_label = next((canon for token, canon in {
            "向东": "向东", "东向": "向东", "向西": "向西", "西向": "向西",
            "向北": "向北", "北向": "向北", "向南": "向南", "南向": "向南",
        }.items() if token in clause), None)
        if dir_label in ("向东", "向西"):
            road = next((name for name in _self._ROW_MAP if name in clause), None)
        elif dir_label in ("向北", "向南"):
            road = next((name for name in _self._COL_MAP if name in clause), None)
        else:
            road = next((name for name in _self._ROW_MAP if name in clause), None)
            if road is None:
                road = next((name for name in _self._COL_MAP if name in clause), None)
        if road is None:
            return None

        value_match = re.search(r"(\d+(?:\.\d+)?)", clause)
        weight = float(value_match.group(1)) if value_match else _self._text_to_weight(clause)
        if weight is None:
            return None

        return {
            "road": road,
            "dir": _self._normalize_dir_label(dir_label),
            "range": _self._extract_sparse_range_label(clause),
            "level": _self._weight_to_level(weight),
            "weight": float(weight),
            "text": clause,
        }

    def _expand_sparse_anchor_to_edges(_self, road: str, dir_label: str, range_label: str) -> list[str]:
        return expand_anchor_edges(road, dir_label, range_label)

    def _register_sparse_anchor(_self, bundle: dict, anchors_by_key: dict, anchor: dict, source: str) -> None:
        road = anchor.get("road")
        if not road:
            return
        weight = anchor.get("weight")
        if weight is None:
            weight = _self._text_to_weight(anchor.get("level", ""))
        if weight is None:
            return
        dir_label = _self._normalize_dir_label(anchor.get("dir"))
        range_label = anchor.get("range") or "局部"
        level = anchor.get("level") or _self._weight_to_level(float(weight))
        edges = _self._expand_sparse_anchor_to_edges(road, dir_label, range_label)
        if not edges:
            return

        base_conf = {
            "structured": 0.92,
            "clause": 0.78,
            "legacy": 0.52,
        }.get(source, 0.68)
        if dir_label not in {"未指明", "双向"}:
            base_conf += 0.03
        if range_label not in {"局部", "全文", "未指明", ""}:
            base_conf += 0.03
        if level not in {"待判断", "正常"}:
            base_conf += 0.02
        conf = max(0.15, min(0.98, base_conf))

        key = f"{road}|{dir_label}|{range_label}"
        current = anchors_by_key.get(key)
        if current is not None:
            new_strength = abs(float(weight) - 2.0)
            cur_strength = abs(float(current["weight"]) - 2.0)
            if new_strength < cur_strength or (new_strength == cur_strength and conf <= float(current.get("confidence", 0.0))):
                return
            if round(float(weight), 2) != round(float(current["weight"]), 2):
                bundle.setdefault("conflicts", []).append({
                    "key": key,
                    "kept": round(float(weight), 2),
                    "discarded": round(float(current["weight"]), 2),
                })

        anchors_by_key[key] = {
            "road": road,
            "dir": dir_label,
            "range": range_label,
            "level": level,
            "weight": round(float(weight), 2),
            "source": source,
            "confidence": round(conf, 3),
            "edges": list(edges),
            "text": anchor.get("text", ""),
        }

        for eid in edges:
            old_weight = bundle["mapped_weights"].get(eid, 2.0)
            if eid not in bundle["mapped_weights"] or abs(float(weight) - 2.0) > abs(old_weight - 2.0):
                bundle["mapped_weights"][eid] = round(float(weight), 2)
            bundle["edge_confidence"][eid] = max(bundle["edge_confidence"].get(eid, 0.0), conf)
            if float(weight) >= 9.0:
                bundle.setdefault("protected_edges", []).append(eid)

    def _parse_sparse_output_bundle(_self, raw: str) -> dict:
        return shared_parse_sparse_output_bundle(
            raw,
            scene_type=getattr(_self, "_last_scene_type", classify_scene_type(text=raw or "")),
        )

    def _parse_sparse_output(_self, raw: str) -> dict:
        bundle = _self._parse_sparse_output_bundle(raw)
        return dict(bundle.get("mapped_weights", {}))

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
        gat_model_path: str = "/root/autodl-tmp/gat_model_v2_gated.pt",
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

    def free_flow_route(self, net, start_eid, end_eid):
        """真正未绕行基线：按 free-flow travel time 的最短路，不使用当前拥堵权重。"""
        return routing_free_flow_route(
            net,
            start_eid,
            end_eid,
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


# SUMO 仿真引擎
class SUMOSimulationEngine:
    def __init__(self, config):
        self.config = config

    def run_simulation(
        _self,
        path_tuple,
        scene_profile,
        environment_weight_tuple=(),
        *,
        scene_weight_layer_payload=None,
        scene_mode: str = "benchmark",
    ):
        path = list(path_tuple)
        step_len = _self.config.get("STEP_LENGTH", 0.1)
        tripinfo_path = _self.config.get("SUMO_TRIPINFO", "/root/autodl-tmp/SUMO/tripinfo.xml")
        planner_vehicle_id = str(_self.config.get("SUMO_PLANNER_VEHICLE_ID", "llm_veh"))
        planner_route_id = str(_self.config.get("SUMO_PLANNER_ROUTE_ID", "llm_route"))
        planner_vtype = str(_self.config.get("SUMO_PLANNER_VTYPE", "car"))
        planner_vclass = str(_self.config.get("SUMO_PLANNER_VCLASS", "passenger"))
        depart_lane = str(_self.config.get("SUMO_DEPART_LANE", "best"))
        depart_pos = str(_self.config.get("SUMO_DEPART_POS", "base"))
        depart_speed = str(_self.config.get("SUMO_DEPART_SPEED", "0"))
        min_speed_samples = int(_self.config.get("SUMO_MIN_SPEED_SAMPLES", 5) or 5)
        net, _ = _load_net_cached(_self.config.get("SUMO_NET_PATH"))
        scene_injector = SUMOSceneProfileApplier(
            _self.config,
            scene_profile,
            runtime_spillover_payload=dict(scene_weight_layer_payload or {}),
            scene_mode=scene_mode,
        )
        runtime_affected_edges = (
            set(dict(scene_profile.incident_edges or {}).keys())
            | set(tuple(scene_profile.blocked_edges or ()))
            | set(dict(scene_profile.lane_reduction_edges or {}).keys())
        )
        if str(scene_mode or "") == "showcase":
            runtime_affected_edges |= set(dict((scene_weight_layer_payload or {}).get("spillover_weights") or {}).keys())

        def _route_affected_overlap_ratio(route_edges):
            route_list = [str(edge_id) for edge_id in list(route_edges or []) if str(edge_id).strip()]
            if not route_list:
                return float("nan")
            return len(set(route_list) & runtime_affected_edges) / max(len(route_list), 1)

        original_start_edge = path[0] if path else ""
        original_end_edge = path[-1] if path else ""
        simulation_result = {
            "ok": False,
            "stage": "init",
            "reason": "simulation_not_started",
            "error_message": "",
            "scene_profile": scene_profile.scene_name,
            "vehicle_id": planner_vehicle_id,
            "vehicle_type": planner_vtype,
            "vehicle_vclass": planner_vclass,
            "original_start_edge": original_start_edge,
            "final_start_edge": original_start_edge,
            "original_end_edge": original_end_edge,
            "final_end_edge": original_end_edge,
            "original_route_edges": list(path),
            "final_route_edges": list(path),
            "route_validation": {"ok": False, "reason": "not_validated"},
            "route_repair_actions": [],
            "vehicle_inserted": False,
            "vehicle_departed": False,
            "vehicle_arrived": False,
            "simulation_truncated": False,
            "completion_status": "not_started",
            "tripinfo_arrival_s": float("nan"),
            "tripinfo_vaporized": "",
            "speed_data": [],
            "pos_data": [],
            "speed_samples_collected": 0,
            "speed_status": "not_started",
            "speed_reason": "simulation_not_started",
            "travel_time": 0.0,
            "avg_speed": 0.0,
            "time_loss_s": None,
            "waiting_time_s": None,
            "stop_count": None,
            "depart_delay_s": float("nan"),
            "route_length_edges": len(path),
            "affected_edge_overlap_ratio": _route_affected_overlap_ratio(path),
            "sumo_cmd": [],
            "reroute_policy": build_reroute_policy(scene_profile),
            "runtime_spillover_summary": scene_injector.runtime_spillover_summary(),
            "debug_events": [],
        }

        def _record_debug(message: str, **payload):
            debug_entry = {"message": message}
            if payload:
                debug_entry.update(sanitize_component_args(payload))
            simulation_result["debug_events"].append(debug_entry)
            logging.info("%s | %s", message, json.dumps(debug_entry, ensure_ascii=False, sort_keys=True))

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

        sumo_cmd = scene_injector.build_sumo_command(
            tripinfo_path=tripinfo_path,
            step_length=step_len,
        )
        simulation_result["sumo_cmd"] = list(sumo_cmd)

        try:
            traci.start(sumo_cmd)
            scene_injector.bootstrap_after_start()
            _record_debug(
                "SUMO scene injector ready",
                scene_mode=scene_mode,
                runtime_spillover_summary=simulation_result.get("runtime_spillover_summary", {}),
            )

            warmup_target = max(
                int(scene_profile.warmup_seconds),
                int(scene_profile.evaluation_start_time),
            )
            while traci.simulation.getTime() < warmup_target:
                traci.simulationStep()
                scene_injector.on_simulation_step()

            route_debug = build_safe_vehicle_route(
                net,
                original_start_edge=original_start_edge,
                original_end_edge=original_end_edge,
                original_route_edges=path,
                vtype_id=planner_vtype,
                vclass=planner_vclass,
            )
            simulation_result.update(
                {
                    "stage": route_debug.get("stage", "sumo_route_build"),
                    "reason": route_debug.get("reason", "unknown"),
                    "original_start_edge": route_debug.get("original_start_edge", original_start_edge),
                    "final_start_edge": route_debug.get("final_start_edge", original_start_edge),
                    "original_end_edge": route_debug.get("original_end_edge", original_end_edge),
                    "final_end_edge": route_debug.get("final_end_edge", original_end_edge),
                    "original_route_edges": list(route_debug.get("original_route_edges") or path),
                    "final_route_edges": list(route_debug.get("final_route_edges") or []),
                    "route_validation": dict(route_debug.get("route_validation") or {}),
                    "route_repair_actions": list(route_debug.get("repair_actions") or []),
                    "depart_candidates": list(route_debug.get("depart_candidates") or []),
                    "arrival_candidates": list(route_debug.get("arrival_candidates") or []),
                    "used_sumo_fallback": bool(route_debug.get("used_sumo_fallback")),
                    "route_length_edges": len(list(route_debug.get("final_route_edges") or [])),
                    "affected_edge_overlap_ratio": _route_affected_overlap_ratio(
                        route_debug.get("final_route_edges") or []
                    ),
                }
            )
            _record_debug(
                "SUMO route build validation",
                original_start_edge=simulation_result["original_start_edge"],
                final_start_edge=simulation_result["final_start_edge"],
                original_end_edge=simulation_result["original_end_edge"],
                final_end_edge=simulation_result["final_end_edge"],
                vehicle_type=planner_vtype,
                vehicle_vclass=planner_vclass,
                final_route_edges_count=len(simulation_result["final_route_edges"]),
                route_validation_result=simulation_result["route_validation"],
                route_repair_actions=simulation_result["route_repair_actions"],
                reroute_effective=simulation_result["reroute_policy"],
            )
            if not route_debug.get("ok"):
                simulation_result["speed_status"] = "upstream_failure"
                simulation_result["speed_reason"] = route_debug.get("reason", "sumo_route_build_failed")
                simulation_result["error_message"] = (
                    f"无法为 {planner_vehicle_id} 构造合法 SUMO 路径：{simulation_result['speed_reason']}"
                )
                scene_injector.restore_base_state()
                traci.close()
                return simulation_result

            traci.route.add(planner_route_id, simulation_result["final_route_edges"])
            traci.vehicle.add(
                planner_vehicle_id,
                planner_route_id,
                depart="now",
                departLane=depart_lane,
                departPos=depart_pos,
                departSpeed=depart_speed,
                typeID=planner_vtype,
            )
            scene_injector.register_planner_vehicle(planner_vehicle_id)
            simulation_result["vehicle_inserted"] = True
            simulation_result["stage"] = "vehicle_inserted"
            depart_time = float(traci.simulation.getTime())

            step, max_steps = 0, 36000
            active_seen = False
            speed_data, pos_data = [], []
            vehicle_last_speed = {}
            vehicle_stop_count = {}
            evaluation_end_time = max(float(scene_profile.evaluation_end_time), depart_time + step_len)

            while step < max_steps and float(traci.simulation.getTime()) <= evaluation_end_time:
                traci.simulationStep()
                scene_injector.on_simulation_step()
                active_vehicle_ids = traci.vehicle.getIDList()
                if planner_vehicle_id in active_vehicle_ids:
                    simulation_result["vehicle_departed"] = True
                    active_seen = True
                    planner_speed_mps = float(traci.vehicle.getSpeed(planner_vehicle_id))
                    if planner_vehicle_id not in vehicle_last_speed:
                        vehicle_last_speed[planner_vehicle_id] = planner_speed_mps
                        vehicle_stop_count[planner_vehicle_id] = 0
                    else:
                        if vehicle_last_speed[planner_vehicle_id] >= 0.1 and planner_speed_mps < 0.1:
                            vehicle_stop_count[planner_vehicle_id] += 1
                        vehicle_last_speed[planner_vehicle_id] = planner_speed_mps
                    current_speed = float(planner_speed_mps * 3.6)
                    if current_speed > 0.01:
                        speed_data.append(current_speed)
                        pos_data.append(traci.vehicle.getPosition(planner_vehicle_id))
                elif active_seen:
                    simulation_result["vehicle_arrived"] = True
                    break
                step += 1

            travel_time = max(0.0, float(traci.simulation.getTime()) - depart_time)
            avg_speed = float(np.mean(speed_data)) if speed_data else 0.0
            speed_reason = "ok"
            speed_status = "ok"
            if not simulation_result["vehicle_inserted"]:
                speed_status = "upstream_failure"
                speed_reason = "vehicle_not_inserted"
            elif not simulation_result["vehicle_departed"]:
                speed_status = "upstream_failure"
                speed_reason = "vehicle_not_departed"
            elif len(speed_data) < min_speed_samples:
                speed_status = "not_enough_speed_samples"
                speed_reason = "route_too_short_or_simulation_ended_early"

            simulation_result.update(
                {
                    "ok": simulation_result["vehicle_inserted"] and simulation_result["vehicle_departed"],
                    "stage": "simulation_complete" if simulation_result["vehicle_departed"] else "vehicle_depart",
                    "reason": "ok" if simulation_result["vehicle_departed"] else speed_reason,
                    "speed_data": list(speed_data),
                    "pos_data": list(pos_data),
                    "speed_samples_collected": len(speed_data),
                    "speed_status": speed_status,
                    "speed_reason": speed_reason,
                    "travel_time": travel_time,
                    "avg_speed": avg_speed,
                    "stop_count": int(vehicle_stop_count.get(planner_vehicle_id, 0)),
                    "completion_status": (
                        "arrived" if simulation_result["vehicle_arrived"] else "evaluation_window_active"
                    ),
                }
            )
            scene_injector.restore_base_state()
            traci.close()

            for _ in range(15):
                if os.path.exists(tripinfo_path) and os.path.getsize(tripinfo_path) > 10:
                    try:
                        tree = ET.parse(tripinfo_path)
                        for ti in tree.getroot().findall("tripinfo"):
                            if ti.get("id") == planner_vehicle_id:
                                travel_time = float(ti.get("duration", travel_time))
                                spd = float(ti.get("speed", avg_speed / 3.6))
                                avg_speed = spd * 3.6
                                arrival_s = float(ti.get("arrival", float("nan")))
                                vaporized = str(ti.get("vaporized", "") or "").strip()
                                trip_completed = math.isfinite(arrival_s) and arrival_s >= 0 and not vaporized
                                simulation_result["vehicle_arrived"] = bool(trip_completed)
                                simulation_result["simulation_truncated"] = not bool(trip_completed)
                                simulation_result["completion_status"] = (
                                    "arrived" if trip_completed else "truncated_at_evaluation_end"
                                )
                                simulation_result["tripinfo_arrival_s"] = arrival_s
                                simulation_result["tripinfo_vaporized"] = vaporized
                                simulation_result["time_loss_s"] = float(ti.get("timeLoss", 0.0))
                                simulation_result["waiting_time_s"] = float(ti.get("waitingTime", 0.0))
                                simulation_result["depart_delay_s"] = float(ti.get("departDelay", float("nan")))
                                break
                        break
                    except Exception:
                        pass
                time.sleep(0.3)

            simulation_result["travel_time"] = travel_time
            simulation_result["avg_speed"] = avg_speed
            _record_debug(
                "SUMO vehicle lifecycle",
                vehicle_inserted=simulation_result["vehicle_inserted"],
                vehicle_departed=simulation_result["vehicle_departed"],
                vehicle_arrived=simulation_result["vehicle_arrived"],
                speed_samples_collected=simulation_result["speed_samples_collected"],
                simulation_steps=step,
                route_length=len(simulation_result["final_route_edges"]),
                speed_status=simulation_result["speed_status"],
                speed_reason=simulation_result["speed_reason"],
            )
            logging.info(
                "SUMO done: t=%.1fs spd=%.1fkm/h route_edges=%d",
                travel_time,
                avg_speed,
                len(simulation_result["final_route_edges"]),
            )
            return simulation_result

        except Exception as e:
            logging.error("SUMO error: %s", e)
            simulation_result.update(
                {
                    "ok": False,
                    "stage": simulation_result.get("stage") or "sumo_runtime",
                    "reason": str(e),
                    "error_message": str(e),
                    "speed_status": "upstream_failure",
                    "speed_reason": str(e),
                }
            )
            try:
                scene_injector.restore_base_state()
            except Exception:
                pass
            try:
                traci.close()
            except Exception:
                pass
            return simulation_result


# 路网坐标（迁移到 viz.py）
def _load_net_cached(net_path: str):
    """兼容层：委托到 viz.load_net_cached。"""
    return load_net_cached(net_path)


# 可视化引擎
class VisualizationEngine:
    def __init__(self, config):
        self.config = config

    def load_net_coords(self, net_path):
        return _load_net_cached(net_path)

    def build_folium_network_map(
        self,
        weight_dict=None,
        raw_incident_weights=None,
        spillover_weights=None,
        spillover_hops=None,
        gat_smoothed_weights=None,
        overlay_stats=None,
        path=None,
        focus_path=False,
        baseline_path=None,
        planned_time_s=None,
        baseline_time_s=None,
        map_style="realistic_overlay",
        basemap_style="nolabel_light",
        layer_visibility=None,
    ):
        return build_folium_map(
            self.config.get("SUMO_NET_PATH"),
            osm_path=self.config.get("OSM_MAP_PATH", ""),
            weight_dict=weight_dict,
            raw_incident_weights=raw_incident_weights,
            spillover_weights=spillover_weights,
            spillover_hops=spillover_hops,
            gat_smoothed_weights=gat_smoothed_weights,
            overlay_stats=overlay_stats,
            path=path,
            baseline_path=baseline_path,
            planned_time_s=planned_time_s,
            baseline_time_s=baseline_time_s,
            map_style=map_style,
            basemap_style=basemap_style,
            focus_path=focus_path,
            layer_visibility=layer_visibility,
        )

    def export_folium_map(self, map_obj, filename: str):
        output_path = os.path.join(self.config.get("RESULTS_DIR"), filename)
        return export_folium_map(map_obj, output_path)

    def _visual_meta(self):
        return load_roadnet_visual_meta(self.config.get("SUMO_NET_PATH"))

    def _coords_to_xy(self, coords):
        if not coords:
            return [], []
        return [pt[0] for pt in coords] + [None], [pt[1] for pt in coords] + [None]

    def _coords_midpoint(self, coords):
        if not coords:
            return None, None
        pt = coords[len(coords) // 2]
        return pt[0], pt[1]

    def _edge_road_name(self, edge_id: str) -> str:
        return shared_edge_road_name(edge_id)

    def _display_bundle(self, map_style="realistic_overlay"):
        return load_display_geo_bundle(
            self.config.get("SUMO_NET_PATH"),
            self.config.get("OSM_MAP_PATH", ""),
            map_style,
        )

    def _geo_coords_to_xy(self, coords):
        if not coords:
            return [], []
        return [pt[1] for pt in coords] + [None], [pt[0] for pt in coords] + [None]

    def _grid_meta(self):
        return load_roadnet_meta(self.config.get("SUMO_NET_PATH"))

    def _grid_node_xy(self, row_idx: int, col_idx: int):
        return float(col_idx) * 1.12, float(-row_idx)

    def _grid_edge_coords(self, edge_id: str, offset: float = 0.0):
        meta = self._grid_meta().edge_meta.get(edge_id)
        if not meta:
            return []
        axis = meta.get("axis")
        row_idx = int(meta.get("row", 0))
        col_idx = int(meta.get("col", 0))
        if axis == "H":
            west = self._grid_node_xy(row_idx, col_idx)
            east = self._grid_node_xy(row_idx, col_idx + 1)
            coords = [west, east] if meta.get("dir_code") == "E" else [east, west]
            return [(x, y + offset) for x, y in coords]
        north = self._grid_node_xy(row_idx, col_idx)
        south = self._grid_node_xy(row_idx + 1, col_idx)
        coords = [north, south] if meta.get("dir_code") == "S" else [south, north]
        return [(x + offset, y) for x, y in coords]

    def _grid_midpoint(self, edge_id: str, offset: float = 0.0):
        coords = self._grid_edge_coords(edge_id, offset=offset)
        if not coords:
            return None, None
        pt = coords[len(coords) // 2]
        return pt[0], pt[1]

    def _congestion_level(self, weight: float | None) -> str:
        value = _safe_float(weight, 2.0)
        if value >= 9.0:
            return "closed"
        if value >= 7.0:
            return "severe"
        if value >= 4.5:
            return "medium"
        if value >= 2.5:
            return "light"
        return "clear"

    def _congestion_style(self, weight: float | None):
        level = self._congestion_level(weight)
        palette = {
            "clear": ("畅通", "#22C55E", 5, 0.82),
            "light": ("轻度拥堵", "#FACC15", 6, 0.86),
            "medium": ("中度拥堵", "#F97316", 7, 0.9),
            "severe": ("严重拥堵", "#DC2626", 8, 0.94),
            "closed": ("封闭", "#7F1D1D", 9, 0.96),
        }
        return level, palette[level]

    def _road_style(self, edge_id: str):
        meta = self._grid_meta().edge_meta.get(edge_id, {})
        palette = {
            "A": ("主干道", "#52606D", 4.2, 0.78),
            "S": ("次干道", "#7F8EA3", 3.4, 0.66),
            "M": ("支路", "#B7C4D1", 2.4, 0.58),
        }
        return palette.get(meta.get("rtype"), palette["M"])

    def _route_metrics(self, baseline_time_s, planned_time_s):
        if baseline_time_s is None or planned_time_s is None:
            return None
        saved = float(baseline_time_s) - float(planned_time_s)
        ratio = (saved / float(baseline_time_s) * 100.0) if float(baseline_time_s) > 1e-9 else 0.0
        neutral_threshold = max(3.0, abs(float(baseline_time_s)) * 0.003)
        return {
            "baseline": float(baseline_time_s),
            "planned": float(planned_time_s),
            "saved": saved,
            "ratio": ratio,
            "trend": "equal" if abs(saved) <= neutral_threshold else ("better" if saved > 0 else "worse"),
        }

    def _apply_city_background(self, fig):
        meta = self._visual_meta()
        for poly in meta.get("building_polygons", []):
            xs = [pt[0] for pt in poly] + [poly[0][0]]
            ys = [pt[1] for pt in poly] + [poly[0][1]]
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor="rgba(148,163,184,0.08)",
                line=dict(color="rgba(148,163,184,0.18)", width=1),
                hoverinfo="skip",
                showlegend=False,
            ))
        return meta

    def _plot_grid_route_map(
        self,
        congestion_dict,
        path,
        *,
        baseline_path=None,
        focus_path=True,
        planned_time_s=None,
        baseline_time_s=None,
        map_style="realistic_overlay",
    ):
        map_style = "realistic_overlay" if str(map_style).strip().lower().startswith("realistic") else "clean_schematic"
        display_bundle = self._display_bundle(map_style)
        meta = self._grid_meta()
        fig = go.Figure()
        bg_color = "#F8FAFC" if map_style == "clean_schematic" else "#EEF2F5"
        block_fill = "rgba(241,245,249,0.92)" if map_style == "clean_schematic" else "rgba(255,255,255,0.58)"

        for poly in display_bundle.get("block_polygons", []):
            xs, ys = self._geo_coords_to_xy(poly + [poly[0]])
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=block_fill,
                line=dict(color="rgba(148,163,184,0.14)", width=1),
                hoverinfo="skip",
                showlegend=False,
            ))

        path = list(path or [])
        baseline_path = list(baseline_path or [])
        route_union = set(path) | set(baseline_path)
        congestion_dict = dict(congestion_dict or {})
        edge_geo_coords = display_bundle.get("edge_geo_coords", {})
        planned_path_line = remap_logical_path_to_display_path(
            path,
            edge_geo_coords=edge_geo_coords,
            offset_m=0.0,
            smooth_strength=0.18 if map_style == "realistic_overlay" else 0.0,
        )
        baseline_path_line = remap_logical_path_to_display_path(
            baseline_path,
            edge_geo_coords=edge_geo_coords,
            offset_m=7.0 if map_style == "realistic_overlay" else 4.0,
            smooth_strength=0.16 if map_style == "realistic_overlay" else 0.0,
        )

        reference_added = {"主干道": False, "次干道": False, "支路": False}
        for edge_id in meta.edge_ids:
            coords = edge_geo_coords.get(edge_id)
            if not coords:
                continue
            road_label, road_color, road_width, road_opacity = self._road_style(edge_id)
            xs, ys = self._geo_coords_to_xy(coords)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=road_color, width=road_width - (0.2 if map_style == "realistic_overlay" else 0.0)),
                opacity=min(road_opacity, 0.72) if map_style == "realistic_overlay" else road_opacity,
                hovertemplate=f"{road_label}<br>{edge_id}<extra></extra>",
                name=road_label,
                showlegend=not reference_added[road_label],
            ))
            reference_added[road_label] = True

        congestion_added = set()
        for edge_id in meta.edge_ids:
            weight = _safe_float(congestion_dict.get(edge_id), 2.0)
            level, (level_label, color, width, opacity) = self._congestion_style(weight)
            if edge_id not in route_union and level == "clear":
                continue
            coords = edge_geo_coords.get(edge_id)
            if not coords:
                continue
            xs, ys = self._geo_coords_to_xy(coords)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=width - (1.2 if map_style == "realistic_overlay" else 0.0)),
                opacity=0.92 if edge_id in route_union else min(opacity, 0.72),
                hovertemplate=(
                    f"<b>{self._edge_road_name(edge_id)}</b><br>{edge_id}"
                    f"<br>{level_label} · score={weight:.1f}<extra></extra>"
                ),
                name=level_label,
                showlegend=level_label not in congestion_added,
            ))
            congestion_added.add(level_label)

        baseline_legend_added = False
        for edge_id in baseline_path:
            weight = _safe_float(congestion_dict.get(edge_id), 2.0)
            level = self._congestion_level(weight)
            coords = list(edge_geo_coords.get(edge_id) or [])
            if not coords:
                continue
            if map_style == "realistic_overlay":
                mean_lat = sum(pt[0] for pt in coords) / len(coords)
                dx = (coords[-1][1] - coords[0][1]) * 111320.0 * math.cos(math.radians(mean_lat))
                dy = (coords[-1][0] - coords[0][0]) * 110540.0
                length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                nx = -dy / length
                ny = dx / length
                dlon = (7.0 * nx) / max(111320.0 * math.cos(math.radians(mean_lat)), 1e-6)
                dlat = (7.0 * ny) / 110540.0
                coords = [(lat + dlat, lon + dlon) for lat, lon in coords]
            xs, ys = self._geo_coords_to_xy(coords)
            if level in {"severe", "closed"}:
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="rgba(127,29,29,0.28)", width=10 if map_style == "realistic_overlay" else 12),
                    hoverinfo="skip",
                    showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#DC2626", width=4, dash="dash"),
                opacity=0.72,
                hovertemplate=f"原始路线<br>{edge_id}<extra></extra>",
                name="原始路线",
                showlegend=not baseline_legend_added,
            ))
            baseline_legend_added = True

        if baseline_path_line:
            bx, by = self._geo_coords_to_xy(baseline_path_line)
            fig.add_trace(go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color="#DC2626", width=4, dash="dash"),
                opacity=0.82,
                hoverinfo="skip",
                showlegend=False,
            ))

        planned_legend_added = False
        for edge_id in path:
            coords = edge_geo_coords.get(edge_id)
            if not coords:
                continue
            xs, ys = self._geo_coords_to_xy(coords)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.76)", width=9 if map_style == "realistic_overlay" else 10),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#2563EB", width=5 if map_style == "realistic_overlay" else 6),
                opacity=0.95,
                hovertemplate=f"规划路线<br>{edge_id}<extra></extra>",
                name="规划路线",
                showlegend=not planned_legend_added,
            ))
            planned_legend_added = True

        if planned_path_line:
            px, py = self._geo_coords_to_xy(planned_path_line)
            fig.add_trace(go.Scatter(
                x=px,
                y=py,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.92)", width=9 if map_style == "realistic_overlay" else 10),
                hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=px,
                y=py,
                mode="lines",
                line=dict(color="#2563EB", width=5 if map_style == "realistic_overlay" else 6),
                opacity=0.98,
                hoverinfo="skip",
                showlegend=False,
            ))

        if planned_path_line:
            start_pt = planned_path_line[0]
            end_pt = planned_path_line[-1]
            fig.add_trace(go.Scatter(
                x=[start_pt[1]], y=[start_pt[0]], mode="markers+text",
                marker=dict(size=9, color="#16A34A", line=dict(color="white", width=2)),
                text=["起点"], textposition="top center",
                textfont=dict(size=11, color="#166534"),
                name="起点", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[end_pt[1]], y=[end_pt[0]], mode="markers+text",
                marker=dict(size=9, color="#F59E0B", line=dict(color="white", width=2)),
                text=["终点"], textposition="top center",
                textfont=dict(size=11, color="#92400E"),
                name="终点", showlegend=False,
            ))

        time_box = self._route_metrics(baseline_time_s, planned_time_s)
        annotations = []
        if time_box:
            if time_box["trend"] == "equal":
                positive_gain = False
                accent = "#64748B"
                delta_label = "时间变化"
                ratio_label = "变化比例"
                status_text = "收益结论：当前案例持平/接近"
            else:
                positive_gain = time_box["saved"] > 0
                accent = "#16A34A" if positive_gain else "#EA580C"
                delta_label = "节省时间" if positive_gain else "增加时间"
                ratio_label = "节省比例" if positive_gain else "增加比例"
                status_text = "收益结论：正收益" if positive_gain else "收益结论：负收益"
            annotations.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    align="left",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.96)",
                    bordercolor=accent,
                    borderwidth=1,
                    borderpad=8,
                    font=dict(size=11, color="#334155"),
                    text=(
                        f"基线路线行程时间：{time_box['baseline']:.1f} s<br>"
                        f"规划路线行程时间：{time_box['planned']:.1f} s<br>"
                        f"<span style='color:{accent};'><b>{delta_label}：{abs(time_box['saved']):.1f} s</b></span><br>"
                        f"{ratio_label}：{abs(time_box['ratio']):.1f}%<br>{status_text}"
                    ),
                )
            )

        fig.update_layout(
            title=dict(
                text="Realistic Block Overlay Map" if map_style == "realistic_overlay" else "Clean Schematic Map",
                font=dict(size=15),
                x=0.5,
                y=0.97,
            ),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor=bg_color,
            paper_bgcolor="white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(148,163,184,0.28)",
                borderwidth=1,
            ),
            annotations=annotations,
            height=460,
            margin=dict(l=18, r=18, t=44, b=18),
        )
        return fig

    def plot_route_plotly(
        self,
        edge_coords,
        path,
        weight_dict,
        *,
        baseline_path=None,
        planned_time_s=None,
        baseline_time_s=None,
        map_style="realistic_overlay",
    ):
        return self._plot_grid_route_map(
            weight_dict,
            path,
            baseline_path=baseline_path,
            focus_path=True,
            planned_time_s=planned_time_s,
            baseline_time_s=baseline_time_s,
            map_style=map_style,
        )

    def plot_heatmap_plotly(self, edge_coords, weight_dict):
        fig = go.Figure()
        self._apply_city_background(fig)
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None:
                continue
            xs, ys = self._coords_to_xy(coords)
            w = weight_dict.get(eid, 1.0)
            r = int(255 * min(w / 10.0, 1.0))
            g = int(255 * (1 - min(w / 10.0, 1.0)) * 0.3)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=f"rgb({r},{g},0)", width=max(2, w * 0.8)),
                hovertemplate=f"<b>{self._edge_road_name(eid)}</b><br>{eid}<br>Weight: {w:.1f}<extra></extra>",
                showlegend=False,
            ))
        # 修正 4：Plotly 5.x/6.x 兼容：color 长度与 x/y 一致，cmin/cmax 驱动色条
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
            plot_bgcolor="#F8FAFC", paper_bgcolor="white",
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
        methods = [
            "Dijkstra", "Rule-A*", "DQN-RouteSelector", "GCN-Weight", "Raw-Qwen", "CoT-Qwen",
            "Qwen-LoRA", "Qwen-LoRA + GAT (exploratory)", "R1-Raw", "R1-LoRA",
            MAINLINE_METHOD_NAME, "Sparse-LoRA-v2 + GAT (exploratory)",
        ]
        times   = [round(base*1.10,1), round(base*0.95,1), round(base*0.88,1), round(base*0.82,1),
                   round(base*1.05,1), round(base*0.92,1), round(base*0.85,1), round(base*0.83,1),
                   round(base*0.98,1), round(base*1.08,1), round(base*0.75,1), round(float(your_time),1)]
        rates   = [32, 62, 62, 62, 32, 35, 42, 42, 35, 43, 68, 68]
        costs   = [round(base*0.85,1), round(base*1.20,1), round(base*1.35,1), round(base*1.15,1),
                   round(base*0.90,1), round(base*0.95,1), round(base*1.05,1), round(base*1.08,1),
                   round(base*0.92,1), round(base*0.98,1), round(base*1.10,1), round(float(your_cost),2)]
        colors  = [METHOD_COLORS.get(m, "#94A3B8") for m in methods]

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

    def plot_network_overview(
        self,
        edge_coords,
        weight_dict=None,
        path=None,
        *,
        baseline_path=None,
        map_style="realistic_overlay",
    ):
        fig = self._plot_grid_route_map(
            weight_dict,
            path,
            baseline_path=baseline_path,
            focus_path=False,
            planned_time_s=None,
            baseline_time_s=None,
            map_style=map_style,
        )
        fig.update_layout(height=420)
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
            if not coords or coords[0] is None:
                continue
            xs, ys = self._coords_to_xy(coords)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines", line=dict(color="#E5E7EB", width=2),
                showlegend=False, hoverinfo="skip",
            ))
        if path_dijkstra:
            dx, dy = [], []
            for eid in path_dijkstra:
                c = edge_coords.get(eid)
                if c and c[0]:
                    xs, ys = self._coords_to_xy(c)
                    dx += xs; dy += ys
            fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines",
                line=dict(color="#3B82F6", width=4, dash="dash"),
                name="Dijkstra (Fixed Weight)"))
        if path_llm:
            lx, ly = [], []
            for eid in path_llm:
                c = edge_coords.get(eid)
                if c and c[0]:
                    xs, ys = self._coords_to_xy(c)
                    lx += xs; ly += ys
                    mx, my = self._coords_midpoint(c)
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

    # 真实对比实验可视化
    def _normalize_method_label(self, method_name: str) -> str:
        """统一展示方法名，兼容旧版结果中的别名写法。"""
        return canonical_method_name(method_name)

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
        """统一展示记录字段映射，兼容 legacy 字段 parse_rate/coverage/parsed_ratio/plan_ms/time_s/sumo_time。"""
        total_edges = int(
            rec.get("total_edges")
            or st.session_state.get("edge_total_dynamic")
            or len(getattr(path_engine, "_ALL_EDGES", []))
            or 96
        )
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

        timing_fields = normalize_eval_time_fields(rec)
        planning_time_s = timing_fields.get("planning_time_s")
        if planning_time_s is None:
            planning_time_s = rec.get("time_s")

        return {
            "cost": float(rec.get("path_cost", 0) or 0),
            "constraint": float(rec.get("constraint_rate", 0) or 0),
            "signal_ratio": signal_ratio,
            "parsed_edge_ratio": parsed_ratio,
            "parsed_edges": parsed_edges,
            "signal_edges": signal_edges,
            "default_edges": default_edges,
            "time": float(planning_time_s or 0),
            "sumo": float(timing_fields.get("travel_time_s") or rec.get("sumo_time", 0) or 0),
        }

    def plot_real_compare(self, results_path: str, current_cost: float,
                          current_constraint: float, current_time: float):
        """读取真实对比实验结果并可视化；结果缺失时仅给出明确提示，不做估算回落。"""
        import os, json

        # 使用全局定义的颜色和方法列表

        if not os.path.exists(results_path):
            msg = "未检测到对比实验结果，请先运行对比实验后再展示该页面。"

            def _empty_fig(title_text: str):
                fig = go.Figure()
                fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
                fig.update_layout(title=title_text, template="plotly_white", height=320)
                return fig

            empty_df = pd.DataFrame(columns=[
                "排名", "方法", "Travel Time (s)↓", "约束符合率(%)↑", "非默认路况占比(%)↑", "Planning Time (s)↓", "路径代价↓"
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

        known = [m for m in ALL_METHODS if m in stats]
        extra = [m for m in stats if m not in ALL_METHODS]
        methods = known + extra

        if not methods:
            empty_df = pd.DataFrame(columns=[
                "排名", "方法", "Travel Time (s)↓", "约束符合率(%)↑", "非默认路况占比(%)↑", "Planning Time (s)↓", "路径代价↓"
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
        y_label = "Travel Time (s) ↓" if has_sumo else "路径代价"

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
        cats = ["路径质量", "约束符合", "非默认路况", "Planning效率", "路径效率", "路径质量"]
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
            title=dict(text="Planning Time (s) vs 约束符合率（左上角=理想区域）", font=dict(size=13), x=0.5),
            xaxis=dict(title="Planning Time (s)", showgrid=True, gridcolor="#F3F4F6", range=[-2, max(times) * 1.1 + 2]),
            yaxis=dict(title="约束符合率 (%)", range=[0, 115], showgrid=True, gridcolor="#F3F4F6"),
            plot_bgcolor="white", paper_bgcolor="white", height=320, margin=dict(l=50, r=30, t=50, b=50),
        )

        box_fig = go.Figure()
        for m in methods:
            vals = sumo_by_method.get(m, [])
            if not vals:
                continue
            is_ours = _IS_OURS(m)
            box_fig.add_trace(go.Violin(
                y=[float(v) for v in vals],
                name=m,
                marker_color=METHOD_COLORS.get(m, "#94A3B8"),
                line=dict(width=2.5 if is_ours else 1.5),
                fillcolor=METHOD_COLORS.get(m, "#94A3B8"),
                opacity=0.85 if is_ours else 0.55,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.3,
                pointpos=0,
                marker=dict(size=7 if is_ours else 5, line=dict(color="white", width=1)),
            ))
        box_fig.update_layout(
            title=dict(text="各场景 Travel Time (s) 分布（小提琴+箱线，越低越稳定）", font=dict(size=13), x=0.5),
            yaxis=dict(title="Travel Time (s)", showgrid=True, gridcolor="#F3F4F6"),
            xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
            plot_bgcolor="white", paper_bgcolor="white", height=400, showlegend=False,
            margin=dict(l=55, r=30, t=55, b=100),
        )

        scene_bar_fig = None
        if scenario_data:
            scene_names_full = [s["scenario"] for s in scenario_data]
            scene_names_short = [name.split(" ")[0] if name else f"场景{i + 1}" for i, name in enumerate(scene_names_full)]
            scene_methods = [m for m in methods if any(m in s["methods"] for s in scenario_data)]
            dense_scene_chart = len(scene_methods) > 6
            scene_bar_fig = go.Figure()
            for m in scene_methods:
                y_vals = [
                    s["methods"].get(m, {}).get("sumo") if m in s["methods"] else None
                    for s in scenario_data
                ]
                is_ours = _IS_OURS(m)
                scene_bar_fig.add_trace(go.Bar(
                    name=m,
                    x=scene_names_short,
                    y=[float(v) if v is not None else None for v in y_vals],
                    marker_color=METHOD_COLORS.get(m, "#94A3B8"),
                    opacity=0.94 if is_ours else 0.72,
                    marker_line=dict(color="#7F1D1D" if is_ours else "rgba(0,0,0,0)", width=2.2 if is_ours else 0),
                    text=[f"{v:.0f}s" if v is not None and (is_ours or not dense_scene_chart) else "" for v in y_vals],
                    textposition="outside",
                    textfont=dict(size=9, color="#7F1D1D" if is_ours else "#334155"),
                    customdata=scene_names_full,
                    hovertemplate="%{fullData.name}<br>%{customdata}<br>SUMO: %{y:.1f}s<extra></extra>",
                ))
            scene_bar_fig.update_layout(
                title=dict(text="分场景 Travel Time (s) 对比（完整方法集，越低越好）", font=dict(size=13), x=0.5),
                barmode="group",
                bargap=0.12,
                bargroupgap=0.04,
                yaxis=dict(title="Travel Time (s)", showgrid=True, gridcolor="#F3F4F6"),
                xaxis=dict(tickangle=0, tickfont=dict(size=10)),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.24, x=0.5, xanchor="center", font=dict(size=10)),
                height=460, margin=dict(l=55, r=30, t=60, b=115),
            )

        heatmap_fig = None
        if scenario_data:
            sc_names_short = [
                s["scenario"].split(" ")[0] + (f" {s['scenario'].split(' ')[1]}" if len(s["scenario"].split(" ")) > 1 else "")
                for s in scenario_data
            ]
            heat_methods = [m for m in methods if any(m in s["methods"] for s in scenario_data)]
            z_matrix = []
            z_norm = []
            text_matrix = []
            for m in heat_methods:
                row = [
                    s["methods"].get(m, {}).get("sumo") if m in s["methods"] else None
                    for s in scenario_data
                ]
                valid_vals = [float(v) for v in row if v is not None and float(v) > 0]
                if valid_vals:
                    mn, mx = min(valid_vals), max(valid_vals)
                    same_value = abs(mx - mn) < 1e-6
                    rng = (mx - mn) if not same_value else 1.0
                    norm_row = [
                        None if v is None else (0.5 if same_value else (float(v) - mn) / rng)
                        for v in row
                    ]
                else:
                    norm_row = [None for _ in row]
                z_matrix.append([float(v) if v is not None else None for v in row])
                z_norm.append(norm_row)
                text_matrix.append([f"{float(v):.0f}s" if v is not None else "" for v in row])

            heatmap_fig = go.Figure(go.Heatmap(
                z=z_norm,
                x=sc_names_short,
                y=heat_methods,
                customdata=z_matrix,
                colorscale=[[0.0, "#166534"], [0.25, "#4ADE80"], [0.5, "#FACC15"], [0.75, "#FB923C"], [1.0, "#B91C1C"]],
                zmin=0,
                zmax=1,
                colorbar=dict(title="相对时间<br>（绿=快，红=慢）", tickvals=[0, 0.5, 1], ticktext=["最快", "中等", "最慢"]),
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=10),
                hovertemplate="方法: %{y}<br>场景: %{x}<br>SUMO: %{customdata:.1f}s<extra></extra>",
                hoverongaps=False,
            ))
            heatmap_fig.update_layout(
                title=dict(text="方法 × 场景 Travel Time (s) 热力图（完整方法集，绿=快，红=慢）", font=dict(size=13), x=0.5),
                height=max(350, len(heat_methods) * 38 + 100),
                margin=dict(l=160, r=80, t=60, b=60), paper_bgcolor="white",
                yaxis=dict(tickfont=dict(size=10)), xaxis=dict(tickfont=dict(size=10)),
            )

        sumo_rank = sorted(range(len(methods)), key=lambda i: sumo_t[i])
        rank_map = {methods[i]: r + 1 for r, i in enumerate(sumo_rank)}
        df = pd.DataFrame({
            "排名": [rank_map[m] for m in methods],
            "方法": [str(m) for m in methods],
            "Travel Time (s)↓": [round(float(v), 1) for v in sumo_t],
            "约束符合率(%)↑": [round(float(v), 1) for v in constr],
            "非默认路况占比(%)↑": [round(float(v), 1) for v in signal_ratio_vals],
            "Planning Time (s)↓": [round(float(v), 2) for v in times],
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
                    metric_rec = normalize_edge_metric_record(
                        m,
                        total_edges=(
                            m.get("total_edges")
                            or st.session_state.get("edge_total_dynamic")
                            or len(getattr(path_engine, "_ALL_EDGES", []))
                            or 96
                        ),
                        method_name=m.get("method"),
                    )
                    large_stats[m.get("method", "")].append(float(metric_rec["parsed_edge_ratio"] or 0))

        display_map = {
            "Qwen-LoRA★": "Qwen-LoRA",
            "R1-LoRA★": "R1-LoRA",
            "Sparse-LoRA★★": MAINLINE_METHOD_NAME,
            "Sparse-LoRA+GAT★★★": "Sparse-LoRA-v2 + GAT (exploratory)",
        }
        method_order = ["Sparse-LoRA★★", "Sparse-LoRA+GAT★★★", "Qwen-LoRA★", "R1-LoRA★"]

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

# 数据导出引擎
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


# 全局实例化
config        = ConfigManager()
model_manager = ModelManager(config)
path_engine   = PathPlanningEngine(config)
sumo_engine   = SUMOSimulationEngine(config)
viz_engine    = VisualizationEngine(config)
export_engine = DataExportEngine(config)


# 主界面

def _safe_table(df):
    """统一走安全的 Streamlit DataFrame 渲染链路。"""
    try:
        render_safe_streamlit_table(
            df,
            debug=False,
            debug_title="safe_table",
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        fallback_df, _ = sanitize_dataframe_for_streamlit(df)
        if fallback_df.empty:
            st.info("No data available")
        else:
            st.markdown(safe_table_markdown(fallback_df))

def _compare_result_coverage(path: str):
    """统计结果文件的方法覆盖度，优先选择完整方法集而不是最新的局部实验。"""
    import json

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return (0, 0.0, 0)

    if not isinstance(raw, list):
        return (0, 0.0, 0)

    unique_methods = set()
    scenario_count = 0
    total_methods = 0
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        method_count = 0
        for rec in tc.get("methods", []):
            if not isinstance(rec, dict):
                continue
            name = canonical_method_name(rec.get("method", ""))
            if not name:
                continue
            unique_methods.add(name)
            method_count += 1
        if method_count:
            scenario_count += 1
            total_methods += method_count

    avg_methods = total_methods / scenario_count if scenario_count else 0.0
    return (len(unique_methods), avg_methods, scenario_count)


def _compare_result_schema_score(path: str):
    """Prefer protocolized compare artifacts over legacy starred/old-timing outputs."""
    import json

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return (0, 0, 0)

    if not isinstance(raw, list) or not raw:
        return (0, 0, 0)

    protocolized_rows = 0
    canonical_method_rows = 0
    method_records = 0
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        methods = tc.get("methods", [])
        if not isinstance(methods, list):
            continue
        row_protocolized = False
        row_canonical = True
        for rec in methods:
            if not isinstance(rec, dict):
                continue
            method_records += 1
            method_name = str(rec.get("method", ""))
            if "★" in method_name:
                row_canonical = False
            if {"planning_time_s", "model_infer_time_s", "route_solve_time_s"} & set(rec.keys()):
                row_protocolized = True
        if row_protocolized:
            protocolized_rows += 1
        if row_canonical and methods:
            canonical_method_rows += 1
    return (protocolized_rows, canonical_method_rows, method_records)


def _latest_compare_results_path():
    """优先选择协议化且方法覆盖完整的 compare_results.json，再按时间取最新。"""
    import glob

    patterns = [
        "/root/autodl-tmp/results/runs/*__compare_multi_method__*/compare_results.json",
        "/root/autodl-tmp/results/runs/*__compare_multi_method_live_scene_profile__*/compare_results.json",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    legacy = "/root/autodl-tmp/results/compare_results.json"
    if os.path.exists(legacy):
        candidates.append(legacy)

    unique_candidates = list(dict.fromkeys(candidates))
    if not unique_candidates:
        return legacy

    return max(
        unique_candidates,
        key=lambda path: (
            *_compare_result_schema_score(path),
            *_compare_result_coverage(path),
            os.path.getmtime(path),
        ),
    )


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _format_seconds_text(value, *, precision: int = 2, fallback: str = "No data available") -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return fallback
    return f"{parsed:.{precision}f} s"


def _format_time_window_text(start_value, end_value, *, fallback: str = "No data available") -> str:
    start = _optional_float(start_value)
    end = _optional_float(end_value)
    if start is None or end is None:
        return fallback

    def _fmt_number(number: float) -> str:
        if float(number).is_integer():
            return str(int(number))
        return f"{number:.2f}".rstrip("0").rstrip(".")

    return f"{_fmt_number(start)}-{_fmt_number(end)} s"


def _format_named_time_text(label: str, value, *, precision: int = 2, fallback: str = "No data available") -> str:
    return f"{label}: {_format_seconds_text(value, precision=precision, fallback=fallback)}"


def _format_scene_time_context_text(scene_env_payload) -> str:
    payload = dict(scene_env_payload or {})
    current_scene_time = (
        payload.get("current_scene_time_s")
        if payload.get("current_scene_time_s") is not None
        else payload.get("scene_time_s")
    )
    parts = [
        f"Current Scene Time: {_format_seconds_text(current_scene_time)}",
        f"Warmup: {_format_seconds_text(payload.get('warmup_seconds'), precision=0)}",
        f"Evaluation Window: {_format_time_window_text(payload.get('evaluation_start_time'), payload.get('evaluation_end_time'))}",
    ]
    return " · ".join(parts)


FROZEN_METHOD_METRIC_SPECS = {
    "Travel Time (s)": {"direction": "min", "precision": 2, "label": "Travel Time (s)"},
    "Planning Time (s)": {"direction": "min", "precision": 4, "label": "Planning Time (s)"},
    "Constraint Rate (%)": {"direction": "max", "precision": 2, "label": "Constraint Rate (%)"},
    "Signal Ratio (%)": {"direction": "max", "precision": 2, "label": "Signal Ratio (%)"},
    "model_infer_time_s": {"direction": "min", "precision": 4, "label": "Model Infer Time (s)"},
    "route_solve_time_s": {"direction": "min", "precision": 6, "label": "Route Solve Time (s)"},
}

FROZEN_ABLATION_METRIC_SPECS = {
    "Travel Time (s)": {"direction": "min", "precision": 2, "label": "Travel Time (s)"},
    "Planning Time (s)": {"direction": "min", "precision": 4, "label": "Planning Time (s)"},
    "Constraint Rate (%)": {"direction": "max", "precision": 2, "label": "Constraint Rate (%)"},
    "Signal Ratio (%)": {"direction": "max", "precision": 2, "label": "Signal Ratio (%)"},
    "Parsed Edge Ratio (%)": {"direction": "max", "precision": 2, "label": "Parsed Edge Ratio (%)"},
}

FROZEN_COLUMN_LABELS = {
    "Method Display": "Method",
    "Method (CN)": "方法名称",
    "Variant Display": "Variant",
    "Variant (CN)": "变体名称",
    "Category": "Category",
    "Travel Time rank": "Travel Time rank",
    "Compliance rank": "Compliance rank",
    "Structured Input": "时空增强输入",
    "LLM Component": "LLM Component",
    "GAT Component": "GAT Component",
    "Sparse Mode": "Sparse Mode",
    "Scenes": "Scenes",
    "Δ vs Sparse-LoRA-v2": "Δ vs Sparse-LoRA-v2",
    "Δ vs Prev TT": "Δ vs Prev TT (s)",
    "Δ vs Full TT": "Δ vs Frozen Mainline TT (s)",
    "Δ vs Full Compliance": "Δ vs Frozen Mainline CR (pts)",
    "Ablation badges": "Evidence Badge",
    "Notes": "Notes",
    "Status": "Status",
    "Source File": "Source File",
    "Success Rate": "Success Rate",
    "Average Reward": "Average Reward",
    "Failure Rate": "Failure Rate",
    "Timeout Rate": "Timeout Rate",
    "Training Steps": "Training Steps",
    "Eval Episodes": "Eval Episodes",
    "Train Wall Time (s)": "Train Wall Time (s)",
    "Device": "Device",
    "Protocol": "Protocol",
}

ABLATED_VARIANT_DISPLAY = {
    "Qwen-LoRA": "Qwen-LoRA",
    "LoRA+GAT": "LoRA+GAT (exploratory)",
    "No-SpatioTemporal": "No-SpatioTemporal",
    "Sparse-LoRA": MAINLINE_METHOD_NAME,
    "Sparse-LoRA+GAT": "Sparse-LoRA-v2 + GAT (exploratory)",
}


def _format_freeze_timestamp(ts_text: str | None) -> str:
    return ts_text or "Unavailable"


def _column_label(column_name: str) -> str:
    return FROZEN_COLUMN_LABELS.get(column_name, column_name)


def _rank_metric(series: pd.Series, direction: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ascending = direction == "min"
    return numeric.rank(method="min", ascending=ascending)


def _metric_highlights(series: pd.Series, direction: str):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return set(), set()
    unique_vals = sorted(numeric.unique(), reverse=(direction == "max"))
    best = unique_vals[0]
    second = unique_vals[1] if len(unique_vals) > 1 else None
    best_idx = set(numeric[numeric == best].index.tolist())
    second_idx = set(numeric[numeric == second].index.tolist()) if second is not None else set()
    return best_idx, second_idx


def _format_numeric(value, precision: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{precision}f}"


def _format_delta(delta_value, precision: int = 2) -> str:
    if delta_value is None or pd.isna(delta_value):
        return ""
    delta = float(delta_value)
    sign = "+" if delta > 0 else ""
    delta_class = "paper-delta-positive" if delta > 0 else "paper-delta-negative" if delta < 0 else "paper-delta-neutral"
    return f' <span class="paper-delta {delta_class}">{sign}{delta:.{precision}f}</span>'


def _render_frozen_artifact_meta(bundle, badge_text: str, key_prefix: str, debug: bool = False):
    st.markdown(
        f"""
        <div class="info-strip frozen-source-strip">
        <b>{html.escape(badge_text)}</b><br>
        CSV: <code>{html.escape(str(bundle.csv_artifact.path))}</code><br>
        Last modified: <b>{html.escape(_format_freeze_timestamp(bundle.csv_artifact.last_modified))}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_lines = [
        "- Binding policy: exact preferred frozen CSV path",
        f"- CSV: `{bundle.csv_artifact.path}`",
        f"- CSV modified: `{_format_freeze_timestamp(bundle.csv_artifact.last_modified)}`",
    ]
    if bundle.csv_artifact.resolved_by_search:
        source_lines.append(f"- CSV resolved by nearest filename search from `{bundle.csv_artifact.preferred_path}`")
    if bundle.tex_artifact and bundle.tex_artifact.exists:
        source_lines.append(f"- LaTeX: `{bundle.tex_artifact.path}`")
        source_lines.append(f"- LaTeX modified: `{_format_freeze_timestamp(bundle.tex_artifact.last_modified)}`")
        if bundle.tex_artifact.resolved_by_search:
            source_lines.append(f"- LaTeX resolved by nearest filename search from `{bundle.tex_artifact.preferred_path}`")
    if bundle.md_artifact and bundle.md_artifact.exists:
        source_lines.append(f"- Markdown preview: `{bundle.md_artifact.path}`")
        source_lines.append(f"- Markdown modified: `{_format_freeze_timestamp(bundle.md_artifact.last_modified)}`")
        if bundle.md_artifact.resolved_by_search:
            source_lines.append(f"- Markdown resolved by nearest filename search from `{bundle.md_artifact.preferred_path}`")
    st.caption("\n".join(source_lines))
    if debug:
        with st.expander(f"Debug: {bundle.key} frozen source", expanded=False):
            st.caption(f"path={bundle.csv_artifact.path}")
            st.caption(f"rows={len(bundle.dataframe)} | cols={len(bundle.dataframe.columns)}")
            st.caption(f"columns={list(map(str, bundle.dataframe.columns))}")

    dl_cols = st.columns(3)
    with dl_cols[0]:
        with open(bundle.csv_artifact.path, "rb") as csv_file:
            st.download_button(
                "导出 CSV",
                data=csv_file.read(),
                file_name=bundle.csv_artifact.path.name,
                mime="text/csv",
                use_container_width=True,
                key=f"{key_prefix}_csv_dl",
            )
    with dl_cols[1]:
        if bundle.md_artifact and bundle.md_artifact.exists:
            with open(bundle.md_artifact.path, "rb") as md_file:
                st.download_button(
                    "导出 Markdown",
                    data=md_file.read(),
                    file_name=bundle.md_artifact.path.name,
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"{key_prefix}_md_dl",
                )
        else:
            st.button("导出 Markdown", disabled=True, use_container_width=True, key=f"{key_prefix}_md_disabled")
    with dl_cols[2]:
        if bundle.tex_artifact and bundle.tex_artifact.exists:
            with open(bundle.tex_artifact.path, "rb") as tex_file:
                st.download_button(
                    "导出 LaTeX",
                    data=tex_file.read(),
                    file_name=bundle.tex_artifact.path.name,
                    mime="text/plain",
                    use_container_width=True,
                    key=f"{key_prefix}_tex_dl",
                )
        else:
            st.button("导出 LaTeX", disabled=True, use_container_width=True, key=f"{key_prefix}_tex_disabled")


def _build_badge_html(text: str, tone: str = "neutral") -> str:
    tone_map = {
        "neutral": "badge-neutral",
        "positive": "badge-positive",
        "warning": "badge-warning",
        "main": "badge-main",
    }
    tone_class = tone_map.get(tone, "badge-neutral")
    return f'<span class="paper-badge {tone_class}">{html.escape(text)}</span>'


def _render_scope_banner(title: str, body_lines: list[str]):
    safe_lines = [html.escape(str(line)) for line in body_lines if str(line).strip()]
    body_html = "<br>".join(safe_lines)
    st.markdown(
        f"""
        <div class="info-strip">
        <b>{html.escape(title)}</b><br>
        {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


SHOWCASE_PRESENTATION = {
    "blockage_detour_showcase": {
        "title": "Showcase: Blockage Detour",
        "subtitle": "Positive Gain Showcase",
        "summary_lines": [
            "基线路径穿过主受扰走廊。",
            "规划路径沿替代走廊绕行。",
            "在实际 SUMO 中获得正收益。",
        ],
    },
    "directional_asymmetry_showcase": {
        "title": "Showcase: Directional Asymmetry",
        "subtitle": "Positive Gain Showcase",
        "summary_lines": [
            "基线路径对方向性受扰更敏感。",
            "规划路径更好地利用方向不对称下的替代走廊。",
            "在实际 SUMO 中获得正收益。",
        ],
    },
    "compound_disaster_showcase": {
        "title": "Showcase: Compound Disaster",
        "subtitle": "Neutral Diagnostic Case",
        "summary_lines": [
            "基线路径更容易穿过脆弱走廊。",
            "规划路径会主动避开脆弱走廊。",
            "该案例主要用于解释脆弱走廊与替代走廊关系。",
        ],
    },
    "propagation_range_showcase": {
        "title": "Showcase: Propagation Range",
        "subtitle": "Calibration / Hidden / Debug",
        "summary_lines": [
            "该案例仅保留在 calibration / hidden / debug 区。",
            "不进入 production showcase 默认展示。",
        ],
    },
}


def _resolve_showcase_presentation(display_payload) -> dict[str, Any]:
    showcase_name = str(display_payload.get("showcase_case_name") or "").strip()
    default_title = f"Showcase: {str(display_payload.get('scene_profile') or 'Current Scenario').replace('_', ' ').title()}"
    default_subtitle = "Structured Showcase" if display_payload.get("showcase_structured_mode") else "Current Scenario"
    default_lines = []
    display_reason = str(display_payload.get("showcase_display_reason") or "").strip()
    if display_reason:
        default_lines.append(display_reason)
    return {
        "title": default_title,
        "subtitle": default_subtitle,
        "summary_lines": default_lines,
        **dict(SHOWCASE_PRESENTATION.get(showcase_name) or {}),
    }


def _showcase_gain_snapshot(
    baseline_time: float | None,
    planned_time: float | None,
) -> dict[str, Any]:
    if baseline_time is None or planned_time is None:
        return {
            "status": "unavailable",
            "delta_time": None,
            "delta_ratio_pct": None,
            "conclusion": "收益结论：暂不可用",
        }
    baseline_value = float(baseline_time)
    planned_value = float(planned_time)
    delta_time = baseline_value - planned_value
    delta_ratio_pct = abs(delta_time) / max(baseline_value, 1.0) * 100.0
    neutral_threshold = max(3.0, abs(baseline_value) * 0.003)
    if abs(delta_time) <= neutral_threshold:
        return {
            "status": "neutral",
            "delta_time": abs(delta_time),
            "delta_ratio_pct": delta_ratio_pct,
            "conclusion": "收益结论：当前案例持平/接近",
        }
    if delta_time > 0:
        return {
            "status": "positive",
            "delta_time": abs(delta_time),
            "delta_ratio_pct": delta_ratio_pct,
            "conclusion": "收益结论：正收益",
        }
    return {
        "status": "negative",
        "delta_time": abs(delta_time),
        "delta_ratio_pct": delta_ratio_pct,
        "conclusion": "收益结论：负收益",
    }


def _showcase_time_panel_bundle(
    *,
    baseline_actual_time: float | None,
    planned_actual_time: float | None,
    baseline_fallback_time: float | None,
    planned_fallback_time: float | None,
) -> dict[str, Any]:
    use_actual = baseline_actual_time is not None and planned_actual_time is not None
    baseline_time = float(baseline_actual_time) if use_actual else (
        None if baseline_fallback_time is None else float(baseline_fallback_time)
    )
    planned_time = float(planned_actual_time) if use_actual else (
        None if planned_fallback_time is None else float(planned_fallback_time)
    )
    gain = _showcase_gain_snapshot(baseline_time, planned_time)
    return {
        "use_actual": use_actual,
        "baseline_time": baseline_time,
        "planned_time": planned_time,
        "baseline_label": "基线路线行程时间" if use_actual else "基线路线预计行程时间",
        "planned_label": "规划路线行程时间" if use_actual else "规划路线预计行程时间",
        "gain": gain,
    }


def _build_benefit_box_payload(
    display_payload,
    *,
    comparison_baseline_time,
    comparison_planned_time,
    comparison_source: str,
) -> dict[str, Any]:
    actual_baseline_time = display_payload.get("baseline_sumo_travel_time")
    actual_planned_time = display_payload.get("planned_sumo_travel_time")
    use_actual = actual_baseline_time is not None and actual_planned_time is not None

    baseline_time = actual_baseline_time if use_actual else comparison_baseline_time
    planned_time = actual_planned_time if use_actual else comparison_planned_time
    if baseline_time is None or planned_time is None:
        return {
            "status": "unavailable",
            "title": "Benefit Summary",
            "lines": ["当前未获得可比较的旅行时间结果。"],
            "secondary": None,
        }

    baseline_time = float(baseline_time)
    planned_time = float(planned_time)
    gain_snapshot = _showcase_gain_snapshot(baseline_time, planned_time)
    delta_time = float(gain_snapshot.get("delta_time") or 0.0)
    delta_ratio_pct = float(gain_snapshot.get("delta_ratio_pct") or 0.0)
    structured_saving_seconds = display_payload.get("showcase_saving_seconds")
    structured_saving_ratio = display_payload.get("showcase_saving_ratio")
    if display_payload.get("showcase_structured_mode"):
        if structured_saving_seconds is not None:
            delta_time = abs(float(structured_saving_seconds))
        if structured_saving_ratio is not None:
            delta_ratio_pct = float(structured_saving_ratio) * 100.0
    baseline_label = "基线路线行程时间" if use_actual else "基线路线预计行程时间"
    planned_label = "规划路线行程时间" if use_actual else "规划路线预计行程时间"

    if gain_snapshot["status"] == "neutral":
        status = "neutral"
        title = "Neutral Diagnostic Case"
        lines = [
            f"{baseline_label}：{baseline_time:.1f} s",
            f"{planned_label}：{planned_time:.1f} s",
            "规划路线实际行程时间与基线持平/接近",
            "收益结论：当前案例持平",
            "该案例主要用于解释脆弱走廊与替代走廊关系",
        ]
    elif gain_snapshot["status"] == "positive":
        status = "positive"
        if (
            _resolve_planned_path_source(display_payload) == "scene_aware_shortest_path"
            and int(display_payload.get("anchor_count", 0) or 0) == 0
        ):
            title = "Showcase Route Gain"
        elif int(display_payload.get("anchor_count", 0) or 0) > 0:
            title = "Sparse-LoRA-v2 Route Gain"
        else:
            title = "Positive Gain Showcase"
        lines = [
            f"{baseline_label}：{baseline_time:.1f} s",
            f"{planned_label}：{planned_time:.1f} s",
            f"节省时间：{delta_time:.1f} s",
            f"节省比例：{delta_ratio_pct:.1f}%",
            "收益结论：正收益",
        ]
        if title == "Showcase Route Gain":
            lines.append(
                "Gain is produced by scene-aware route selection under the injected SUMO scene profile, not by sparse parser anchors in this run."
            )
    else:
        status = "negative"
        title = "Negative Case"
        lines = [
            f"{baseline_label}：{baseline_time:.1f} s",
            f"{planned_label}：{planned_time:.1f} s",
            f"增加时间：{delta_time:.1f} s",
            "收益结论：负收益",
            "不放在默认 showcase 区",
        ]

    secondary = None
    if not use_actual:
        secondary = f"当前主框优先展示 SUMO 实际行程时间；本次暂回退到 {comparison_source}。"

    return {
        "status": status,
        "title": title,
        "lines": lines,
        "secondary": secondary,
        "baseline_time": baseline_time,
        "planned_time": planned_time,
        "delta_time": abs(delta_time),
        "delta_ratio_pct": delta_ratio_pct,
    }


def _render_benefit_box(payload: dict[str, Any]) -> None:
    tone = {
        "positive": "positive",
        "neutral": "neutral",
        "negative": "warning",
        "unavailable": "neutral",
    }.get(str(payload.get("status") or "neutral"), "neutral")
    line_html = "<br>".join(html.escape(str(line)) for line in list(payload.get("lines") or []))
    badge_html = _build_badge_html(str(payload.get("title") or "Benefit Summary"), tone)
    secondary = str(payload.get("secondary") or "").strip()
    secondary_html = f"<br><span style='color:#64748B'>{html.escape(secondary)}</span>" if secondary else ""
    st.markdown(
        f"""
        <div class="result-panel">
        {badge_html}<br>
        <div style="margin-top:10px;line-height:1.75">{line_html}{secondary_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_group_strip(groups: list[tuple[str, str]]):
    group_html = "".join(
        f'<span class="group-pill"><b>{html.escape(label)}</b>: {html.escape(desc)}</span>'
        for label, desc in groups
    )
    st.markdown(f'<div class="group-strip">{group_html}</div>', unsafe_allow_html=True)


def _is_rank_like_column(column_name: str) -> bool:
    lowered = str(column_name).strip().lower()
    return ("rank" in lowered) or ("排名" in str(column_name))


def _should_force_string_column(column_name: str) -> bool:
    name = str(column_name).strip().lower()
    exact = {
        "variant",
        "variant display",
        "variant (cn)",
        "method",
        "method display",
        "method (cn)",
        "category",
        "notes",
        "protocol",
        "source file",
        "device",
        "status",
        "gAT component".lower(),
        "llm component",
        "sparse mode",
        "structured input",
        "ablation badges",
        "evidence badge",
    }
    if name in exact:
        return True
    keywords = (
        "variant",
        "method",
        "rank",
        "排名",
        "badge",
        "note",
        "protocol",
        "source",
        "path",
        "component",
        "display",
    )
    return any(token in name for token in keywords)


def _flatten_streamlit_columns(columns) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flattened = []
        for col in columns:
            parts = [str(part) for part in col if str(part) not in {"", "None"}]
            flattened.append(" / ".join(parts) if parts else "")
        return flattened
    return [str(col) for col in columns]


def _dedupe_streamlit_columns(columns: list[str]) -> tuple[list[str], dict[str, str]]:
    seen: dict[str, int] = {}
    renamed: dict[str, str] = {}
    deduped = []
    for original in columns:
        base = str(original)
        count = seen.get(base, 0)
        if count == 0:
            deduped_name = base
        else:
            deduped_name = f"{base}__{count + 1}"
            renamed[base if base not in renamed else f"{base}#{count + 1}"] = deduped_name
        seen[base] = count + 1
        deduped.append(deduped_name)
    return deduped, renamed


def _stringify_streamlit_cell(value) -> str:
    if _is_missing_streamlit_value(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        try:
            return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False, default=str)
    return str(value)


def _is_missing_streamlit_value(value) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, (list, tuple, np.ndarray, pd.Series)):
        return False
    return bool(result)


def normalize_table_for_streamlit(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy(deep=True)
    elif isinstance(data, pd.Series):
        df = data.to_frame().reset_index(drop=True)
    elif data is None:
        df = pd.DataFrame()
    elif isinstance(data, dict):
        try:
            df = pd.DataFrame(data)
        except ValueError:
            df = pd.DataFrame([data])
    elif isinstance(data, (list, tuple)):
        if not data:
            df = pd.DataFrame()
        else:
            try:
                df = pd.DataFrame(data)
            except Exception:
                df = pd.DataFrame({"value": [_stringify_streamlit_cell(item) for item in data]})
    else:
        try:
            df = pd.DataFrame(data)
        except Exception:
            df = pd.DataFrame({"value": [_stringify_streamlit_cell(data)]})

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    return df.copy(deep=True).reset_index(drop=True)


def sanitize_dataframe_for_streamlit(data):
    safe_df = normalize_table_for_streamlit(data)
    debug_info = {
        "input_type": type(data).__name__,
        "original_shape": tuple(safe_df.shape),
        "original_dtypes": {str(col): str(dtype) for col, dtype in safe_df.dtypes.items()},
        "sanitized_dtypes": {},
        "forced_str_columns": [],
        "extension_type_columns": [],
        "mixed_object_columns": [],
        "complex_object_columns": [],
        "duplicate_column_renames": {},
        "multiindex_flattened": isinstance(safe_df.columns, pd.MultiIndex),
        "variant_sample_values": [],
    }

    if safe_df.shape[1] == 0:
        safe_df = pd.DataFrame(columns=["Message"])
        debug_info["created_fallback_schema"] = True
        debug_info["sanitized_dtypes"] = {"Message": "object"}
        return safe_df.iloc[0:0], debug_info

    flat_columns = _flatten_streamlit_columns(safe_df.columns)
    flat_columns = [
        column if str(column).strip() and str(column).strip().lower() not in {"none", "nan"} else f"Column {idx + 1}"
        for idx, column in enumerate(flat_columns)
    ]
    deduped_columns, renamed = _dedupe_streamlit_columns(flat_columns)
    safe_df.columns = deduped_columns
    debug_info["duplicate_column_renames"] = renamed

    for column in safe_df.columns:
        series = safe_df[column]
        dtype = series.dtype

        if _should_force_string_column(column):
            safe_df[column] = series.astype(object).map(
                lambda value: None if _is_missing_streamlit_value(value) else _stringify_streamlit_cell(value)
            )
            debug_info["forced_str_columns"].append(column)
            continue

        if pd.api.types.is_extension_array_dtype(dtype):
            debug_info["extension_type_columns"].append(column)

        is_datetime_tz = isinstance(dtype, pd.DatetimeTZDtype) if hasattr(pd, "DatetimeTZDtype") else False
        if pd.api.types.is_datetime64_any_dtype(dtype) or is_datetime_tz:
            safe_df[column] = series.map(
                lambda value: None if _is_missing_streamlit_value(value) else pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            )
            debug_info["forced_str_columns"].append(column)
            continue

        if _is_rank_like_column(column):
            safe_df[column] = pd.Series(series.map(
                lambda value: "" if _is_missing_streamlit_value(value) else str(int(float(value)))
            ), dtype="string")
            debug_info["forced_str_columns"].append(column)
            continue

        if pd.api.types.is_bool_dtype(dtype):
            safe_df[column] = series.astype(object).map(
                lambda value: None if _is_missing_streamlit_value(value) else bool(value)
            )
            continue

        if pd.api.types.is_integer_dtype(dtype):
            safe_df[column] = series.astype(object).map(
                lambda value: None if _is_missing_streamlit_value(value) else int(value)
            )
            continue

        if pd.api.types.is_float_dtype(dtype):
            safe_df[column] = pd.to_numeric(series, errors="coerce")
            continue

        if pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype):
            safe_df[column] = series.astype(object).map(
                lambda value: None if _is_missing_streamlit_value(value) else str(value)
            )
            debug_info["forced_str_columns"].append(column)
            continue

        sample_values = []
        for raw_value in series.tolist():
            if _is_missing_streamlit_value(raw_value):
                continue
            if isinstance(raw_value, np.generic):
                raw_value = raw_value.item()
            sample_values.append(raw_value)
            if len(sample_values) >= 50:
                break

        complex_found = any(isinstance(value, (list, dict, tuple, set)) for value in sample_values)
        mixed_types = {type(value).__name__ for value in sample_values}
        if complex_found:
            debug_info["complex_object_columns"].append(column)
        if len(mixed_types) > 1:
            debug_info["mixed_object_columns"].append(column)

        if complex_found or len(mixed_types) > 1 or pd.api.types.is_object_dtype(dtype):
            safe_df[column] = series.astype(object).map(_stringify_streamlit_cell)
            debug_info["forced_str_columns"].append(column)
            continue

        safe_df[column] = series.map(
            lambda value: None if _is_missing_streamlit_value(value) else value.item() if isinstance(value, np.generic) else value
        )

    if "Variant" in safe_df.columns:
        safe_df["Variant"] = safe_df["Variant"].astype(object).map(_stringify_streamlit_cell)
        debug_info["forced_str_columns"].append("Variant")
        debug_info["variant_sample_values"] = safe_df["Variant"].head(5).tolist()
    else:
        debug_info["variant_sample_values"] = []

    safe_df = make_arrow_safe(safe_df)
    debug_info["forced_str_columns"] = sorted(set(debug_info["forced_str_columns"]))
    debug_info["extension_type_columns"] = sorted(set(debug_info["extension_type_columns"]))
    debug_info["mixed_object_columns"] = sorted(set(debug_info["mixed_object_columns"]))
    debug_info["complex_object_columns"] = sorted(set(debug_info["complex_object_columns"]))
    debug_info["sanitized_dtypes"] = {str(col): str(dtype) for col, dtype in safe_df.dtypes.items()}
    return safe_df, debug_info


def _render_dataframe_debug_info(title: str, debug_info: dict):
    st.caption(f"{title} 原始 dtypes: {json.dumps(debug_info['original_dtypes'], ensure_ascii=False)}")
    st.caption(f"{title} 清洗后 dtypes: {json.dumps(debug_info['sanitized_dtypes'], ensure_ascii=False)}")
    st.caption(f"{title} 强制转 str 的列: {debug_info['forced_str_columns'] or '[]'}")
    st.caption(f"{title} pandas 扩展类型列: {debug_info['extension_type_columns'] or '[]'}")
    st.caption(f"{title} 混合 object 列: {debug_info['mixed_object_columns'] or '[]'}")
    st.caption(f"{title} list/dict/tuple/set 列: {debug_info['complex_object_columns'] or '[]'}")
    st.caption(f"{title} Variant 样本前5个: {debug_info['variant_sample_values'] or '[]'}")


def _can_render_with_pyarrow(df: pd.DataFrame):
    try:
        import pyarrow as pa
        pa.Table.from_pandas(df, preserve_index=False)
        return True, None
    except Exception as exc:
        return False, exc


def _build_manual_arrow_table(df: pd.DataFrame):
    try:
        import pyarrow as pa
        arrays = {}
        for column in df.columns:
            arrays[str(column)] = pa.array(df[column].tolist())
        return pa.Table.from_pydict(arrays), None
    except Exception as exc:
        return None, exc


def render_safe_streamlit_table(
    df,
    *,
    debug: bool = False,
    debug_title: str = "table",
    use_container_width: bool = True,
    hide_index: bool = True,
):
    safe_df, debug_info = sanitize_dataframe_for_streamlit(df)
    if debug:
        with st.expander(f"Debug: {debug_title} dataframe sanitize", expanded=False):
            _render_dataframe_debug_info(debug_title, debug_info)

    if safe_df.empty:
        st.info("No data available")
        return safe_df, debug_info

    arrow_ok, arrow_exc = _can_render_with_pyarrow(safe_df)
    if not arrow_ok:
        st.info(f"{debug_title} 已自动切换为兼容 HTML 视图，表格样式保持与论文表一致。")
        fallback_df = safe_df if hide_index else safe_df.reset_index()
        st.markdown(
            _build_html_table(
                fallback_df,
                visible_columns=list(fallback_df.columns),
                metric_specs={},
                row_type_fn=lambda _row: "",
            ),
            unsafe_allow_html=True,
        )
        if debug:
            with st.expander(f"Debug: {debug_title} fallback HTML", expanded=False):
                _render_dataframe_debug_info(debug_title, debug_info)
                st.caption(f"from_pandas={arrow_exc}")
        return safe_df, debug_info
    try:
        st.dataframe(safe_df, use_container_width=use_container_width, hide_index=hide_index)
    except Exception as exc:
        st.info(f"{debug_title} 已自动切换为兼容 HTML 视图，表格样式保持与论文表一致。")
        fallback_df = safe_df if hide_index else safe_df.reset_index()
        st.markdown(
            _build_html_table(
                fallback_df,
                visible_columns=list(fallback_df.columns),
                metric_specs={},
                row_type_fn=lambda _row: "",
            ),
            unsafe_allow_html=True,
        )
        if debug:
            with st.expander(f"Debug: {debug_title} fallback HTML", expanded=False):
                _render_dataframe_debug_info(debug_title, debug_info)
                st.caption(str(exc))
    return safe_df, debug_info


def _build_html_table(
    df: pd.DataFrame,
    visible_columns: list[str],
    metric_specs: dict,
    row_type_fn,
    delta_reference: dict[str, float] | None = None,
) -> str:
    best_sets = {}
    second_sets = {}
    for column, spec in metric_specs.items():
        if column in df.columns:
            best_sets[column], second_sets[column] = _metric_highlights(df[column], spec["direction"])

    html_rows = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        row_kind = row_type_fn(row)
        row_classes = ["paper-row"]
        if row_kind:
            row_classes.append(row_kind)
        cell_html = []
        for column in visible_columns:
            value = row.get(column)
            display = html.escape(str(value if not pd.isna(value) else "n/a"))
            cell_classes = []
            if column in metric_specs:
                precision = metric_specs[column]["precision"]
                display = _format_numeric(value, precision)
                delta_html = ""
                if delta_reference and column in delta_reference and pd.notna(value):
                    delta_html = _format_delta(float(value) - float(delta_reference[column]), precision=2)
                display = f"{html.escape(display)}{delta_html}"
                if row_idx in best_sets.get(column, set()):
                    cell_classes.append("metric-best")
                elif row_idx in second_sets.get(column, set()):
                    cell_classes.append("metric-second")
            elif isinstance(value, str) and value.startswith("<span class="):
                display = value
            elif _is_rank_like_column(column) and not _is_missing_streamlit_value(value):
                display = html.escape(str(int(float(value))))
            elif isinstance(value, (int, np.integer)) and not pd.isna(value):
                display = html.escape(str(int(value)))
            elif isinstance(value, (float, np.floating)) and not pd.isna(value):
                if str(column).startswith("Δ "):
                    display = html.escape(f"{float(value):+.2f}")
                else:
                    display = html.escape(f"{float(value):.2f}")
            cell_class_attr = f' class="{" ".join(cell_classes)}"' if cell_classes else ""
            cell_html.append(f"<td{cell_class_attr}>{display}</td>")
        html_rows.append(f'<tr class="{" ".join(row_classes)}">{"".join(cell_html)}</tr>')

    header_html = "".join(f"<th>{html.escape(_column_label(col))}</th>" for col in visible_columns)
    return f"""
    <div class="paper-table-wrap">
    <table class="paper-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>
    {''.join(html_rows)}
    </tbody>
    </table>
    </div>
    """


@st.cache_data(show_spinner=False)
def _cached_frozen_bundle(bundle_key: str, cache_signature: tuple):
    return load_frozen_table_bundle(bundle_key)


def _load_cached_frozen_bundle(bundle_key: str):
    return _cached_frozen_bundle(bundle_key, frozen_table_cache_signature(bundle_key))


@st.cache_data(show_spinner=False)
def _load_truncation_summary(summary_path: str):
    if not os.path.exists(summary_path):
        return pd.DataFrame()
    df = pd.read_csv(summary_path)
    if df.empty:
        return df
    df["model_display"] = df["model"].map(lambda value: canonical_method_name(str(value).strip()))
    variant_label_map = {
        "short": "Short",
        "medium": "Medium",
        "long": "Long",
        "noisy_long": "Noisy Long",
    }
    df["variant_display"] = df["variant"].map(lambda value: variant_label_map.get(str(value).strip(), str(value)))
    for column in [
        "complexity_level",
        "avg_prompt_tokens",
        "avg_used_tokens",
        "truncation_rate",
        "parse_failure_rate",
        "avg_parsed_edge_ratio",
        "avg_signal_ratio",
        "avg_inference_time_s",
        "avg_anchor_count",
        "avg_parse_confidence",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["model_display", "complexity_level"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_strict_protocol_rows(summary_path_items: tuple[tuple[str, str], ...]):
    records = []
    for variant_key, csv_path in summary_path_items:
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        row_specs = [
            ("overall", ("scope", "overall"), None),
            ("simple_local", ("scope", "val_normal:scene_type"), ("scope_name", "simple_local")),
            ("anti_truncation", ("scope", "split"), ("scope_name", "val_anti_truncation")),
        ]
        for slice_key, primary_cond, secondary_cond in row_specs:
            mask = df[primary_cond[0]].astype(str) == str(primary_cond[1])
            if secondary_cond is not None:
                mask &= df[secondary_cond[0]].astype(str) == str(secondary_cond[1])
            if not mask.any():
                continue
            row = df.loc[mask].iloc[0].to_dict()
            parsed = {
                "variant_key": variant_key,
                "variant_display": STRICT_PROTOCOL_DISPLAY_NAMES.get(variant_key, variant_key),
                "slice_key": slice_key,
                "slice_display": STRICT_PROTOCOL_SLICE_LABELS.get(slice_key, slice_key),
                "source_path": csv_path,
            }
            for key, value in row.items():
                parsed[key] = value
            for numeric_key in [
                "count",
                "target_recall",
                "gt_changed_edge_recall",
                "parse_fail_rate_strict",
                "no_anchor_rate",
                "no_anchor_when_gt_changed_rate",
                "anchor_count",
                "parse_confidence",
                "inference_time",
            ]:
                parsed[numeric_key] = _optional_float(row.get(numeric_key))
            records.append(parsed)
    return pd.DataFrame(records)


def _render_truncation_resilience_panel():
    truncation_df = _load_truncation_summary(TRUNCATION_SUMMARY_PATH)
    st.markdown('<div class="sec-hdr">🛡️ 主线为何不怕长文本噪声</div>', unsafe_allow_html=True)
    st.caption(
        "结论先行：Sparse-LoRA-v2 的 prompt 长度保持稳定，因此在 noisy-long 描述下不出现 truncation / parse failure。"
    )
    if truncation_df.empty:
        st.info("未找到抗截断实验汇总文件；已跳过该区块。")
        return

    model_order = [MAINLINE_METHOD_NAME, "Qwen-LoRA", "R1-LoRA"]
    truncation_df = truncation_df.copy()
    truncation_df["model_display"] = pd.Categorical(
        truncation_df["model_display"],
        categories=model_order,
        ordered=True,
    )
    truncation_df["variant_display"] = pd.Categorical(
        truncation_df["variant_display"],
        categories=["Short", "Medium", "Long", "Noisy Long"],
        ordered=True,
    )
    truncation_df = truncation_df.sort_values(["model_display", "variant_display"]).reset_index(drop=True)

    truncation_mtime = datetime.fromtimestamp(
        os.path.getmtime(TRUNCATION_SUMMARY_PATH),
        UTC,
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"Source: `{TRUNCATION_SUMMARY_PATH}` · modified={truncation_mtime}")

    summary_cols = st.columns(3)
    sparse_noisy = truncation_df[
        (truncation_df["model_display"] == MAINLINE_METHOD_NAME)
        & (truncation_df["variant"] == "noisy_long")
    ]
    qwen_noisy = truncation_df[
        (truncation_df["model_display"] == "Qwen-LoRA")
        & (truncation_df["variant"] == "noisy_long")
    ]
    r1_noisy = truncation_df[
        (truncation_df["model_display"] == "R1-LoRA")
        & (truncation_df["variant"] == "noisy_long")
    ]
    sparse_noisy_row = sparse_noisy.iloc[0] if not sparse_noisy.empty else None
    qwen_noisy_row = qwen_noisy.iloc[0] if not qwen_noisy.empty else None
    r1_noisy_row = r1_noisy.iloc[0] if not r1_noisy.empty else None

    summary_cols[0].metric(
        "Sparse-LoRA-v2: Noisy-Long Truncation",
        (
            f"{_format_numeric(sparse_noisy_row['truncation_rate'], 1)}%"
            if sparse_noisy_row is not None else "n/a"
        ),
        "bounded sparse prompt",
    )
    summary_cols[1].metric(
        "Qwen-LoRA: Noisy-Long Parse Fail",
        (
            f"{_format_numeric(qwen_noisy_row['parse_failure_rate'], 1)}%"
            if qwen_noisy_row is not None else "n/a"
        ),
        "dense full-edge output breaks first",
    )
    summary_cols[2].metric(
        "R1-LoRA: Noisy-Long Parse Fail",
        (
            f"{_format_numeric(r1_noisy_row['parse_failure_rate'], 1)}%"
            if r1_noisy_row is not None else "n/a"
        ),
        "same failure pattern on long noisy inputs",
    )

    fig_left, fig_right = st.columns([1, 1], gap="medium")
    with fig_left:
        trunc_fig = px.line(
            truncation_df,
            x="variant_display",
            y="truncation_rate",
            color="model_display",
            markers=True,
            category_orders={
                "variant_display": ["Short", "Medium", "Long", "Noisy Long"],
                "model_display": model_order,
            },
            color_discrete_map={
                MAINLINE_METHOD_NAME: METHOD_COLORS[MAINLINE_METHOD_NAME],
                "Qwen-LoRA": METHOD_COLORS["Qwen-LoRA"],
                "R1-LoRA": METHOD_COLORS["R1-LoRA"],
            },
            template="plotly_white",
            title="Truncation Rate: sparse mainline stays flat",
        )
        trunc_fig.update_layout(height=340, xaxis_title="", yaxis_title="Truncation Rate (%)", legend_title_text="")
        st.plotly_chart(trunc_fig, use_container_width=True)
    with fig_right:
        parse_fig = px.line(
            truncation_df,
            x="variant_display",
            y="parse_failure_rate",
            color="model_display",
            markers=True,
            category_orders={
                "variant_display": ["Short", "Medium", "Long", "Noisy Long"],
                "model_display": model_order,
            },
            color_discrete_map={
                MAINLINE_METHOD_NAME: METHOD_COLORS[MAINLINE_METHOD_NAME],
                "Qwen-LoRA": METHOD_COLORS["Qwen-LoRA"],
                "R1-LoRA": METHOD_COLORS["R1-LoRA"],
            },
            template="plotly_white",
            title="Parse Failure: dense baselines collapse on noisy-long",
        )
        parse_fig.update_layout(height=340, xaxis_title="", yaxis_title="Parse Failure Rate (%)", legend_title_text="")
        st.plotly_chart(parse_fig, use_container_width=True)

    token_fig = px.bar(
        truncation_df,
        x="variant_display",
        y="avg_used_tokens",
        color="model_display",
        barmode="group",
        category_orders={
            "variant_display": ["Short", "Medium", "Long", "Noisy Long"],
            "model_display": model_order,
        },
        color_discrete_map={
            MAINLINE_METHOD_NAME: METHOD_COLORS[MAINLINE_METHOD_NAME],
            "Qwen-LoRA": METHOD_COLORS["Qwen-LoRA"],
            "R1-LoRA": METHOD_COLORS["R1-LoRA"],
        },
        template="plotly_white",
        title="Prompt Budget: sparse stays bounded while dense prompts explode",
    )
    token_fig.update_layout(height=320, xaxis_title="", yaxis_title="Used Tokens", legend_title_text="")
    st.plotly_chart(token_fig, use_container_width=True)

    visible_columns = [
        column for column in [
            "model_display",
            "variant_display",
            "avg_prompt_tokens",
            "avg_used_tokens",
            "truncation_rate",
            "parse_failure_rate",
            "avg_inference_time_s",
            "avg_anchor_count",
        ]
        if column in truncation_df.columns
    ]
    with st.expander("查看抗截断汇总表", expanded=False):
        render_safe_streamlit_table(
            truncation_df[visible_columns].rename(
                columns={
                    "model_display": "Model",
                    "variant_display": "Variant",
                    "avg_prompt_tokens": "Prompt Tokens",
                    "avg_used_tokens": "Used Tokens",
                    "truncation_rate": "Truncation Rate (%)",
                    "parse_failure_rate": "Parse Failure Rate (%)",
                    "avg_inference_time_s": "Inference Time (s)",
                    "avg_anchor_count": "Anchor Count",
                }
            ),
            debug=False,
            debug_title="appendix_truncation_summary",
            use_container_width=True,
            hide_index=True,
        )


def _render_strict_protocol_panel():
    strict_df = _load_strict_protocol_rows(tuple(STRICT_PROTOCOL_SUMMARY_PATHS.items()))
    st.markdown('<div class="sec-hdr">📏 冻结主线 vs Patch 分支（严格协议）</div>', unsafe_allow_html=True)
    st.caption(
        "结论先行：`stage4_fix` 仍是最安全的默认主线；patch 分支只有局部轻微波动，没有形成可替代的稳定升级。"
    )
    if strict_df.empty:
        st.info("未找到 strict protocol 汇总文件；已跳过该区块。")
        return

    variant_order = ["stage4_fix", "stage4c_fix", "stage4d_micro"]
    slice_order = ["Overall", "anti_truncation", "simple_local"]
    strict_df = strict_df.copy()
    strict_df["variant_display"] = pd.Categorical(
        strict_df["variant_display"],
        categories=[STRICT_PROTOCOL_DISPLAY_NAMES[key] for key in variant_order],
        ordered=True,
    )
    strict_df["slice_display"] = pd.Categorical(
        strict_df["slice_display"],
        categories=slice_order,
        ordered=True,
    )
    strict_df = strict_df.sort_values(["slice_display", "variant_display"]).reset_index(drop=True)

    latest_mtime = max(
        os.path.getmtime(path)
        for path in STRICT_PROTOCOL_SUMMARY_PATHS.values()
        if os.path.exists(path)
    )
    strict_mtime = datetime.fromtimestamp(latest_mtime, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"Sources: {len(strict_df['source_path'].unique())} files · latest_modified={strict_mtime}")

    overall_mainline = strict_df[
        (strict_df["variant_key"] == "stage4_fix") & (strict_df["slice_key"] == "overall")
    ]
    simple_mainline = strict_df[
        (strict_df["variant_key"] == "stage4_fix") & (strict_df["slice_key"] == "simple_local")
    ]
    anti_mainline = strict_df[
        (strict_df["variant_key"] == "stage4_fix") & (strict_df["slice_key"] == "anti_truncation")
    ]
    overall_row = overall_mainline.iloc[0] if not overall_mainline.empty else None
    simple_row = simple_mainline.iloc[0] if not simple_mainline.empty else None
    anti_row = anti_mainline.iloc[0] if not anti_mainline.empty else None

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Frozen Mainline Overall Recall",
        f"{_format_numeric(overall_row['target_recall'], 1)}%" if overall_row is not None else "n/a",
        "strict overall",
    )
    metric_cols[1].metric(
        "anti_truncation Recall",
        f"{_format_numeric(anti_row['gt_changed_edge_recall'], 1)}%" if anti_row is not None else "n/a",
        "robustness slice stays strong",
    )
    metric_cols[2].metric(
        "simple_local Strict Parse Fail",
        f"{_format_numeric(simple_row['parse_fail_rate_strict'], 3)}" if simple_row is not None else "n/a",
        "remaining hard slice",
    )

    strict_compare_df = strict_df.copy()
    strict_compare_df["strict_success_rate"] = 100.0 - strict_compare_df["parse_fail_rate_strict"].fillna(0.0) * 100.0

    chart_left, chart_right = st.columns([1, 1], gap="medium")
    with chart_left:
        recall_fig = px.bar(
            strict_compare_df,
            x="slice_display",
            y="gt_changed_edge_recall",
            color="variant_display",
            barmode="group",
            category_orders={
                "slice_display": slice_order,
                "variant_display": [STRICT_PROTOCOL_DISPLAY_NAMES[key] for key in variant_order],
            },
            color_discrete_map={
                STRICT_PROTOCOL_DISPLAY_NAMES["stage4_fix"]: METHOD_COLORS[MAINLINE_METHOD_NAME],
                STRICT_PROTOCOL_DISPLAY_NAMES["stage4c_fix"]: "#2563EB",
                STRICT_PROTOCOL_DISPLAY_NAMES["stage4d_micro"]: "#94A3B8",
            },
            template="plotly_white",
            title="Changed-Edge Recall: stage4_fix remains the safest default",
        )
        recall_fig.update_layout(height=340, xaxis_title="", yaxis_title="Recall (%)", legend_title_text="")
        st.plotly_chart(recall_fig, use_container_width=True)
    with chart_right:
        fail_fig = px.bar(
            strict_compare_df,
            x="slice_display",
            y="parse_fail_rate_strict",
            color="variant_display",
            barmode="group",
            category_orders={
                "slice_display": slice_order,
                "variant_display": [STRICT_PROTOCOL_DISPLAY_NAMES[key] for key in variant_order],
            },
            color_discrete_map={
                STRICT_PROTOCOL_DISPLAY_NAMES["stage4_fix"]: METHOD_COLORS[MAINLINE_METHOD_NAME],
                STRICT_PROTOCOL_DISPLAY_NAMES["stage4c_fix"]: "#2563EB",
                STRICT_PROTOCOL_DISPLAY_NAMES["stage4d_micro"]: "#94A3B8",
            },
            template="plotly_white",
            title="Strict Parse Failure: bottleneck stays in simple_local, not anti_truncation",
        )
        fail_fig.update_layout(height=340, xaxis_title="", yaxis_title="Parse Fail Rate", legend_title_text="")
        st.plotly_chart(fail_fig, use_container_width=True)

    protocol_table = strict_compare_df[
        [
            "variant_display",
            "slice_display",
            "target_recall",
            "gt_changed_edge_recall",
            "parse_fail_rate_strict",
            "no_anchor_rate",
            "anchor_count",
            "parse_confidence",
            "inference_time",
        ]
    ].rename(
        columns={
            "variant_display": "Variant",
            "slice_display": "Slice",
            "target_recall": "Target Recall (%)",
            "gt_changed_edge_recall": "Changed-Edge Recall (%)",
            "parse_fail_rate_strict": "Strict Parse Fail",
            "no_anchor_rate": "No Anchor Rate",
            "anchor_count": "Anchor Count",
            "parse_confidence": "Parse Confidence",
            "inference_time": "Inference Time (s)",
        }
    )
    with st.expander("查看 strict protocol 汇总表", expanded=False):
        render_safe_streamlit_table(
            protocol_table,
            debug=False,
            debug_title="appendix_strict_protocol_summary",
            use_container_width=True,
            hide_index=True,
        )


def _render_method_comparison_panel(debug: bool = False):
    bundle = _load_cached_frozen_bundle("method_comparison")
    df = bundle.dataframe.copy()
    df["__paper_order"] = range(len(df))
    df["Method Display"] = df["Method"].apply(canonical_method_name)
    df["Travel Time rank"] = _rank_metric(df["Travel Time (s)"], "min")
    df["Compliance rank"] = _rank_metric(df["Constraint Rate (%)"], "max")

    mainline_mask = df["Method Display"] == MAINLINE_METHOD_NAME
    if mainline_mask.any():
        mainline_row = df.loc[mainline_mask].iloc[0]
        delta_reference = {
            column: float(mainline_row[column])
            for column in FROZEN_METHOD_METRIC_SPECS
            if column in df.columns and pd.notna(mainline_row[column])
        }
    else:
        delta_reference = {}

    best_tt = df.loc[pd.to_numeric(df["Travel Time (s)"], errors="coerce").idxmin()]
    best_comp = df.loc[pd.to_numeric(df["Constraint Rate (%)"], errors="coerce").idxmax()]
    main_summary = df.loc[mainline_mask].iloc[0] if mainline_mask.any() else None

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Best Travel Time",
        best_tt["Method Display"],
        f'{_format_numeric(best_tt["Travel Time (s)"], 2)} s',
    )
    metric_cols[1].metric(
        "Best Compliance",
        best_comp["Method Display"],
        f'{_format_numeric(best_comp["Constraint Rate (%)"], 2)} %',
    )
    if main_summary is not None:
        metric_cols[2].metric(
            "Sparse-LoRA-v2 Summary",
            f'{_format_numeric(main_summary["Travel Time (s)"], 2)} s',
            f'TT rank #{int(main_summary["Travel Time rank"])} | Compliance rank #{int(main_summary["Compliance rank"])}',
        )
    else:
        metric_cols[2].metric("Sparse-LoRA-v2 Summary", "n/a", "主方法未在冻结表中找到")

    st.caption("本区直接绑定 final frozen 主表；最优 / 次优值与 Sparse-LoRA-v2 高亮均以冻结结果为准。")
    _render_frozen_artifact_meta(bundle, "Main table = final frozen results", "method_frozen", debug=debug)
    _render_group_strip([
        ("Efficiency", "Travel Time, Planning Time"),
        ("Compliance", "Constraint Rate"),
        ("Signal Coverage", "Signal Ratio"),
        ("Runtime Detail", "model_infer_time_s, route_solve_time_s"),
    ])

    # 主表固定展示，GAT 等探索项放到附录。
    _MAIN_CATEGORIES = {"Classical", "GNN", "RL", "LLM-Prompt", "Ours"}
    filtered_df = df.copy()
    if "Category" in filtered_df.columns:
        _extended_df = filtered_df[~filtered_df["Category"].isin(_MAIN_CATEGORIES)].copy()
        filtered_df = filtered_df[filtered_df["Category"].isin(_MAIN_CATEGORIES)].reset_index(drop=True)
    else:
        _extended_df = pd.DataFrame()
    filtered_df = filtered_df.sort_values("__paper_order", ascending=True, na_position="last").reset_index(drop=True)

    # 投屏和截图用的固定列顺序。
    visible_columns = [c for c in [
        "Method Display", "Method (CN)", "Category",
        "Travel Time (s)", "Planning Time (s)", "Constraint Rate (%)", "Signal Ratio (%)",
    ] if c in filtered_df.columns or c == "Method Display"]
    show_delta = False

    if not _extended_df.empty:
        _ext_names = ", ".join(_extended_df["Method Display"].tolist()) if "Method Display" in _extended_df.columns else ""
        st.caption(
            f"**Extended / exploratory 方法已移至附录区（不参与主表排名）**：{_ext_names}。"
            " 详见 Appendix / Exploratory tab。"
        )

    def _row_type(row):
        if row.get("Method Display") == MAINLINE_METHOD_NAME:
            return "mainline-row"
        if str(row.get("Category", "")).lower() == "extended":
            return "extended-row"
        return ""

    delta_chart_df = filtered_df[["Method Display", "Travel Time (s)"]].copy()
    if mainline_mask.any():
        main_tt = float(mainline_row["Travel Time (s)"])
        delta_chart_df["Δ vs Sparse-LoRA-v2 (s)"] = pd.to_numeric(delta_chart_df["Travel Time (s)"], errors="coerce") - main_tt
    else:
        delta_chart_df["Δ vs Sparse-LoRA-v2 (s)"] = np.nan

    table_html = _build_html_table(
        filtered_df,
        visible_columns=visible_columns,
        metric_specs={column: spec for column, spec in FROZEN_METHOD_METRIC_SPECS.items() if column in visible_columns},
        row_type_fn=_row_type,
        delta_reference=None,
    )
    # 主表占满宽度，答辩时先看这一张。
    st.markdown(table_html, unsafe_allow_html=True)

    # 图表默认收起，需要细看时再展开。
    with st.expander("图表详情（点击展开）", expanded=False):
        chart_left, chart_right, chart_delta = st.columns([1, 1, 1], gap="medium")
        with chart_left:
            if not filtered_df.empty:
                tt_fig = px.bar(
                    filtered_df,
                    x="Method Display",
                    y="Travel Time (s)",
                    color="Method Display",
                    template="plotly_white",
                    title="Travel Time",
                )
                tt_fig.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="Travel Time (s)")
                st.plotly_chart(tt_fig, use_container_width=True)
        with chart_right:
            if not filtered_df.empty:
                compliance_fig = px.bar(
                    filtered_df,
                    x="Method Display",
                    y="Constraint Rate (%)",
                    color="Method Display",
                    template="plotly_white",
                    title="Constraint Rate",
                )
                compliance_fig.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="Constraint Rate (%)")
                st.plotly_chart(compliance_fig, use_container_width=True)
        with chart_delta:
            if not delta_chart_df.empty:
                delta_fig = px.bar(
                    delta_chart_df,
                    x="Method Display",
                    y="Δ vs Sparse-LoRA-v2 (s)",
                    color="Δ vs Sparse-LoRA-v2 (s)",
                    color_continuous_scale="RdYlGn_r",
                    template="plotly_white",
                    title="Δ Travel Time vs Sparse-LoRA-v2",
                )
                delta_fig.update_layout(height=320, xaxis_title="", yaxis_title="Delta Travel Time (s)")
                st.plotly_chart(delta_fig, use_container_width=True)


def _render_ablation_panel(debug: bool = False):
    bundle = _load_cached_frozen_bundle("ablation_study")
    raw_df = bundle.dataframe.copy()
    summary_df = build_ablation_summary(raw_df)
    summary_df["__paper_order"] = range(len(summary_df))
    summary_df["Variant Display"] = summary_df["Variant"].map(lambda x: ABLATED_VARIANT_DISPLAY.get(x, str(x)))
    summary_df["Structured Input"] = summary_df["Variant"].map(
        lambda x: "No" if str(x) == "No-SpatioTemporal" else "Yes"
    )

    frozen_mainline_mask = summary_df["Variant"] == "Sparse-LoRA"
    frozen_mainline_row = summary_df.loc[frozen_mainline_mask].iloc[0] if frozen_mainline_mask.any() else None

    summary_df["Δ vs Prev TT"] = summary_df["Travel Time (s)"] - summary_df["Travel Time (s)"].shift(1)
    if frozen_mainline_row is not None:
        summary_df["Δ vs Full TT"] = summary_df["Travel Time (s)"] - float(frozen_mainline_row["Travel Time (s)"])
        summary_df["Δ vs Full Compliance"] = summary_df["Constraint Rate (%)"] - float(frozen_mainline_row["Constraint Rate (%)"])
    else:
        summary_df["Δ vs Full TT"] = np.nan
        summary_df["Δ vs Full Compliance"] = np.nan

    def _ablation_badges(row):
        badges = []
        if pd.notna(row.get("Δ vs Prev TT")) and float(row["Δ vs Prev TT"]) <= -5:
            badges.append(_build_badge_html("Travel Time gain", "positive"))
        if pd.notna(row.get("Δ vs Full Compliance")) and float(row["Δ vs Full Compliance"]) >= -1:
            badges.append(_build_badge_html("CR near mainline", "main"))
        if frozen_mainline_row is not None and pd.notna(row.get("Planning Time (s)")):
            plan_gap = abs(float(row["Planning Time (s)"]) - float(frozen_mainline_row["Planning Time (s)"]))
            if plan_gap <= 0.05:
                badges.append(_build_badge_html("runtime overhead small", "warning"))
        return " ".join(badges) if badges else _build_badge_html("numeric summary only", "neutral")

    summary_df["Ablation badges"] = summary_df.apply(_ablation_badges, axis=1)

    best_tt = summary_df.loc[pd.to_numeric(summary_df["Travel Time (s)"], errors="coerce").idxmin()]
    best_comp = summary_df.loc[pd.to_numeric(summary_df["Constraint Rate (%)"], errors="coerce").idxmax()]

    metric_cols = st.columns(3)
    if frozen_mainline_row is not None:
        metric_cols[0].metric(
            "Frozen Mainline Method",
            f"{MAINLINE_METHOD_NAME} / stage4_fix",
            "Mainline method fixed for frozen benchmark",
        )
        metric_cols[1].metric(
            "Frozen Mainline TT",
            f'{_format_numeric(frozen_mainline_row["Travel Time (s)"], 2)} s',
            "Travel Time on frozen mainline row",
        )
        metric_cols[2].metric(
            "Frozen Mainline CR",
            f'{_format_numeric(frozen_mainline_row["Constraint Rate (%)"], 2)}%',
            "Constraint Rate on frozen mainline row",
        )
    else:
        metric_cols[0].metric("Frozen Mainline Method", f"{MAINLINE_METHOD_NAME} / stage4_fix", "mainline row missing")
        metric_cols[1].metric("Frozen Mainline TT", "n/a", "未在冻结消融表中找到 Sparse-LoRA")
        metric_cols[2].metric("Frozen Mainline CR", "n/a", "未在冻结消融表中找到 Sparse-LoRA")

    st.caption(
        "Mainline method = Sparse-LoRA-v2 / stage4_fix. "
        "GAT variants are exploratory extensions and are not treated as the frozen mainline. "
        "Ablation page is for component comparison, not for redefining the main method."
    )
    st.caption("`Structured Input` 来自 `No-SpatioTemporal` 这一官方 variant 名称定义；其他模块状态直接取自 CSV 原列。")
    _render_frozen_artifact_meta(bundle, "Ablation table = final frozen ablation", "ablation_frozen", debug=debug)
    _render_group_strip([
        ("Module Switches", "Structured Input, LLM Component, GAT Component, Sparse Mode"),
        ("Performance", "Travel Time, Constraint Rate, Signal Ratio, Parsed Edge Ratio"),
        ("Delta View", "vs previous row, vs frozen mainline"),
    ])

    # 消融表也固定成论文顺序和列集合。
    raw_scene_df = raw_df.copy()
    visible_columns = [c for c in [
        "Variant Display", "Variant (CN)", "Structured Input", "GAT Component", "Sparse Mode",
        "Travel Time (s)", "Planning Time (s)", "Constraint Rate (%)", "Signal Ratio (%)",
        "Δ vs Full TT",
    ] if c in summary_df.columns or c == "Variant Display"]
    summary_df = summary_df.sort_values("__paper_order", ascending=True, na_position="last").reset_index(drop=True)

    def _row_type(row):
        if row.get("Variant") == "Sparse-LoRA":
            return "fullmethod-row"
        if row.get("Variant") == "No-SpatioTemporal":
            return "ablation-focus-row"
        return ""

    table_html = _build_html_table(
        summary_df,
        visible_columns=visible_columns,
        metric_specs={column: spec for column, spec in FROZEN_ABLATION_METRIC_SPECS.items() if column in visible_columns},
        row_type_fn=_row_type,
        delta_reference=None,
    )

    # 消融表占满宽度。
    st.markdown(table_html, unsafe_allow_html=True)

    # 明细图和原始数据默认收起。
    with st.expander("图表详情（点击展开）", expanded=False):
        expl_cols = st.columns(2)
        expl_cols[0].metric(
            "Lowest Travel Time Among Ablation Variants",
            best_tt["Variant Display"],
            f'{_format_numeric(best_tt["Travel Time (s)"], 2)} s',
        )
        expl_cols[1].metric(
            "Highest Constraint Rate Among Ablation Variants",
            best_comp["Variant Display"],
            f'{_format_numeric(best_comp["Constraint Rate (%)"], 2)} %',
        )
        st.caption("GAT variants are exploratory and remain in the detailed ablation charts, but they are not treated as the frozen mainline.")
        chart_left, chart_right = st.columns([1, 1], gap="medium")
        with chart_left:
            aux_df = summary_df[["Variant Display", "Travel Time (s)"]].copy()
            aux_fig = px.bar(
                aux_df,
                x="Variant Display",
                y="Travel Time (s)",
                color="Variant Display",
                text="Travel Time (s)",
                template="plotly_white",
                title="Ablation: Travel Time",
            )
            aux_fig.update_layout(showlegend=False, height=360, xaxis_title="", yaxis_title="Travel Time (s)")
            aux_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(aux_fig, use_container_width=True)
        with chart_right:
            raw_scene_df["Variant Display"] = raw_scene_df["Variant"].map(lambda x: ABLATED_VARIANT_DISPLAY.get(x, str(x)))
            heatmap_source = raw_scene_df.pivot_table(
                index="Variant Display",
                columns="Scenario",
                values="Travel Time (s)",
                aggfunc="mean",
            )
            if not heatmap_source.empty:
                heatmap_fig = px.imshow(
                    heatmap_source,
                    text_auto=".1f",
                    aspect="auto",
                    color_continuous_scale="YlGnBu_r",
                    title="Per-Scenario Travel Time Heatmap",
                )
                heatmap_fig.update_layout(height=max(300, len(heatmap_source) * 44 + 120))
                st.plotly_chart(heatmap_fig, use_container_width=True)

        module_cols = [col for col in ["Structured Input", "LLM Component", "GAT Component", "Sparse Mode"] if col in summary_df.columns]
        if module_cols:
            module_matrix = summary_df[["Variant Display"] + module_cols].copy()
            for column in module_cols:
                module_matrix[column] = module_matrix[column].map(lambda value: 1 if str(value).strip().lower() in {"yes", "true", "1"} else 0)
            module_fig = px.imshow(
                module_matrix.set_index("Variant Display"),
                text_auto=True,
                aspect="auto",
                color_continuous_scale=[[0, "#E2E8F0"], [1, "#16A34A"]],
                title="Module Switch Matrix",
            )
            module_fig.update_layout(height=max(240, len(module_matrix) * 40 + 100))
            st.plotly_chart(module_fig, use_container_width=True)

        if raw_scene_df["Scenario"].nunique() > 1 and raw_scene_df.groupby("Variant").size().max() > 1:
            box_left, box_right = st.columns([1, 1], gap="medium")
            with box_left:
                box_fig = px.box(
                    raw_scene_df,
                    x="Variant Display",
                    y="Travel Time (s)",
                    color="Variant Display",
                    template="plotly_white",
                    title="Per-Scenario Travel Time Box",
                )
                box_fig.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="Travel Time (s)")
                st.plotly_chart(box_fig, use_container_width=True)
            with box_right:
                violin_fig = px.violin(
                    raw_scene_df,
                    x="Variant Display",
                    y="Constraint Rate (%)",
                    color="Variant Display",
                    box=True,
                    points="all",
                    template="plotly_white",
                    title="Per-Scenario Compliance Violin",
                )
                violin_fig.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="Constraint Rate (%)")
                st.plotly_chart(violin_fig, use_container_width=True)

    with st.expander("查看冻结 CSV 的 raw scenario rows", expanded=False):
        render_safe_streamlit_table(
            raw_scene_df,
            debug=debug,
            debug_title="ablation_raw_rows",
            use_container_width=True,
            hide_index=True,
        )


def _render_appendix_table():
    bundle = _load_cached_frozen_bundle("appendix")
    df = bundle.dataframe.copy()
    if "Method" in df.columns:
        df["Method Display"] = df["Method"].apply(canonical_method_name)
    _render_frozen_artifact_meta(bundle, "Appendix = exploratory only", "appendix_frozen")
    st.caption("Appendix / exploratory rows stay here and do not occupy the main-table position.")

    visible_columns = [column for column in [
        "Method Display", "Method (CN)", "Category", "Travel Time (s)", "Planning Time (s)",
        "Success Rate", "Average Reward", "Failure Rate", "Timeout Rate", "Protocol", "Source File", "Notes"
    ] if column in df.columns]

    def _row_type(_row):
        return "extended-row"

    table_html = _build_html_table(
        df.reset_index(drop=True),
        visible_columns=visible_columns,
        metric_specs={},
        row_type_fn=_row_type,
        delta_reference=None,
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _render_frozen_tables_workspace(total_cost_value: float, weight_dict_value: dict, planning_time_value: float, debug: bool = False):
    # 答辩用数据源与方法定位说明
    st.markdown(
        """
        <div style="background:#1E3A8A;border-radius:12px;padding:14px 18px;margin-bottom:12px;color:#fff">
        <div style="font-size:1.05rem;font-weight:800;margin-bottom:6px">
        实验结果 · 冻结口径说明
        </div>
        <div style="font-size:.88rem;line-height:1.6;opacity:.95">
        <b>主方法</b>：Sparse-LoRA-v2 &nbsp;|&nbsp; <b>上游基线</b>：stage4_fix &nbsp;|&nbsp;
        <b>评估协议</b>：场景 A–F × 6 seeds（101/111/202/303/404/505），live SUMO 闭环<br>
        <b>主表来源</b>：<code>results/final_tables/method_comparison_final.csv</code> &nbsp;（冻结于 2026-04-11）<br>
        <b>方法定位</b>：Classical / GNN / RL / LLM-Prompt = 对比基线 &nbsp;|&nbsp;
        Ours (Sparse-LoRA-v2) = 主方法 &nbsp;|&nbsp; Extended / GAT = exploratory，仅附录
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    refresh_cols = st.columns([1.2, 3.8])
    with refresh_cols[0]:
        if st.button("刷新冻结缓存", key="refresh_frozen_workspace_cache", use_container_width=True):
            _cached_frozen_bundle.clear()
            clear_viz_cache()
            st.success("已刷新 frozen table / map 缓存；当前页重新读取最新文件。")
    with refresh_cols[1]:
        st.caption("数据源绑定到精确的冻结 CSV 路径；如文件更新请点击左侧按钮刷新。")

    method_tab, ablation_tab, appendix_tab = st.tabs([
        "主表 · Method Comparison",
        "消融表 · Ablation Study",
        "附录 · Appendix / Exploratory",
    ])

    with method_tab:
        st.markdown(
            """
            <div class="info-strip">
            <b>主结果表</b>：Sparse-LoRA-v2（绿色高亮行）为主方法；Extended 类别已移出主表，见附录 tab。
            指标越优越好：Travel Time 越低越好，Constraint Rate 越高越好。
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            _render_method_comparison_panel(debug=debug)
        except Exception as method_exc:
            st.error(f"主表加载失败：{method_exc}")
            st.code(traceback.format_exc())

    with ablation_tab:
        st.markdown(
            """
            <div class="info-strip">
            <b>消融研究</b>：Mainline method = Sparse-LoRA-v2 / stage4_fix。该页用于展示时空增强输入 / GAT / Sparse 各模块的贡献。
            GAT variants are exploratory extensions and are not treated as the frozen mainline。
            Ablation page is for component comparison, not for redefining the main method。
            绿色行 = Frozen Mainline（Sparse-LoRA-v2），蓝色行 = No-SpatioTemporal（关键消融对照）。
            Δ vs Frozen Mainline TT 为各变体与冻结主线的行程时间差值。
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            _render_ablation_panel(debug=debug)
        except Exception as ablation_exc:
            st.error(f"消融表加载失败：{ablation_exc}")
            st.code(traceback.format_exc())

    with appendix_tab:
        st.markdown(
            """
            <div class="info-strip">
            <b>附录 / Exploratory only</b>：PPO / GAT 等探索性方法，不参与主表结论，不代表主链路声称。
            PPO 使用简化协议（非 live SUMO 六场景），不可与主表直接比较。
            GAT 变体行程时间与主方法完全相同，signal ratio 仅反映覆盖率，不影响主结论。
            </div>
            """,
            unsafe_allow_html=True,
        )
        try:
            _render_appendix_table()
        except Exception as appendix_exc:
            st.error(f"附录表加载失败：{appendix_exc}")
            st.code(traceback.format_exc())

        st.divider()
        st.markdown('<div class="sec-hdr">📊 Exploratory Visuals (Appendix Only)</div>', unsafe_allow_html=True)
        st.caption("下面的图只作为附录理解辅助，不替代表格。")

        compare_result_path = _latest_compare_results_path()
        constraint_rate = (
            sum(1 for w in weight_dict_value.values() if w > 5.0) / max(len(weight_dict_value), 1) * 100
            if weight_dict_value else 0.0
        )

        control_cols = st.columns([3, 1])
        with control_cols[1]:
            run_compare_btn = False
            if config.get("RUN_MODE", "demo_mode") == "experiment_mode":
                run_compare_btn = st.button(
                    "🔬 运行探索性对比实验",
                    use_container_width=True,
                    help="appendix only；用于更新 exploratory visuals，不影响 frozen main tables。",
                )
            else:
                st.caption("Demo Mode 下隐藏探索性实验运行入口")
        if run_compare_btn:
            with st.spinner("正在运行探索性对比实验，请稍候..."):
                try:
                    res = subprocess.run(
                        ["python", "/root/autodl-tmp/compare/run_compare.py", "--llm"],
                        capture_output=True,
                        text=True,
                        timeout=1200,
                        cwd="/root/autodl-tmp",
                    )
                    if res.returncode == 0:
                        st.success("✅ 探索性对比实验完成！")
                    else:
                        st.error(f"❌ 失败: {res.stderr[-300:]}")
                except Exception as ex:
                    st.error(f"❌ {ex}")

        if os.path.exists(compare_result_path):
            try:
                compare_mtime = datetime.fromtimestamp(
                    os.path.getmtime(compare_result_path),
                    UTC,
                ).strftime("%Y-%m-%d %H:%M:%S UTC")
                bar_fig, radar_fig, scatter_fig, cmp_df, data_src, box_fig, scene_bar_fig, heatmap_fig = viz_engine.plot_real_compare(
                    compare_result_path,
                    current_cost=float(total_cost_value),
                    current_constraint=float(constraint_rate),
                    current_time=float(planning_time_value),
                )
                st.caption(
                    f"Exploratory compare source: `{compare_result_path}`"
                    + ("（real）" if data_src in {"实验数据", "real"} else "（未确认 real）")
                    + f" · modified={compare_mtime}"
                )
                bar_fig.update_layout(height=380)
                st.plotly_chart(bar_fig, use_container_width=True)

                extra_left, extra_right = st.columns([1, 1])
                with extra_left:
                    if heatmap_fig is not None:
                        heatmap_fig.update_layout(height=360)
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    else:
                        st.info("当前 compare 结果未生成热力图。")
                with extra_right:
                    radar_fig.update_layout(height=360)
                    st.plotly_chart(radar_fig, use_container_width=True)

                with st.expander("查看 exploratory compare 表格", expanded=False):
                    if cmp_df is not None and len(cmp_df) > 0:
                        render_safe_streamlit_table(
                            cmp_df,
                            debug=debug,
                            debug_title="appendix_exploratory_compare",
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("对比结果文件存在，但当前未解析出表格。")
            except Exception as compare_exc:
                st.warning(f"exploratory compare 结果存在，但当前解析失败；已跳过该区块以保证附录页稳定展示。原因：{compare_exc}")
                if debug:
                    st.code(traceback.format_exc())
        else:
            st.info("未检测到 compare_results.json；附录页目前仅展示冻结 appendix 表。")

        st.divider()
        try:
            _render_strict_protocol_panel()
        except Exception as strict_exc:
            st.warning(f"严格评估协议对比加载失败；已自动跳过该区块。原因：{strict_exc}")
            if debug:
                st.code(traceback.format_exc())

        st.divider()
        try:
            _render_truncation_resilience_panel()
        except Exception as truncation_exc:
            st.warning(f"抗截断实验图表加载失败；已自动跳过该区块。原因：{truncation_exc}")
            if debug:
                st.code(traceback.format_exc())


def _estimate_route_time_on_weights(net, path, weight_dict) -> float:
    total = 0.0
    for edge_id in list(path or []):
        try:
            edge_obj = net.getEdge(edge_id)
        except Exception:
            edge_obj = None
        if edge_obj is None:
            continue
        total += edge_travel_time(edge_obj, _safe_float((weight_dict or {}).get(edge_id), 2.0))
    return float(total)


def _estimate_route_time_with_cost_fn(net, path, edge_cost_fn) -> float:
    total = 0.0
    for edge_id in list(path or []):
        try:
            edge_obj = net.getEdge(edge_id)
        except Exception:
            edge_obj = None
        if edge_obj is None:
            continue
        total += float(edge_cost_fn(edge_obj))
    return float(total)


def _format_edge_label(edge_id: str) -> str:
    edge_id = str(edge_id or "").strip()
    if not edge_id:
        return "—"
    return f"{shared_edge_road_name(edge_id)} · {edge_id}"


def _summarize_path(path, max_items: int = 7) -> str:
    items = [str(edge).strip() for edge in list(path or []) if str(edge).strip()]
    if not items:
        return "未生成路径"
    if len(items) <= max_items:
        return " → ".join(items)
    head = items[:3]
    tail = items[-2:]
    return " → ".join(head + ["…"] + tail)


def _route_debug_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _edge_overlap_ratio(left_path, right_path) -> float:
    left = list(left_path or [])
    right = list(right_path or [])
    if not left or not right:
        return 0.0
    shared = len(set(left) & set(right))
    return shared / max(len(left), len(right), 1)


RUNTIME_AUDIT_DIR = "/root/autodl-tmp/results/runtime_audit"


def _runtime_audit_safe_number(value):
    parsed = _optional_float(value)
    return parsed if parsed is not None else float("nan")


def _runtime_audit_rows_from_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    planned_sumo = dict(results.get("sumo_result") or {})
    baseline_sumo = dict(results.get("baseline_sumo_result") or {})
    scene_weight_layers = dict(results.get("scene_weight_layers") or {})
    profile_config = dict(results.get("scene_profile_config") or {})
    incident_edges = set(dict(profile_config.get("incident_edges") or {}).keys())
    blocked_edges = set(profile_config.get("blocked_edges") or ())
    lane_reduction_edges = set(dict(profile_config.get("lane_reduction_edges") or {}).keys())
    spillover_edges = set(dict(scene_weight_layers.get("spillover_weights") or {}).keys())
    affected_edges = incident_edges | blocked_edges | lane_reduction_edges | spillover_edges

    def _ratio(path_edges):
        path_list = [str(edge_id) for edge_id in list(path_edges or []) if str(edge_id).strip()]
        if not path_list:
            return float("nan")
        return len(set(path_list) & affected_edges) / max(len(path_list), 1)

    def _row(method_name: str, path_key: str, sumo_payload: dict[str, Any], travel_time_key: str) -> dict[str, Any]:
        path_edges = list(results.get(path_key) or sumo_payload.get("final_route_edges") or [])
        return {
            "scene": str(results.get("showcase_case_name") or results.get("scene_profile") or ""),
            "scene_profile": str(results.get("scene_profile") or ""),
            "method": method_name,
            "travel_time_s": _runtime_audit_safe_number(
                results.get(travel_time_key, sumo_payload.get("travel_time"))
            ),
            "waiting_time_s": _runtime_audit_safe_number(sumo_payload.get("waiting_time_s")),
            "time_loss_s": _runtime_audit_safe_number(sumo_payload.get("time_loss_s")),
            "stop_count": _runtime_audit_safe_number(sumo_payload.get("stop_count")),
            "depart_delay_s": _runtime_audit_safe_number(sumo_payload.get("depart_delay_s")),
            "route_length_edges": int(sumo_payload.get("route_length_edges") or len(path_edges or [])),
            "affected_edge_overlap_ratio": _runtime_audit_safe_number(
                sumo_payload.get("affected_edge_overlap_ratio", _ratio(path_edges))
            ),
            "incident_edge_overlap_ratio": _runtime_audit_safe_number(_ratio_with_edge_set(path_edges, incident_edges | blocked_edges | lane_reduction_edges)),
            "spillover_edge_overlap_ratio": _runtime_audit_safe_number(_ratio_with_edge_set(path_edges, spillover_edges)),
        }

    return [
        _row("planned", "planned_path", planned_sumo, "planned_sumo_travel_time_s"),
        _row("baseline", "baseline_path", baseline_sumo, "baseline_sumo_travel_time_s"),
    ]


def _ratio_with_edge_set(path_edges, edge_set: set[str]) -> float:
    path_list = [str(edge_id) for edge_id in list(path_edges or []) if str(edge_id).strip()]
    if not path_list:
        return float("nan")
    return len(set(path_list) & set(edge_set or set())) / max(len(path_list), 1)


def _write_runtime_audit_metrics(results: dict[str, Any]) -> str | None:
    rows = _runtime_audit_rows_from_results(results)
    if not rows:
        return None
    os.makedirs(RUNTIME_AUDIT_DIR, exist_ok=True)
    scene_key = str(results.get("showcase_case_name") or results.get("scene_profile") or "latest").strip() or "latest"
    safe_scene_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", scene_key).strip("_") or "latest"
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scene": scene_key,
        "scene_profile": str(results.get("scene_profile") or ""),
        "rows": rows,
        "explanation": (
            "Baseline travel time is high mainly because its route overlaps with runtime-injected affected edges. "
            "The model/planned route avoids part of this affected band."
        ),
    }
    json_path = os.path.join(RUNTIME_AUDIT_DIR, f"runtime_metrics_{safe_scene_key}.json")
    csv_path = os.path.join(RUNTIME_AUDIT_DIR, f"runtime_metrics_{safe_scene_key}.csv")
    latest_path = os.path.join(RUNTIME_AUDIT_DIR, "runtime_metrics_latest.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path


def _render_runtime_audit_expander(results: dict[str, Any] | None = None) -> None:
    with st.expander("Why is baseline slow?", expanded=False):
        audit_payload = None
        audit_path = None
        candidate_paths = []
        if results:
            scene_key = str(results.get("showcase_case_name") or results.get("scene_profile") or "").strip()
            if scene_key:
                safe_scene_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", scene_key).strip("_")
                candidate_paths.append(os.path.join(RUNTIME_AUDIT_DIR, f"runtime_metrics_{safe_scene_key}.json"))
        candidate_paths.extend(
            [
                os.path.join(RUNTIME_AUDIT_DIR, "runtime_metrics_latest.json"),
                os.path.join(RUNTIME_AUDIT_DIR, "audit_summary.json"),
            ]
        )
        for path_candidate in candidate_paths:
            if not os.path.exists(path_candidate):
                continue
            try:
                with open(path_candidate, "r", encoding="utf-8") as handle:
                    audit_payload = json.load(handle)
                audit_path = path_candidate
                break
            except Exception:
                audit_payload = None
        if not audit_payload:
            st.caption("runtime audit not available")
            return

        rows = list(audit_payload.get("rows") or [])
        if not rows and "baseline_affected_edge_overlap_ratio" in audit_payload:
            rows = [
                {
                    "method": "baseline",
                    "affected_edge_overlap_ratio": audit_payload.get("baseline_affected_edge_overlap_ratio"),
                    "incident_edge_overlap_ratio": audit_payload.get("baseline_incident_overlap_ratio"),
                    "spillover_edge_overlap_ratio": audit_payload.get("baseline_spillover_overlap_ratio"),
                },
                {
                    "method": "planned",
                    "affected_edge_overlap_ratio": audit_payload.get("planned_affected_edge_overlap_ratio"),
                    "incident_edge_overlap_ratio": audit_payload.get("planned_incident_overlap_ratio"),
                    "spillover_edge_overlap_ratio": audit_payload.get("planned_spillover_overlap_ratio"),
                },
            ]
        if not rows:
            st.caption("runtime audit not available")
            return

        df = pd.DataFrame(rows)
        wanted_cols = [
            "method",
            "affected_edge_overlap_ratio",
            "incident_edge_overlap_ratio",
            "spillover_edge_overlap_ratio",
            "waiting_time_s",
            "time_loss_s",
            "stop_count",
        ]
        visible_cols = [col for col in wanted_cols if col in df.columns]
        safe_df = make_arrow_safe(df[visible_cols])
        st.dataframe(safe_df, use_container_width=True, hide_index=True)
        baseline_row = next((row for row in rows if str(row.get("method")) == "baseline"), {})
        planned_row = next((row for row in rows if str(row.get("method")) == "planned"), {})
        baseline_crosses = (
            _safe_float(baseline_row.get("incident_edge_overlap_ratio"), 0.0) > 0
            or _safe_float(baseline_row.get("spillover_edge_overlap_ratio"), 0.0) > 0
        )
        planned_crosses = (
            _safe_float(planned_row.get("incident_edge_overlap_ratio"), 0.0) > 0
            or _safe_float(planned_row.get("spillover_edge_overlap_ratio"), 0.0) > 0
        )
        st.caption(
            f"baseline crosses affected edge: {'yes' if baseline_crosses else 'no'} | "
            f"planned crosses affected edge: {'yes' if planned_crosses else 'no'}"
        )
        st.caption(
            "Baseline travel time is high mainly because its route overlaps with runtime-injected affected edges. "
            "The model/planned route avoids part of this affected band."
        )
        if audit_path:
            st.caption(f"audit source: {audit_path}")


def _route_debug_rows(
    *,
    selected_scene_profile,
    selected_start_edge,
    selected_end_edge,
    baseline_path,
    planned_path,
    baseline_overlap_ratio,
    baseline_estimated_time,
    planned_estimated_time,
    baseline_sumo_travel_time,
    planned_sumo_travel_time,
    final_route_after_safe_route_build,
    whether_case_is_positive_gain,
):
    return [
        {"字段": "selected_scene_profile", "值": _route_debug_value(selected_scene_profile)},
        {"字段": "selected_start_edge", "值": _route_debug_value(selected_start_edge)},
        {"字段": "selected_end_edge", "值": _route_debug_value(selected_end_edge)},
        {"字段": "baseline_path", "值": _route_debug_value(list(baseline_path or []))},
        {"字段": "planned_path", "值": _route_debug_value(list(planned_path or []))},
        {"字段": "baseline_overlap_ratio", "值": _route_debug_value(baseline_overlap_ratio)},
        {"字段": "baseline_estimated_time", "值": _route_debug_value(baseline_estimated_time)},
        {"字段": "planned_estimated_time", "值": _route_debug_value(planned_estimated_time)},
        {"字段": "baseline_sumo_travel_time", "值": _route_debug_value(baseline_sumo_travel_time)},
        {"字段": "planned_sumo_travel_time", "值": _route_debug_value(planned_sumo_travel_time)},
        {"字段": "whether_case_is_positive_gain", "值": _route_debug_value(whether_case_is_positive_gain)},
        {
            "字段": "final_route_after_safe_route_build",
            "值": _route_debug_value(final_route_after_safe_route_build),
        },
    ]


def _render_metric_panel(title: str, value_text: str, subtitle: str, accent_color: str) -> None:
    st.markdown(
        f"""
        <div class="result-panel" style="border-left:6px solid {accent_color};">
            <div class="result-panel-title">{title}</div>
            <div class="result-panel-value">{value_text}</div>
            <div class="result-panel-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_showcase_time_panels(
    *,
    planning_time_display,
    model_infer_display,
    route_solve_display,
    baseline_actual_time,
    planned_actual_time,
    baseline_fallback_time,
    planned_fallback_time,
) -> None:
    time_bundle = _showcase_time_panel_bundle(
        baseline_actual_time=baseline_actual_time,
        planned_actual_time=planned_actual_time,
        baseline_fallback_time=baseline_fallback_time,
        planned_fallback_time=planned_fallback_time,
    )
    gain = dict(time_bundle.get("gain") or {})
    status = str(gain.get("status") or "unavailable")
    accent = {
        "positive": "#16A34A",
        "neutral": "#64748B",
        "negative": "#EA580C",
        "unavailable": "#94A3B8",
    }.get(status, "#94A3B8")
    source_suffix = "SUMO actual" if time_bundle.get("use_actual") else "fallback estimate"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_metric_panel(
            "规划耗时 Planning Time",
            _format_seconds_text(planning_time_display),
            " · ".join([
                _format_named_time_text("infer", model_infer_display),
                _format_named_time_text("solve", route_solve_display),
            ]),
            "#2563EB",
        )
    with c2:
        _render_metric_panel(
            f"{time_bundle['baseline_label']} Baseline",
            _format_seconds_text(time_bundle.get("baseline_time")),
            source_suffix,
            "#64748B",
        )
    with c3:
        _render_metric_panel(
            f"{time_bundle['planned_label']} Planned",
            _format_seconds_text(time_bundle.get("planned_time")),
            source_suffix,
            "#EA580C",
        )
    with c4:
        gain_value = str(gain.get("conclusion") or "收益结论：暂不可用").replace("收益结论：", "")
        gain_subtitle = "无法计算收益"
        if gain.get("delta_time") is not None and status != "unavailable":
            gain_subtitle = (
                f"时间差 {float(gain['delta_time']):.1f} s · 比例 {float(gain.get('delta_ratio_pct') or 0.0):.1f}%"
            )
        _render_metric_panel(
            "收益结论 Gain",
            gain_value,
            gain_subtitle,
            accent,
        )


def _render_dual_map_view(
    *,
    selector_label: str,
    view_key: str,
    folium_builder,
    plotly_builder,
    folium_component_key: str,
    folium_height: int = 460,
    plotly_height: int = 520,
    html_export_name: str | None = None,
    download_label: str | None = None,
    default_basemap: str = "nolabel_light",
    default_layer_overrides: dict[str, bool] | None = None,
):
    layer_defaults = {
        "reference_map": True,
        "road_skeleton": True,
        "mapped_nodes": False,
        "mapped_edges": False,
        "raw_incident_layer": True,
        "spillover_layer": True,
        "gat_smoothed_layer": False,
        "baseline_route": True,
        "planned_route": True,
        "congestion": True,
    }
    layer_labels = {
        "reference_map": "Reference map",
        "road_skeleton": "Road skeleton",
        "mapped_nodes": "Debug mapped nodes",
        "mapped_edges": "Debug mapped edges",
        "raw_incident_layer": "Raw incident",
        "spillover_layer": "Spillover",
        "gat_smoothed_layer": "GAT smoothed",
        "baseline_route": "baseline route",
        "planned_route": "planned route",
        "congestion": "Congestion",
    }
    for key, value in dict(default_layer_overrides or {}).items():
        if key in layer_defaults:
            layer_defaults[key] = bool(value)

    layer_visibility = {}
    with st.expander("地图图层设置", expanded=False):
        st.caption("这里可以切换地图图层；默认主展示已按 production showcase 视图配置。Debug mapped nodes / edges 仅建议在排查时开启。")
        debug_cols = st.columns(2)
        for idx, (key, label) in enumerate(layer_labels.items()):
            with debug_cols[idx % 2]:
                layer_visibility[key] = st.checkbox(
                    label,
                    value=st.session_state.get(f"{view_key}_{key}", layer_defaults[key]),
                    key=f"{view_key}_{key}",
                )

    basemap_options = {
        "No-Label Light Basemap": "nolabel_light",
        "Minimal": "minimal",
    }
    basemap_default_label = next(
        (label for label, token in basemap_options.items() if token == default_basemap),
        "No-Label Light Basemap",
    )
    basemap_label = st.radio(
        "Basemap",
        list(basemap_options.keys()),
        horizontal=True,
        index=list(basemap_options.keys()).index(basemap_default_label),
        key=f"{view_key}_basemap_mode",
        help="No-Label 适合答辩展示与截图；Minimal 会最大程度弱化底图。",
    )
    basemap_style = basemap_options[basemap_label]

    map_style = st.radio(
        "地图风格",
        ["Realistic Block Overlay Map", "Clean Schematic Map"],
        horizontal=True,
        key=f"{view_key}_style_mode",
        help="默认使用 Realistic Block Overlay Map：真实底图 + 规则图映射后的街区骨架覆盖层；Clean Schematic Map 保留更规则的论文示意图。",
    )
    map_view = st.radio(
        selector_label,
        ["Real Map (Folium)", "Fallback Flat Map (Plotly)"],
        horizontal=True,
        key=view_key,
        help="Folium 用于交互式论文演示；若 iframe / HTML 加载较慢或浏览器不兼容，可切换到 Plotly 备用平面图。",
    )
    st.caption("默认主展示：No-Label Light Basemap + Realistic Block Overlay Map + Real Map (Folium)。切换底图、风格或视图只改变展示层，不会重复计算核心规划逻辑。")

    if map_view == "Real Map (Folium)":
        st.info("Loading Real Map (Folium)... 若 HTML / iframe 加载较慢、空白或浏览器不兼容，可切换到 Fallback Flat Map (Plotly)。")
        try:
            with st.spinner("Loading Real Map (Folium)..."):
                map_obj, map_meta = folium_builder(map_style, layer_visibility, basemap_style)
                map_meta = sanitize_component_args(map_meta or {})
                folium_html = render_folium_html(map_obj)
            status_text = str((map_meta or {}).get("status_text", "")).strip()
            if status_text:
                st.caption(status_text)
            components.html(folium_html, height=folium_height, scrolling=False)
            if html_export_name and download_label:
                route_map_path = viz_engine.export_folium_map(map_obj, html_export_name)
                with open(route_map_path, "rb") as fp:
                    st.download_button(
                        download_label,
                        data=fp.read(),
                        file_name=html_export_name,
                        mime="text/html",
                        key=f"download_{view_key}",
                    )
            return "folium"
        except Exception as map_exc:
            logging.error("Folium render failed: %s", map_exc, exc_info=True)
            st.warning(
                f"Folium 加载失败或浏览器兼容性受限，已自动回退到 Plotly 平面图。"
                f" 也可手动切换为 `Fallback Flat Map (Plotly)`。错误：{map_exc}"
            )
            with st.expander("Folium debug", expanded=False):
                st.code(traceback.format_exc())

    else:
        st.info("当前使用 Fallback Flat Map (Plotly)。结果与主链路一致，仅切换展示方式。")

    plot_fig = plotly_builder(map_style)
    plot_fig.update_layout(height=plotly_height)
    st.plotly_chart(plot_fig, use_container_width=True)
    return "plotly"


def _format_percent_text(value, *, fallback: str = "N/A") -> str:
    parsed = _optional_float(value)
    if parsed is None:
        return fallback
    return f"{parsed * 100.0:.1f}%"


def _resolve_planned_path_source(display_payload: dict[str, Any]) -> str:
    source = str(display_payload.get("planned_path_source") or "").strip()
    if source:
        return source
    path_algo = str(display_payload.get("path_algo") or "").lower()
    if display_payload.get("showcase_structured_mode") or "scene-aware shortest path" in path_algo:
        return "scene_aware_shortest_path"
    if display_payload.get("fallback_used"):
        return "fallback"
    if "rule" in path_algo:
        return "rule"
    return "parser"


def _model_constraint_summary(display_payload: dict[str, Any], edge_total: int) -> dict[str, Any]:
    schema = build_model_parsed_constraints({**dict(display_payload), "total_edges": edge_total})
    return {
        "planned_path_source": schema["planned_path_source"],
        "parser_source": schema["parser_source"],
        "contribution": schema["model_contribution"],
        "anchor_count": int(schema["sparse_anchor_count"] or 0),
        "parsed_edges": int(schema["parsed_changed_edges"] or 0),
        "edge_total": int(schema["total_edges"] or edge_total or 0),
        "parse_confidence": _safe_float(schema["parse_confidence"], 0.0),
    }


def _scene_runtime_summary(display_payload: dict[str, Any]) -> dict[str, Any]:
    schema = build_scene_runtime_constraints(display_payload)
    return {
        "scene_profile": schema["scene_profile"],
        "raw_incident_edges": int(schema["raw_incident_edges"] or 0),
        "spillover_edges": int(schema["spillover_edges"] or 0),
        "runtime_spillover": "ON" if schema["runtime_spillover_enabled"] else "OFF",
        "baseline_overlap": schema["baseline_affected_overlap"],
        "planned_overlap": schema["planned_affected_overlap"],
        "actual_scene_affected_edges": int(schema["raw_incident_edges"] or 0) + int(schema["spillover_edges"] or 0),
    }


def _render_showcase_constraint_cards(
    display_payload: dict[str, Any],
    *,
    edge_total: int,
) -> None:
    model_summary = _model_constraint_summary(display_payload, edge_total)
    runtime_summary = _scene_runtime_summary(display_payload)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Model Parsed Constraints**")
        _render_metric_panel(
            "Parser Source",
            model_summary["parser_source"],
            f"planned_path_source={model_summary['planned_path_source']}",
            "#2563EB",
        )
        _render_metric_panel(
            "Sparse Anchors",
            str(model_summary["anchor_count"]),
            (
                f"Parsed Changed Edges: {model_summary['parsed_edges']}/{model_summary['edge_total']} · "
                f"Parse Confidence: {model_summary['parse_confidence']:.2f}"
            ),
            "#0F766E",
        )
        _render_metric_panel(
            "Model Contribution",
            model_summary["contribution"],
            "Active only when parser anchors drive the route constraints",
            "#7C3AED",
        )
        if model_summary["parser_source"] == "showcase_bypass":
            st.info(
                "Showcase structured mode: planned route is generated from scene-aware shortest path. "
                "This run is for scenario visualization, not parser performance evaluation."
            )
    with right:
        st.markdown("**Scene Runtime Constraints**")
        _render_metric_panel(
            "Scene Profile",
            runtime_summary["scene_profile"],
            f"SUMO Runtime Spillover: {runtime_summary['runtime_spillover']}",
            "#EA580C",
        )
        _render_metric_panel(
            "Injected Affected Edges",
            f"{runtime_summary['raw_incident_edges']} raw / {runtime_summary['spillover_edges']} spillover",
            "Runtime incident/spillover injection, separate from parser output",
            "#B45309",
        )
        _render_metric_panel(
            "Affected Overlap",
            f"baseline {_format_percent_text(runtime_summary['baseline_overlap'])}",
            f"planned {_format_percent_text(runtime_summary['planned_overlap'])}",
            "#64748B",
        )


def _render_showcase_route_cards(
    *,
    gain_summary: dict[str, Any],
) -> None:
    baseline_time = gain_summary.get("baseline_travel_time_s")
    planned_time = gain_summary.get("planned_travel_time_s")
    delta = gain_summary.get("time_saving_s")
    ratio_pct = gain_summary.get("saving_ratio_pct")
    source_suffix = str(gain_summary.get("gain_attribution") or "")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_metric_panel("Baseline Travel Time", _format_seconds_text(baseline_time), source_suffix, "#64748B")
    with c2:
        _render_metric_panel("Planned Travel Time", _format_seconds_text(planned_time), source_suffix, "#EA580C")
    with c3:
        _render_metric_panel("Time Saving", _format_seconds_text(delta), "baseline - planned", "#16A34A" if (delta or 0) > 0 else "#94A3B8")
    with c4:
        _render_metric_panel(
            "Saving Ratio",
            "N/A" if ratio_pct is None else f"{float(ratio_pct):.1f}%",
            "relative to baseline travel time",
            "#16A34A" if (ratio_pct or 0) > 0 else "#94A3B8",
        )


def _summarize_sparse_for_defense(display_payload, max_items: int = 3) -> str:
    anchor_rows = list(display_payload.get("anchor_rows") or [])
    anomaly_rows = list(display_payload.get("anomaly_rows") or [])
    if anchor_rows:
        return "；".join(
            f"{row['锚点描述']} ({row['异常等级']}, {row['映射边数']} edges)"
            for row in anchor_rows[:max_items]
        )
    if anomaly_rows:
        return "；".join(
            f"{row['异常边']} (w={row['规划权重']:.1f}, Δ={row['相对默认变化']:+.1f})"
            for row in anomaly_rows[:max_items]
        )
    return "未检测到显式异常锚点，当前路径按默认权重与模型输出补全后生成。"


def _scene_env_summary_text(scene_env_payload) -> str:
    if not scene_env_payload:
        return "场景环境摘要暂无可用数据。"
    reroute_status = str(scene_env_payload.get("reroute_effective_status") or "unknown")
    reroute_reason = str(scene_env_payload.get("reroute_reason") or "")
    time_context = _format_scene_time_context_text(scene_env_payload)
    return (
        f"TLS Profile: {scene_env_payload.get('tls_profile_name', 'unknown')} | "
        f"Reroute: {reroute_status}"
        f" ({_format_seconds_text(scene_env_payload.get('reroute_period'), precision=0)}) | "
        f"Seed: {scene_env_payload.get('simulation_seed', 'unknown')} | "
        f"{time_context} | "
        f"Incident: {scene_env_payload.get('incident_summary', '0 条')} | "
        f"Blocked: {scene_env_payload.get('blocked_summary', '0 条')} | "
        f"Lane Reduction: {scene_env_payload.get('lane_reduction_summary', '0 条')} | "
        f"Spillover: {scene_env_payload.get('spillover_summary', 'off')}"
        + (f" | Reroute Reason: {reroute_reason}" if reroute_reason else "")
    )


def _preview_sequence_text(values, limit: int = 4) -> str:
    seq = [str(item).strip() for item in list(values or []) if str(item).strip()]
    if not seq:
        return "none"
    preview = ", ".join(seq[:limit])
    if len(seq) > limit:
        preview += f", ... (+{len(seq) - limit})"
    return preview


def _preview_mapping_text(mapping, *, limit: int = 4, value_sep: str = "=") -> str:
    items = []
    for key, value in dict(mapping or {}).items():
        key_str = str(key).strip()
        if not key_str:
            continue
        items.append(f"{key_str}{value_sep}{value}")
    if not items:
        return "none"
    preview = ", ".join(items[:limit])
    if len(items) > limit:
        preview += f", ... (+{len(items) - limit})"
    return preview


def _format_weight_range_text(range_info) -> str:
    info = dict(range_info or {})
    if not info:
        return "—"
    min_value = info.get("min")
    max_value = info.get("max")
    if min_value is None or max_value is None:
        return "—"
    return f"{_safe_float(min_value, 0.0):.2f}-{_safe_float(max_value, 0.0):.2f}"


def _format_level_count_text(counts) -> str:
    info = dict(counts or {})
    ordered = [
        ("light", "light"),
        ("moderate", "moderate"),
        ("severe", "severe"),
        ("closed", "closed"),
    ]
    parts = [f"{label}={int(info.get(key, 0) or 0)}" for key, label in ordered if int(info.get(key, 0) or 0) > 0]
    return ", ".join(parts) if parts else "none"


def _build_overlay_difference_rows(
    *,
    spillover_stats,
    raw_weights,
    spillover_weights,
    final_planning_weights,
    gat_applied: bool,
    signal_edges: int,
) -> list[dict[str, str | int]]:
    stats = dict(spillover_stats or {})
    raw_only = dict(raw_weights or {})
    spillover_only = dict(spillover_weights or {})
    rows = [
        {
            "Overlay": "Raw Incident Layer",
            "Scope": "主受扰边 / 封闭 / lane reduction",
            "Edges": int(stats.get("raw_incident_count", len(raw_only)) or 0),
            "Weight Band": _format_weight_range_text(stats.get("raw_weight_range")),
            "Level Mix": _format_level_count_text(stats.get("raw_level_counts")),
            "Meaning": "benchmark raw scene 口径；showcase/benchmark 共用的直接事件边",
        },
        {
            "Overlay": "Spillover Layer",
            "Scope": (
                f"1-hop={int(stats.get('spillover_1hop_count', 0) or 0)} / "
                f"2-hop={int(stats.get('spillover_2hop_count', 0) or 0)}"
            ),
            "Edges": int(stats.get("spillover_total_count", len(spillover_only)) or 0),
            "Weight Band": (
                f"1-hop {_format_weight_range_text(stats.get('spillover_1hop_weight_range'))} | "
                f"2-hop {_format_weight_range_text(stats.get('spillover_2hop_weight_range'))}"
            ),
            "Level Mix": (
                f"1-hop {_format_level_count_text(stats.get('spillover_1hop_level_counts'))} | "
                f"2-hop {_format_level_count_text(stats.get('spillover_2hop_level_counts'))}"
            ),
            "Meaning": "仅 showcase spillover-enhanced scene 启用；同时进入 planner / overlay / SUMO runtime，不写回 benchmark 主结果",
        },
    ]
    if gat_applied:
        non_default_final = {
            edge_id: weight
            for edge_id, weight in dict(final_planning_weights or {}).items()
            if abs(_safe_float(weight, 2.0) - 2.0) > 1e-6
        }
        rows.append(
            {
                "Overlay": "GAT-Smoothed Layer",
                "Scope": "LLM → GAT refinement (exploratory only)",
                "Edges": int(len(non_default_final)),
                "Weight Band": _format_weight_range_text(_weight_range_for_rows(non_default_final)),
                "Level Mix": f"signal_edges={int(signal_edges or len(non_default_final))}",
                "Meaning": "exploratory overlay only；若与 raw/spillover 差异稳定可复现，才进入重训候选讨论",
            }
        )
    else:
        rows.append(
            {
                "Overlay": "GAT-Smoothed Layer",
                "Scope": "LLM → GAT refinement (exploratory only)",
                "Edges": 0,
                "Weight Band": "N/A",
                "Level Mix": "not enabled",
                "Meaning": "本次未启用 GAT smooth overlay；当前展示重点是 raw incident 与拓扑 spillover 的差异",
            }
        )
    return rows


def _weight_range_for_rows(weights) -> dict[str, float] | None:
    values = [_safe_float(weight, 2.0) for weight in dict(weights or {}).values()]
    if not values:
        return None
    return {"min": min(values), "max": max(values)}


def _build_scene_env_payload(scene_profile, scene_weight_layers=None, *, scene_mode: str = "benchmark") -> dict:
    scene_cfg = scene_profile_to_dict(scene_profile)
    reroute_policy = build_reroute_policy(scene_profile)
    speed_profile = dict(scene_cfg.get("speed_profile") or {})
    incident_edges = dict(scene_cfg.get("incident_edges") or {})
    blocked_edges = list(scene_cfg.get("blocked_edges") or [])
    lane_reduction_edges = dict(scene_cfg.get("lane_reduction_edges") or {})
    scene_weight_layers = dict(scene_weight_layers or {})
    spillover_stats = dict(scene_weight_layers.get("stats") or {})
    spillover_enabled = bool(scene_weight_layers.get("spillover_enabled"))
    return {
        "scene_name": str(scene_cfg.get("scene_name") or "unknown"),
        "scene_type": str(scene_cfg.get("scene_type") or "unknown"),
        "speed_profile_rows": [
            {"参数": "base_speed_factor", "值": speed_profile.get("base_speed_factor", "—")},
            {"参数": "blocked_speed_factor", "值": speed_profile.get("blocked_speed_factor", "—")},
            {
                "参数": "lane_reduction_speed_factor",
                "值": speed_profile.get("lane_reduction_speed_factor", "—"),
            },
            {"参数": "minimum_edge_speed_mps", "值": speed_profile.get("minimum_edge_speed_mps", "—")},
        ],
        "tls_profile_name": str(scene_cfg.get("tls_profile_name") or "unknown"),
        "reroute_enabled": bool(reroute_policy.get("reroute_enabled")),
        "reroute_period": int(reroute_policy.get("reroute_period", 0) or 0),
        "reroute_reason": str(reroute_policy.get("reason") or ""),
        "reroute_effective_status": str(reroute_policy.get("effective_status") or "unknown"),
        "simulation_seed": int(scene_cfg.get("simulation_seed", 0) or 0),
        "current_scene_time_s": scene_cfg.get("current_scene_time_s"),
        "warmup_seconds": int(scene_cfg.get("warmup_seconds", 0) or 0),
        "evaluation_start_time": int(scene_cfg.get("evaluation_start_time", 0) or 0),
        "evaluation_end_time": int(scene_cfg.get("evaluation_end_time", 0) or 0),
        "incident_summary": (
            f"{len(incident_edges)} 条: {_preview_mapping_text(incident_edges)}"
            if incident_edges else "0 条"
        ),
        "blocked_summary": (
            f"{len(blocked_edges)} 条: {_preview_sequence_text(blocked_edges)}"
            if blocked_edges else "0 条"
        ),
        "lane_reduction_summary": (
            f"{len(lane_reduction_edges)} 条: {_preview_mapping_text(lane_reduction_edges, value_sep='→')}"
            if lane_reduction_edges else "0 条"
        ),
        "scene_mode": str(scene_mode or "benchmark"),
        "spillover_enabled": spillover_enabled,
        "spillover_summary": (
            f"{'showcase_on' if spillover_enabled else 'benchmark_raw'} · "
            f"raw={int(spillover_stats.get('raw_incident_count', 0))} · "
            f"1-hop={int(spillover_stats.get('spillover_1hop_count', 0))} · "
            f"2-hop={int(spillover_stats.get('spillover_2hop_count', 0))}"
        ),
        "spillover_stats": spillover_stats,
        "config_snapshot": scene_cfg,
        "reroute_policy": reroute_policy,
    }


def _build_request_signature(
    *,
    scene_profile_name: str,
    scene_type: str,
    workspace_scene_mode: str,
    start_edge: str,
    end_edge: str,
    traffic_constraint: str,
    path_algo: str,
    selected_model: str,
    use_gat: bool,
    gat_alpha: float,
    use_manual: bool,
    showcase_case_name: str = "",
) -> str:
    payload = {
        "scene_profile_name": str(scene_profile_name or "").strip(),
        "scene_type": str(scene_type or "").strip(),
        "workspace_scene_mode": str(workspace_scene_mode or "benchmark").strip(),
        "start_edge": str(start_edge or "").strip(),
        "end_edge": str(end_edge or "").strip(),
        "traffic_constraint": str(traffic_constraint or "").strip(),
        "path_algo": str(path_algo or "").strip(),
        "selected_model": canonical_method_name(selected_model),
        "use_gat": bool(use_gat),
        "gat_alpha": round(_safe_float(gat_alpha, 0.0), 3),
        "use_manual": bool(use_manual),
        "showcase_case_name": str(showcase_case_name or "").strip(),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _build_result_display_payload(result_record, *, edge_total: int, path_algo: str, scene_weights=None):
    record = dict(result_record or {})
    time_fields = normalize_eval_time_fields(record)
    sparse_parse = dict(record.get("sparse_parse") or {})
    weights = dict(record.get("final_planning_weights") or record.get("weights") or {})
    mapped_weights = dict(sparse_parse.get("mapped_weights") or record.get("raw_llm_weights") or {})
    edge_confidence = dict(sparse_parse.get("edge_confidence") or {})
    anchors = list(sparse_parse.get("anchors") or [])
    path = list(record.get("path") or [])
    scene_cfg = dict(record.get("scene_profile_config") or {})
    scene_weight_layers = dict(record.get("scene_weight_layers") or {})
    reroute_policy = dict(record.get("scene_reroute_policy") or {})
    sumo_result = dict(record.get("sumo_result") or {})
    baseline_sumo_result = dict(record.get("baseline_sumo_result") or {})
    baseline_path = list(record.get("baseline_path") or [])
    planned_path = list(record.get("planned_path") or path)
    baseline_estimated_time = record.get("baseline_estimated_time_s")
    planned_estimated_time = record.get("planned_estimated_time_s")
    baseline_sumo_travel_time = record.get("baseline_sumo_travel_time_s")
    planned_sumo_travel_time = record.get("planned_sumo_travel_time_s", time_fields.get("travel_time_s"))
    baseline_overlap_ratio = _edge_overlap_ratio(baseline_path, planned_path)
    if planned_sumo_travel_time is not None and baseline_sumo_travel_time is not None:
        positive_gain = bool(float(planned_sumo_travel_time) + 1e-9 < float(baseline_sumo_travel_time))
    elif planned_estimated_time is not None and baseline_estimated_time is not None:
        positive_gain = bool(float(planned_estimated_time) + 1e-9 < float(baseline_estimated_time))
    else:
        positive_gain = None
    final_route_after_safe_route_build = record.get("final_route_after_safe_route_build")
    if not final_route_after_safe_route_build:
        final_route_after_safe_route_build = {
            "baseline": list(baseline_sumo_result.get("final_route_edges") or []),
            "planned": list(sumo_result.get("final_route_edges") or []),
        }
    evaluation_window = list(
        record.get("scene_evaluation_window")
        or [
            scene_cfg.get("evaluation_start_time"),
            scene_cfg.get("evaluation_end_time"),
        ]
    )

    anchor_rows = []
    for anchor in anchors:
        road = str(anchor.get("road") or "未识别").strip()
        dir_label = str(anchor.get("dir") or "未指明").strip()
        range_label = str(anchor.get("range") or "局部").strip() or "局部"
        edges = list(anchor.get("edges") or expand_anchor_edges(road, dir_label, range_label))
        anchor_rows.append(
            {
                "锚点描述": f"{road} · {dir_label} · {range_label}",
                "异常等级": str(anchor.get("level") or "待判断"),
                "映射边数": len(edges),
                "代表边": ", ".join(edges[:3]) + (" ..." if len(edges) > 3 else ""),
                "置信度": round(_safe_float(anchor.get("confidence"), 0.0), 2),
            }
        )

    anomaly_rows = []
    anomaly_source = mapped_weights or {
        edge_id: weight
        for edge_id, weight in weights.items()
        if abs(_safe_float(weight, 2.0) - 2.0) > 1e-6
    }
    for edge_id, weight in sorted(
        anomaly_source.items(),
        key=lambda item: (abs(_safe_float(item[1], 2.0) - 2.0), _safe_float(item[1], 2.0)),
        reverse=True,
    ):
        delta = _safe_float(weight, 2.0) - 2.0
        if abs(delta) < 1e-6:
            continue
        anomaly_rows.append(
            {
                "异常边": _format_edge_label(edge_id),
                "规划权重": round(_safe_float(weight, 2.0), 2),
                "相对默认变化": round(delta, 2),
                "置信度": round(_safe_float(edge_confidence.get(edge_id), 0.0), 2),
            }
        )

    route_rows = []
    for idx, edge_id in enumerate(path, start=1):
        route_weight = _safe_float(weights.get(edge_id, 2.0), 2.0)
        row = {
            "段序": idx,
            "路段": _format_edge_label(edge_id),
            "规划权重": round(route_weight, 2),
            "相对默认变化": round(route_weight - 2.0, 2),
        }
        if scene_weights is not None:
            row["环境权重"] = round(_safe_float(scene_weights.get(edge_id, 2.0), 2.0), 2)
        route_rows.append(row)

    route_focus_rows = sorted(
        route_rows,
        key=lambda row: (row.get("规划权重", 0.0), abs(row.get("相对默认变化", 0.0))),
        reverse=True,
    )

    reroute_summary = str(reroute_policy.get("effective_status") or "disabled")
    reroute_reason = str(reroute_policy.get("reason") or "")
    if bool(reroute_policy.get("reroute_enabled")):
        reroute_summary = (
            f"enabled · period={int(reroute_policy.get('reroute_period', 0))}s · "
            f"threshold={_safe_float(reroute_policy.get('reroute_threshold_factor'), 1.0):.2f}x + "
            f"{_safe_float(reroute_policy.get('reroute_threshold_constant'), 0.0):.1f}s"
        )
    elif reroute_reason:
        reroute_summary = f"disabled · {reroute_reason}"

    exploratory_flags = []
    if bool(record.get("gat_applied")):
        exploratory_flags.append("GAT enabled (exploratory only, not mainline)")
    if _IS_EXPLORATORY_METHOD(record.get("model", "")) and not exploratory_flags:
        exploratory_flags.append("exploratory method selected")

    anchor_count = int(sparse_parse.get("anchor_count", len(anchor_rows)))
    parsed_edges = int(record.get("parsed_edges", len(mapped_weights)))
    parsed_edge_ratio = _safe_float(
        record.get("parsed_edge_ratio"),
        (parsed_edges / max(edge_total, 1) * 100.0),
    )
    signal_edges = int(record.get("signal_edges", 0))
    signal_ratio = _safe_float(
        record.get("signal_ratio"),
        (signal_edges / max(edge_total, 1) * 100.0),
    )
    spillover_stats = dict(scene_weight_layers.get("stats") or {})
    raw_scene_weights = dict(record.get("scene_raw_weights") or scene_weight_layers.get("raw_only_weights") or {})
    spillover_scene_weights = dict(record.get("scene_spillover_weights") or scene_weight_layers.get("spillover_weights") or {})
    scene_mode = str(record.get("scene_mode") or "benchmark")
    sumo_runtime_spillover_summary = dict(record.get("sumo_runtime_spillover_summary") or {})
    overlay_diff_rows = _build_overlay_difference_rows(
        spillover_stats=spillover_stats,
        raw_weights=raw_scene_weights,
        spillover_weights=spillover_scene_weights,
        final_planning_weights=weights,
        gat_applied=bool(record.get("gat_applied")),
        signal_edges=signal_edges,
    )

    eval_start = evaluation_window[0] if len(evaluation_window) >= 1 else scene_cfg.get("evaluation_start_time")
    eval_end = evaluation_window[1] if len(evaluation_window) >= 2 else scene_cfg.get("evaluation_end_time")

    return {
        "model_name": canonical_method_name(record.get("model", "")),
        "scene_profile": str(record.get("scene_profile") or scene_cfg.get("scene_name") or "unknown"),
        "scene_type": str(record.get("scene_type") or sparse_parse.get("scene_type") or "unknown"),
        "scene_profile_config": scene_cfg,
        "path_algo": path_algo,
        "planned_path_source": str(record.get("planned_path_source") or ""),
        "raw_output": str(record.get("raw_output") or record.get("raw_model_output") or ""),
        "llm_parsed": record.get("llm_parsed"),
        "sparse_parse": dict(sparse_parse),
        "edge_confidence": dict(edge_confidence),
        "conflicts": list(sparse_parse.get("conflicts") or []),
        "protected_edges": list(sparse_parse.get("protected_edges") or []),
        "path": path,
        "planned_path": planned_path,
        "baseline_path": baseline_path,
        "path_summary": _summarize_path(path),
        "start_edge": str(record.get("start") or ""),
        "end_edge": str(record.get("end") or ""),
        "total_cost": _safe_float(record.get("total_cost"), 0.0),
        "time_fields": time_fields,
        "avg_speed": _safe_float(record.get("avg_speed"), 0.0),
        "anchor_rows": anchor_rows,
        "anomaly_rows": anomaly_rows,
        "route_rows": route_rows,
        "route_focus_rows": route_focus_rows,
        "anchor_count": anchor_count,
        "parse_confidence": _safe_float(sparse_parse.get("parse_confidence"), 0.0),
        "parsed_edges": parsed_edges,
        "total_edges": int(edge_total or 0),
        "parsed_edge_ratio": parsed_edge_ratio,
        "signal_edges": signal_edges,
        "signal_ratio": signal_ratio,
        "scene_mode": scene_mode,
        "workspace_scene_mode": str(record.get("workspace_scene_mode") or scene_mode),
        "showcase_structured_mode": bool(record.get("showcase_structured_mode")),
        "showcase_case_name": str(record.get("showcase_case_name") or ""),
        "showcase_display_reason": str(record.get("showcase_display_reason") or ""),
        "showcase_affected_edges": list(record.get("showcase_affected_edges") or []),
        "showcase_saving_seconds": (
            None if record.get("showcase_saving_seconds") is None else float(record.get("showcase_saving_seconds"))
        ),
        "showcase_saving_ratio": (
            None if record.get("showcase_saving_ratio") is None else float(record.get("showcase_saving_ratio"))
        ),
        "scene_weight_layers": scene_weight_layers,
        "scene_raw_weights": raw_scene_weights,
        "scene_spillover_weights": spillover_scene_weights,
        "spillover_stats": spillover_stats,
        "sumo_runtime_spillover_summary": sumo_runtime_spillover_summary,
        "overlay_diff_rows": overlay_diff_rows,
        "gat_applied": bool(record.get("gat_applied")),
        "final_planning_weights": dict(weights),
        "exploratory_summary": "; ".join(exploratory_flags) if exploratory_flags else "未启用 exploratory 选项",
        "exploratory_enabled": bool(exploratory_flags),
        "tls_profile": str(record.get("scene_tls_profile") or scene_cfg.get("tls_profile_name") or "unknown"),
        "reroute_summary": reroute_summary,
        "reroute_reason": reroute_reason,
        "seed": int(scene_cfg.get("simulation_seed", 0) or 0),
        "current_scene_time_s": record.get("current_scene_time_s"),
        "warmup_seconds": int(scene_cfg.get("warmup_seconds", 0) or 0),
        "evaluation_start_time": int(eval_start or 0),
        "evaluation_end_time": int(eval_end or 0),
        "protocol_snapshot": dict(record.get("sim_eval_protocol") or {}),
        "baseline_estimated_time": (
            None if baseline_estimated_time is None else float(baseline_estimated_time)
        ),
        "planned_estimated_time": (
            None if planned_estimated_time is None else float(planned_estimated_time)
        ),
        "baseline_sumo_travel_time": (
            None if baseline_sumo_travel_time is None else float(baseline_sumo_travel_time)
        ),
        "planned_sumo_travel_time": (
            None if planned_sumo_travel_time is None else float(planned_sumo_travel_time)
        ),
        "waiting_time_s": record.get("waiting_time_s"),
        "time_loss_s": record.get("time_loss_s"),
        "stop_count": record.get("stop_count"),
        "baseline_waiting_time_s": record.get("baseline_waiting_time_s"),
        "baseline_time_loss_s": record.get("baseline_time_loss_s"),
        "baseline_stop_count": record.get("baseline_stop_count"),
        "baseline_sumo_ok": bool(baseline_sumo_result.get("ok")),
        "baseline_sumo_stage": str(baseline_sumo_result.get("stage") or "not_run"),
        "baseline_sumo_reason": str(baseline_sumo_result.get("reason") or ""),
        "baseline_sumo_error_message": str(baseline_sumo_result.get("error_message") or ""),
        "baseline_sumo_route_summary": (
            f"start {baseline_sumo_result.get('original_start_edge', '—')} -> {baseline_sumo_result.get('final_start_edge', '—')} | "
            f"end {baseline_sumo_result.get('original_end_edge', '—')} -> {baseline_sumo_result.get('final_end_edge', '—')}"
        ),
        "baseline_sumo_route_repair_actions": list(baseline_sumo_result.get("route_repair_actions") or []),
        "final_route_after_safe_route_build": final_route_after_safe_route_build,
        "baseline_overlap_ratio": baseline_overlap_ratio,
        "planned_affected_edge_overlap_ratio": record.get("affected_edge_overlap_ratio"),
        "baseline_affected_edge_overlap_ratio": record.get("baseline_affected_edge_overlap_ratio"),
        "whether_case_is_positive_gain": positive_gain,
        "fallback_used": bool(record.get("fallback_used")),
        "fallback_reason": str(record.get("fallback_reason") or ""),
        "route_debug_rows": _route_debug_rows(
            selected_scene_profile=str(record.get("scene_profile") or scene_cfg.get("scene_name") or "unknown"),
            selected_start_edge=str(record.get("start") or ""),
            selected_end_edge=str(record.get("end") or ""),
            baseline_path=baseline_path,
            planned_path=planned_path,
            baseline_overlap_ratio=baseline_overlap_ratio,
            baseline_estimated_time=baseline_estimated_time,
            planned_estimated_time=planned_estimated_time,
            baseline_sumo_travel_time=baseline_sumo_travel_time,
            planned_sumo_travel_time=planned_sumo_travel_time,
            final_route_after_safe_route_build=final_route_after_safe_route_build,
            whether_case_is_positive_gain=positive_gain,
        ),
        "sumo_ok": bool(sumo_result.get("ok")),
        "sumo_stage": str(sumo_result.get("stage") or "not_run"),
        "sumo_reason": str(sumo_result.get("reason") or ""),
        "sumo_error_message": str(sumo_result.get("error_message") or ""),
        "sumo_route_summary": (
            f"start {sumo_result.get('original_start_edge', '—')} -> {sumo_result.get('final_start_edge', '—')} | "
            f"end {sumo_result.get('original_end_edge', '—')} -> {sumo_result.get('final_end_edge', '—')}"
        ),
        "sumo_route_repair_actions": list(sumo_result.get("route_repair_actions") or []),
        "sumo_speed_status": str(sumo_result.get("speed_status") or "not_started"),
        "sumo_speed_reason": str(sumo_result.get("speed_reason") or ""),
        "sumo_speed_samples": int(sumo_result.get("speed_samples_collected", 0) or 0),
        "sumo_debug_events": list(sumo_result.get("debug_events") or []),
    }


def _build_showcase_preview_record(active_showcase_case, scene_profile, *, scene_mode: str) -> dict[str, Any]:
    scene_weight_layers = dict(active_showcase_case.get("scene_weight_layers") or {})
    planned_time_s = _safe_float(active_showcase_case.get("planned_time_s"), 0.0)
    baseline_time_s = _safe_float(active_showcase_case.get("baseline_time_s"), 0.0)
    planned_path = list(active_showcase_case.get("planned_path") or [])
    baseline_path = list(active_showcase_case.get("baseline_path") or [])
    protocol = build_sim_eval_protocol(scene_profile)
    return {
        "timestamp": datetime.now().isoformat(),
        "model": MAINLINE_METHOD_NAME,
        "path_algo": "Showcase / Scene-aware Shortest Path",
        "planned_path_source": "scene_aware_shortest_path",
        "mode": "preview",
        "scene_type": scene_profile.scene_type,
        "detected_scene_type": scene_profile.scene_type,
        "constraint": str(active_showcase_case.get("display_reason") or ""),
        "scene_profile": scene_profile.scene_name,
        "scene_profile_config": scene_profile_to_dict(scene_profile),
        "scene_mode": scene_mode,
        "workspace_scene_mode": scene_mode,
        "showcase_structured_mode": True,
        "showcase_case_name": str(active_showcase_case.get("showcase_name") or ""),
        "showcase_display_reason": str(active_showcase_case.get("display_reason") or ""),
        "showcase_affected_edges": list(active_showcase_case.get("affected_edges") or []),
        "showcase_saving_seconds": active_showcase_case.get("saving_seconds"),
        "showcase_saving_ratio": active_showcase_case.get("saving_ratio"),
        "showcase_preferred_baseline_type": str(active_showcase_case.get("preferred_baseline_type") or ""),
        "scene_weight_layers": scene_weight_layers,
        "scene_raw_weights": dict(scene_weight_layers.get("raw_only_weights") or {}),
        "scene_spillover_weights": dict(scene_weight_layers.get("spillover_weights") or {}),
        "scene_spillover_hops": dict(scene_weight_layers.get("spillover_hop_by_edge") or {}),
        "sim_eval_protocol": protocol_to_dict(protocol),
        "scene_tls_profile": scene_profile.tls_profile_name,
        "scene_evaluation_window": [
            int(scene_profile.evaluation_start_time),
            int(scene_profile.evaluation_end_time),
        ],
        "scene_reroute_policy": build_reroute_policy(scene_profile),
        "sumo_result": {
            "ok": True,
            "travel_time": planned_time_s,
            "avg_speed": None,
            "speed_status": "preview",
            "speed_reason": "showcase_preview",
            "runtime_spillover_summary": {},
            "original_start_edge": active_showcase_case.get("start_edge"),
            "final_start_edge": active_showcase_case.get("start_edge"),
            "original_end_edge": active_showcase_case.get("end_edge"),
            "final_end_edge": active_showcase_case.get("end_edge"),
            "final_route_edges": planned_path,
            "route_repair_actions": [],
            "debug_events": [],
        },
        "baseline_sumo_result": {
            "ok": True,
            "travel_time": baseline_time_s,
            "avg_speed": None,
            "speed_status": "preview",
            "speed_reason": "showcase_preview",
            "runtime_spillover_summary": {},
            "original_start_edge": active_showcase_case.get("start_edge"),
            "final_start_edge": active_showcase_case.get("start_edge"),
            "original_end_edge": active_showcase_case.get("end_edge"),
            "final_end_edge": active_showcase_case.get("end_edge"),
            "final_route_edges": baseline_path,
            "route_repair_actions": [],
            "debug_events": [],
        },
        "sumo_runtime_spillover_summary": {},
        "baseline_sumo_runtime_spillover_summary": {},
        "start": str(active_showcase_case.get("start_edge") or ""),
        "end": str(active_showcase_case.get("end_edge") or ""),
        "path": planned_path,
        "planned_path": planned_path,
        "baseline_path": baseline_path,
        "baseline_path_source": str(active_showcase_case.get("preferred_baseline_type") or "free_flow_shortest_path"),
        "total_cost": _safe_float(
            (active_showcase_case.get("debug") or {}).get("planned_selection_cost_s"),
            planned_time_s,
        ),
        "travel_time": planned_time_s,
        "avg_speed": 0.0,
        "pos_data": [],
        "speed_data": [],
        "weights": dict(active_showcase_case.get("scene_weights") or {}),
        "final_planning_weights": dict(active_showcase_case.get("scene_weights") or {}),
        "sparse_parse": {
            "scene_type": scene_profile.scene_type,
            "anchors": [],
            "anchor_count": 0,
            "mapped_weights": {},
            "edge_confidence": {},
            "parse_confidence": 1.0,
            "conflicts": [],
            "protected_edges": list(active_showcase_case.get("affected_edges") or []),
        },
        "gat_applied": False,
        "planning_time_s": 0.0,
        "model_infer_time_s": 0.0,
        "route_solve_time_s": 0.0,
        "baseline_estimated_time_s": baseline_time_s,
        "planned_estimated_time_s": planned_time_s,
        "baseline_sumo_travel_time_s": baseline_time_s,
        "planned_sumo_travel_time_s": planned_time_s,
        "planned_affected_edge_overlap_ratio": _edge_overlap_ratio(planned_path, active_showcase_case.get("affected_edges") or []),
        "baseline_affected_edge_overlap_ratio": _edge_overlap_ratio(baseline_path, active_showcase_case.get("affected_edges") or []),
        "whether_case_is_positive_gain": bool(planned_time_s + 1e-9 < baseline_time_s),
        "parsed_edges": 0,
        "parsed_edge_ratio": 0.0,
        "signal_edges": 0,
        "signal_ratio": 0.0,
        "scene_profile_selection_mode": "structured_showcase_preview",
        "scene_profile_locked": True,
        "fallback_used": False,
        "fallback_reason": "showcase_preview",
    }


def _render_sumo_status_banner(display_payload) -> None:
    if display_payload.get("sumo_ok"):
        if display_payload.get("sumo_speed_status") == "not_enough_speed_samples":
            st.info(
                "本次未形成有效速度曲线：速度样本不足。"
                f" 原因：{display_payload.get('sumo_speed_reason', 'unknown')}，"
                f" 样本数={display_payload.get('sumo_speed_samples', 0)}。"
            )
        return

    reason = display_payload.get("sumo_error_message") or display_payload.get("sumo_reason") or "unknown"
    st.error(
        "SUMO 本次未完成有效发车/路径执行。"
        f" stage={display_payload.get('sumo_stage', 'unknown')}，reason={reason}"
    )
    repair_actions = list(display_payload.get("sumo_route_repair_actions") or [])
    if repair_actions:
        st.caption("SUMO route repair: " + " | ".join(repair_actions))
    st.caption("SUMO route debug: " + str(display_payload.get("sumo_route_summary", "—")))


def _render_main_result_panels(
    *,
    weight_ph,
    path_ph,
    metric_ph,
    display_payload,
    display_mode: str,
    edge_total: int,
    weight_dict,
    path,
    edge_coords,
    severe: int,
    medium: int,
    clear: int,
    travel_time: float,
    avg_speed: float,
    comply_rate: float,
    baseline_path_ref,
    scene_weights,
    speed_data,
    defense_mode: bool = False,
    show_debug_details: bool = False,
    scene_env_payload=None,
):
    time_fields = display_payload["time_fields"]
    concise_mode = display_mode == "答辩简洁模式"
    anchor_limit = 4 if concise_mode else 10
    anomaly_limit = 4 if concise_mode else 12
    route_limit = 6 if concise_mode else 20
    map_congestion = dict(scene_weights or weight_dict or {})
    baseline_path = list(baseline_path_ref or display_payload.get("baseline_path") or [])
    planned_path = list(display_payload.get("planned_path") or path or [])
    net_for_map, _ = _load_net_cached(config.get("SUMO_NET_PATH"))
    planned_route_map_time = display_payload.get("planned_estimated_time")
    if planned_route_map_time is None:
        planned_route_map_time = (
            _estimate_route_time_on_weights(net_for_map, planned_path, map_congestion) if planned_path else None
        )
    baseline_route_map_time = display_payload.get("baseline_estimated_time")
    if baseline_route_map_time is None:
        baseline_route_map_time = (
            _estimate_route_time_on_weights(net_for_map, baseline_path, map_congestion)
            if baseline_path else None
        )
    planned_sumo_travel_time = display_payload.get("planned_sumo_travel_time")
    baseline_sumo_travel_time = display_payload.get("baseline_sumo_travel_time")
    if planned_sumo_travel_time is not None and baseline_sumo_travel_time is not None:
        comparison_source = "SUMO actual travel time"
        comparison_planned_time = float(planned_sumo_travel_time)
        comparison_baseline_time = float(baseline_sumo_travel_time)
    else:
        comparison_source = "environment-weight estimate"
        comparison_planned_time = planned_route_map_time
        comparison_baseline_time = baseline_route_map_time
    if comparison_planned_time is not None and comparison_baseline_time is not None:
        delta_value = float(comparison_baseline_time) - float(comparison_planned_time)
        delta_ratio_pct = (delta_value / max(float(comparison_baseline_time), 1.0)) * 100.0
        positive_gain_case = delta_value > 1e-9
        if abs(delta_value) <= max(3.0, abs(float(comparison_baseline_time)) * 0.003):
            route_case_summary = "Neutral Diagnostic Case · 规划路线行程时间与基线持平/接近 · 路径与脆弱走廊关系可解释"
        elif positive_gain_case:
            route_case_summary = (
                f"Positive Gain Showcase · 节省时间 {delta_value:.1f}s · 节省比例 {delta_ratio_pct:.1f}%"
            )
        else:
            route_case_summary = f"Negative Case · 规划路线行程时间增加 {abs(delta_value):.1f}s"
        route_case_summary += f"（{comparison_source}）"
    else:
        positive_gain_case = None
        route_case_summary = "收益估算暂不可用"
    showcase_presentation = _resolve_showcase_presentation(display_payload)
    benefit_box_payload = _build_benefit_box_payload(
        display_payload,
        comparison_baseline_time=comparison_baseline_time,
        comparison_planned_time=comparison_planned_time,
        comparison_source=comparison_source,
    )

    route_debug_df = pd.DataFrame(display_payload.get("route_debug_rows") or [])
    overlay_diff_df = pd.DataFrame(display_payload.get("overlay_diff_rows") or [])
    planned_safe_route = list((display_payload.get("final_route_after_safe_route_build") or {}).get("planned") or [])
    baseline_safe_route = list((display_payload.get("final_route_after_safe_route_build") or {}).get("baseline") or [])
    planned_route_debug = (
        f"planned: {display_payload.get('sumo_route_summary', '—')} | "
        f"baseline: {display_payload.get('baseline_sumo_route_summary', '—')}"
    )
    showcase_scope_lines = [
        "这里展示的是当前冻结 showcase 的地图、路径与收益结果。",
        "Benchmark 结论请以 `Benchmark / 冻结论文表` 中的 final frozen main table / ablation 为准。",
    ]
    if display_payload.get("showcase_structured_mode"):
        showcase_scope_lines.append(
            "当前案例为结构化 showcase：baseline = free-flow shortest path，planned = scene-aware shortest path。"
        )

    model_summary = _model_constraint_summary(display_payload, edge_total)
    runtime_summary = _scene_runtime_summary(display_payload)
    gain_summary = build_route_gain_summary(display_payload)

    weight_ph.empty()
    with weight_ph.container():
        _render_scope_banner("Showcase Overview", showcase_scope_lines)
        if model_summary["anchor_count"] == 0 and model_summary["planned_path_source"] == "scene_aware_shortest_path":
            st.warning(
                "当前输出未形成显式 sparse anchors；该 showcase 使用 scene-aware shortest path 生成规划路线，因此收益不能归因于 Sparse-LoRA-v2 parser。"
            )
        if model_summary["parsed_edges"] == 0 and runtime_summary["actual_scene_affected_edges"] > 0:
            st.info(
                "模型解析异常边为 0，但 SUMO 场景仍包含 runtime incident/spillover 注入；请区分模型输出与场景注入。"
            )
        st.markdown(
            f"""
            <div class="result-panel">
            <div style="font-size:1.14rem;font-weight:800;color:#0F172A">{html.escape(showcase_presentation['title'])}</div>
            <div style="margin-top:4px">{_build_badge_html(showcase_presentation['subtitle'], 'positive' if benefit_box_payload.get('status') == 'positive' else 'neutral')}</div>
            <div style="margin-top:10px;line-height:1.75">
            {"<br>".join(html.escape(str(line)) for line in list(showcase_presentation.get("summary_lines") or []))}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**{html.escape(str(gain_summary['gain_title']))}**")
        _render_benefit_box(benefit_box_payload)
        _render_showcase_route_cards(gain_summary=gain_summary)

    path_ph.empty()
    with path_ph.container():
        st.markdown('<div class="sec-hdr">🗺️ Showcase Route Map</div>', unsafe_allow_html=True)
        with st.expander("Map legend and topology propagation notes", expanded=False):
            st.markdown(
                f"""
                <div class="info-strip">
                <b>Layers</b>: Full road skeleton · Raw incident · Spillover · baseline route · planned route<br>
                <b>Runtime scene</b>: mode={display_payload.get('scene_mode', 'benchmark')} · raw={runtime_summary['raw_incident_edges']} · spillover={runtime_summary['spillover_edges']}<br>
                <b>Route</b>: {display_payload['path_summary']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        _render_dual_map_view(
            selector_label="地图显示模式",
            view_key="planned_route_map_view",
            folium_builder=lambda style, layer_visibility=None, basemap_style="nolabel_light": viz_engine.build_folium_network_map(
                weight_dict=map_congestion,
                raw_incident_weights=display_payload.get("scene_raw_weights"),
                spillover_weights=display_payload.get("scene_spillover_weights"),
                spillover_hops=(display_payload.get("scene_weight_layers") or {}).get("spillover_hop_by_edge"),
                gat_smoothed_weights=(display_payload.get("final_planning_weights") if display_payload.get("gat_applied") else None),
                overlay_stats=display_payload.get("spillover_stats"),
                path=planned_path,
                focus_path=True,
                baseline_path=baseline_path,
                planned_time_s=comparison_planned_time,
                baseline_time_s=comparison_baseline_time,
                map_style=style,
                basemap_style=basemap_style,
                layer_visibility=layer_visibility,
            ),
            plotly_builder=lambda style: viz_engine.plot_route_plotly(
                edge_coords,
                planned_path,
                map_congestion,
                baseline_path=baseline_path,
                planned_time_s=comparison_planned_time,
                baseline_time_s=comparison_baseline_time,
                map_style=style,
            ),
            folium_component_key="planned_route_folium",
            folium_height=700,
            plotly_height=700,
            html_export_name="planned_route_latest.html",
            download_label="下载路径 HTML 地图",
            default_layer_overrides={"gat_smoothed_layer": bool(display_payload.get("gat_applied"))},
        )

    metric_ph.empty()
    with metric_ph.container():
        _render_showcase_constraint_cards(display_payload, edge_total=edge_total)
        _render_runtime_audit_expander(st.session_state.get("results"))
        render_debug_panel(display_payload)
    return

    if defense_mode:
        weight_ph.empty()
        with weight_ph.container():
            st.markdown('<div class="sec-hdr">🛡️ Showcase Summary</div>', unsafe_allow_html=True)
            _render_scope_banner("Showcase Overview", showcase_scope_lines)
            st.markdown(
                f"""
                <div class="result-panel">
                <div style="font-size:1.1rem;font-weight:800;color:#0F172A">{html.escape(showcase_presentation['title'])}</div>
                <div style="margin-top:4px">{_build_badge_html(showcase_presentation['subtitle'], 'positive' if benefit_box_payload.get('status') == 'positive' else 'neutral')}</div>
                <div style="margin-top:10px;line-height:1.75">
                {"<br>".join(html.escape(str(line)) for line in list(showcase_presentation.get("summary_lines") or []))}
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_benefit_box(benefit_box_payload)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Showcase", display_payload["scene_profile"])
            k2.metric("主方法", MAINLINE_METHOD_NAME)
            k3.metric("Sparse Anchors", f"{display_payload['anchor_count']} 个")
            k4.metric("路径段数", f"{len(display_payload['path'])} 段")
            st.markdown(
                f"""
                <div class="info-strip">
                <b>方法摘要</b>：{MAINLINE_METHOD_NAME} 作为默认主链路，负责生成结构化规划路径。<br>
                <b>稀疏异常摘要</b>：{_summarize_sparse_for_defense(display_payload)}<br>
                <b>最终路径</b>：{display_payload['path_summary']}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if show_debug_details:
                st.markdown("**Debug: sparse anchors / 关键异常明细**")
                if display_payload["anchor_rows"]:
                    _safe_table(pd.DataFrame(display_payload["anchor_rows"][:anchor_limit]))
                if display_payload["anomaly_rows"]:
                    _safe_table(pd.DataFrame(display_payload["anomaly_rows"][:anomaly_limit]))

        path_ph.empty()
        with path_ph.container():
            st.markdown(
                f"""
                <div class="info-strip">
                <b>路径概览</b>：{display_payload['path_summary']}<br>
                <b>展示图层</b>：Full road skeleton · Raw incident · Spillover · baseline route · planned route<br>
                <b>场景外溢</b>：mode={display_payload.get('scene_mode', 'benchmark')} · raw={int((display_payload.get('spillover_stats') or {}).get('raw_incident_count', 0))} ·
                1-hop={int((display_payload.get('spillover_stats') or {}).get('spillover_1hop_count', 0))} ·
                2-hop={int((display_payload.get('spillover_stats') or {}).get('spillover_2hop_count', 0))}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if show_debug_details and not route_debug_df.empty:
                st.markdown("**路线收益调试字段**")
                _safe_table(route_debug_df)
            if show_debug_details and not overlay_diff_df.empty:
                st.markdown("**Raw / Spillover / GAT overlay 对照**")
                _safe_table(overlay_diff_df)
            _render_dual_map_view(
                selector_label="地图显示模式",
                view_key="planned_route_map_view",
                folium_builder=lambda style, layer_visibility=None, basemap_style="nolabel_light": viz_engine.build_folium_network_map(
                    weight_dict=map_congestion,
                    raw_incident_weights=display_payload.get("scene_raw_weights"),
                    spillover_weights=display_payload.get("scene_spillover_weights"),
                    spillover_hops=(display_payload.get("scene_weight_layers") or {}).get("spillover_hop_by_edge"),
                    gat_smoothed_weights=(display_payload.get("final_planning_weights") if display_payload.get("gat_applied") else None),
                    overlay_stats=display_payload.get("spillover_stats"),
                    path=planned_path,
                    focus_path=True,
                    baseline_path=baseline_path,
                    planned_time_s=comparison_planned_time,
                    baseline_time_s=comparison_baseline_time,
                    map_style=style,
                    basemap_style=basemap_style,
                    layer_visibility=layer_visibility,
                ),
                plotly_builder=lambda style: viz_engine.plot_route_plotly(
                    edge_coords,
                    planned_path,
                    map_congestion,
                    baseline_path=baseline_path,
                    planned_time_s=comparison_planned_time,
                    baseline_time_s=comparison_baseline_time,
                    map_style=style,
                ),
                folium_component_key="planned_route_folium",
                folium_height=460,
                plotly_height=520,
                html_export_name="planned_route_latest.html",
                download_label="下载路径 HTML 地图",
                default_layer_overrides={"gat_smoothed_layer": bool(display_payload.get("gat_applied"))},
            )

        metric_ph.empty()
        with metric_ph.container():
            planning_time_display = _optional_float(time_fields.get("planning_time_s"))
            model_infer_display = _optional_float(time_fields.get("model_infer_time_s"))
            route_solve_display = _optional_float(time_fields.get("route_solve_time_s"))
            travel_time_display = (
                _optional_float(planned_sumo_travel_time)
                if planned_sumo_travel_time is not None
                else _optional_float(time_fields.get("travel_time_s"))
            )
            _render_sumo_status_banner(display_payload)
            if baseline_path and not display_payload.get("baseline_sumo_ok"):
                st.info(
                    "baseline 路线未形成有效 SUMO travel time，收益状态已回退到估算时间。"
                    f" stage={display_payload.get('baseline_sumo_stage', 'unknown')}，"
                    f" reason={display_payload.get('baseline_sumo_error_message') or display_payload.get('baseline_sumo_reason', 'unknown')}"
                )
            _render_showcase_time_panels(
                planning_time_display=planning_time_display,
                model_infer_display=model_infer_display,
                route_solve_display=route_solve_display,
                baseline_actual_time=baseline_sumo_travel_time,
                planned_actual_time=planned_sumo_travel_time,
                baseline_fallback_time=comparison_baseline_time,
                planned_fallback_time=comparison_planned_time,
            )
            st.markdown(
                f"""
                <div class="info-strip">
                <b>场景环境摘要</b>：{_scene_env_summary_text(scene_env_payload)}<br>
                <b>时间上下文</b>：{_format_scene_time_context_text(scene_env_payload)}<br>
                <b>时间口径</b>：Planning Time = 模型推理 + 路径求解；Travel Time = 路线在 SUMO 中的实际行程时间。<br>
                <b>收益口径</b>：优先使用 SUMO 实际行程时间；若基线或规划路线未完成 SUMO，则回退到估算时间。<br>
                <b>safe route build</b>：planned {len(planned_safe_route)} edges / baseline {len(baseline_safe_route)} edges
                </div>
                """,
                unsafe_allow_html=True,
            )
            if show_debug_details:
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("scene_type", display_payload["scene_type"])
                d2.metric("Model Infer Time", _format_seconds_text(model_infer_display))
                d3.metric("Route Solve Time", _format_seconds_text(route_solve_display))
                d4.metric("avg_speed", f"{avg_speed:.1f} km/h")
                if display_payload["route_rows"]:
                    with st.expander("Debug: route solve 明细", expanded=False):
                        _safe_table(pd.DataFrame(display_payload["route_rows"][:route_limit]))
                if scene_env_payload:
                    with st.expander("Debug: 场景环境细表", expanded=False):
                        env_df = pd.DataFrame(
                            [
                                {"配置项": "scene_profile", "当前值": scene_env_payload.get("scene_name", "unknown")},
                                {"配置项": "speed_profile", "当前值": _preview_mapping_text(scene_env_payload.get("config_snapshot", {}).get("speed_profile", {}))},
                                {"配置项": "tls_profile_name", "当前值": scene_env_payload.get("tls_profile_name", "unknown")},
                                {
                                    "配置项": "reroute",
                                    "当前值": (
                                        f"{scene_env_payload.get('reroute_effective_status')} / "
                                        f"{_format_seconds_text(scene_env_payload.get('reroute_period'), precision=0)} / "
                                        f"{scene_env_payload.get('reroute_reason')}"
                                    ),
                                },
                                {"配置项": "simulation_seed", "当前值": scene_env_payload.get("simulation_seed", 0)},
                                {"配置项": "Current Scene Time", "当前值": _format_seconds_text(scene_env_payload.get("current_scene_time_s"))},
                                {"配置项": "warmup_seconds", "当前值": _format_seconds_text(scene_env_payload.get("warmup_seconds"), precision=0)},
                                {
                                    "配置项": "evaluation window",
                                    "当前值": _format_time_window_text(
                                        scene_env_payload.get("evaluation_start_time"),
                                        scene_env_payload.get("evaluation_end_time"),
                                    ),
                                },
                                {"配置项": "incident", "当前值": scene_env_payload.get("incident_summary", "0 条")},
                                {"配置项": "blocked", "当前值": scene_env_payload.get("blocked_summary", "0 条")},
                                {"配置项": "lane reduction", "当前值": scene_env_payload.get("lane_reduction_summary", "0 条")},
                            ]
                        )
                        _safe_table(env_df)
                if display_payload["sumo_debug_events"]:
                    with st.expander("Debug: SUMO route / vehicle lifecycle", expanded=False):
                        _safe_table(pd.DataFrame(display_payload["sumo_debug_events"]))
            return

    weight_ph.empty()
    with weight_ph.container():
        _render_scope_banner("Showcase Overview", showcase_scope_lines)
        st.markdown(
            f"""
            <div class="result-panel">
            <div style="font-size:1.14rem;font-weight:800;color:#0F172A">{html.escape(showcase_presentation['title'])}</div>
            <div style="margin-top:4px">{_build_badge_html(showcase_presentation['subtitle'], 'positive' if benefit_box_payload.get('status') == 'positive' else 'neutral')}</div>
            <div style="margin-top:10px;line-height:1.75">
            {"<br>".join(html.escape(str(line)) for line in list(showcase_presentation.get("summary_lines") or []))}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_benefit_box(benefit_box_payload)

        st.markdown('<div class="sec-hdr">方法摘要</div>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("showcase", display_payload["scene_profile"])
        a2.metric("Sparse Anchors", f"{display_payload['anchor_count']} 个")
        a3.metric("关键异常边", f"{display_payload['parsed_edges']}/{edge_total}")
        a4.metric(
            "Exploratory",
            "ON" if display_payload["exploratory_enabled"] else "OFF",
            delta=display_payload["exploratory_summary"],
            delta_color="off",
        )
        st.markdown(
            f"""
            <div class="info-strip">
            <b>主方法</b>：{display_payload['model_name']} &nbsp;|&nbsp;
            <b>scene_type</b>：{display_payload['scene_type']} &nbsp;|&nbsp;
            <b>parse_confidence</b>：{display_payload['parse_confidence']:.2f} &nbsp;|&nbsp;
            <b>非默认信号边</b>：{display_payload['signal_edges']}/{edge_total} ({display_payload['signal_ratio']:.1f}%) &nbsp;|&nbsp;
            <b>actual scene raw/spillover</b>：{int((display_payload.get('spillover_stats') or {}).get('raw_incident_count', 0))}/
            {int((display_payload.get('spillover_stats') or {}).get('spillover_total_count', 0))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if display_payload["anchor_rows"]:
            st.markdown("**Sparse anchors**")
            _safe_table(pd.DataFrame(display_payload["anchor_rows"][:anchor_limit]))
        else:
            st.info("当前输出未形成显式 sparse anchors，规划器按默认权重与已解析异常边继续执行。")
        _actual_scene_edges = int((display_payload.get("spillover_stats") or {}).get("raw_incident_count", 0)) + int((display_payload.get("spillover_stats") or {}).get("spillover_total_count", 0))
        if _actual_scene_edges > int(display_payload.get("parsed_edges", 0)):
            st.caption(
                f"区分说明：稀疏解析当前只显式覆盖 {int(display_payload.get('parsed_edges', 0))} 条异常边，"
                f"而实际场景（raw + spillover）涉及约 {_actual_scene_edges} 条边；若两者差距较大，说明模型对拓扑外溢仍有优化空间。"
            )

        if display_payload["anomaly_rows"]:
            if concise_mode:
                top_anomaly_text = "；".join(
                    f"{row['异常边']} (w={row['规划权重']:.1f}, Δ={row['相对默认变化']:+.1f})"
                    for row in display_payload["anomaly_rows"][:3]
                )
                st.markdown(
                    f'<div class="info-strip"><b>关键异常</b>：{top_anomaly_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("**关键异常映射**")
                _safe_table(pd.DataFrame(display_payload["anomaly_rows"][:anomaly_limit]))

        st.markdown('<div class="sec-hdr">规划摘要</div>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Planner", display_payload["path_algo"].split("（")[0].strip())
        b2.metric("路径段数", f"{len(display_payload['path'])} 段")
        b3.metric("路径总代价", f"{display_payload['total_cost']:.2f}")
        b4.metric("Route Solve Time", _format_seconds_text(time_fields.get("route_solve_time_s")))
        st.markdown(
            f"""
            <div class="info-strip">
            <b>起终点</b>：{display_payload['start_edge']} → {display_payload['end_edge']} &nbsp;|&nbsp;
            <b>Route</b>：{display_payload['path_summary']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if display_payload["route_focus_rows"]:
            st.markdown("**边权 / 代价关键变化**")
            _safe_table(pd.DataFrame(display_payload["route_focus_rows"][:route_limit]))
        else:
            st.info("当前没有可展示的 route solve 结果。")

        if not concise_mode and display_payload["route_rows"]:
            with st.expander("查看完整 route solve 细节", expanded=False):
                _safe_table(pd.DataFrame(display_payload["route_rows"]))
                full_top_rows = [
                    {
                        "路段": _format_edge_label(eid),
                        "规划权重": round(_safe_float(w, 2.0), 2),
                        "拥堵等级": (
                            "Severe" if _safe_float(w, 2.0) >= 8.5 else
                            "Heavy" if _safe_float(w, 2.0) >= 6.0 else
                            "Medium" if _safe_float(w, 2.0) >= 4.0 else
                            "Light" if _safe_float(w, 2.0) >= 2.5 else
                            "Clear"
                        ),
                    }
                    for eid, w in sorted(weight_dict.items(), key=lambda item: -item[1])[:12]
                ]
                st.markdown("**全局 Top 拥堵路段**")
                _safe_table(pd.DataFrame(full_top_rows))

    path_ph.empty()
    with path_ph.container():
        st.markdown(
            f"""
            <div class="info-strip">
            <b>路径概览</b>：{display_payload['path_summary']}<br>
            <b>展示图层</b>：Full road skeleton · Raw incident · Spillover · baseline route · planned route<br>
            <b>绕行关系</b>：基线路径与规划路径已同步显示<br>
            <b>场景外溢</b>：mode={display_payload.get('scene_mode', 'benchmark')} · raw={int((display_payload.get('spillover_stats') or {}).get('raw_incident_count', 0))} ·
            1-hop={int((display_payload.get('spillover_stats') or {}).get('spillover_1hop_count', 0))} ·
            2-hop={int((display_payload.get('spillover_stats') or {}).get('spillover_2hop_count', 0))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if (not concise_mode) and not overlay_diff_df.empty:
            st.markdown("**Raw / Spillover / GAT overlay 对照**")
            _safe_table(overlay_diff_df)
        if (not concise_mode) and not route_debug_df.empty:
            st.markdown("**路线收益调试字段**")
            _safe_table(route_debug_df)
        _render_dual_map_view(
            selector_label="地图显示模式",
            view_key="planned_route_map_view",
            folium_builder=lambda style, layer_visibility=None, basemap_style="nolabel_light": viz_engine.build_folium_network_map(
                weight_dict=map_congestion,
                raw_incident_weights=display_payload.get("scene_raw_weights"),
                spillover_weights=display_payload.get("scene_spillover_weights"),
                spillover_hops=(display_payload.get("scene_weight_layers") or {}).get("spillover_hop_by_edge"),
                gat_smoothed_weights=(display_payload.get("final_planning_weights") if display_payload.get("gat_applied") else None),
                overlay_stats=display_payload.get("spillover_stats"),
                path=planned_path,
                focus_path=True,
                baseline_path=baseline_path,
                planned_time_s=comparison_planned_time,
                baseline_time_s=comparison_baseline_time,
                map_style=style,
                basemap_style=basemap_style,
                layer_visibility=layer_visibility,
            ),
            plotly_builder=lambda style: viz_engine.plot_route_plotly(
                edge_coords,
                planned_path,
                map_congestion,
                baseline_path=baseline_path,
                planned_time_s=comparison_planned_time,
                baseline_time_s=comparison_baseline_time,
                map_style=style,
            ),
            folium_component_key="planned_route_folium",
            folium_height=460,
            plotly_height=520,
            html_export_name="planned_route_latest.html",
            download_label="下载路径 HTML 地图",
            default_layer_overrides={"gat_smoothed_layer": bool(display_payload.get("gat_applied"))},
        )

    metric_ph.empty()
    with metric_ph.container():
        planning_time_display = _optional_float(time_fields.get("planning_time_s"))
        model_infer_display = _optional_float(time_fields.get("model_infer_time_s"))
        route_solve_display = _optional_float(time_fields.get("route_solve_time_s"))
        travel_time_display = (
            _optional_float(planned_sumo_travel_time)
            if planned_sumo_travel_time is not None
            else _optional_float(time_fields.get("travel_time_s"))
        )
        _render_sumo_status_banner(display_payload)
        if baseline_path and not display_payload.get("baseline_sumo_ok"):
            st.info(
                "baseline 路线未形成有效 SUMO travel time，收益状态已回退到估算时间。"
                f" stage={display_payload.get('baseline_sumo_stage', 'unknown')}，"
                f" reason={display_payload.get('baseline_sumo_error_message') or display_payload.get('baseline_sumo_reason', 'unknown')}"
            )
        st.caption("时间口径：Planning Time = 模型推理 + 路径求解；Travel Time = 路线实际行程时间。收益优先显示 SUMO actual，fallback estimate 仅作次级参考。")

        m1, m2, m3, m4 = st.columns(4)
        delta_t = None
        if comparison_planned_time is not None and comparison_baseline_time is not None:
            delta_t = comparison_baseline_time - comparison_planned_time
        st.markdown('<div class="sec-hdr">时间指标</div>', unsafe_allow_html=True)
        _render_showcase_time_panels(
            planning_time_display=planning_time_display,
            model_infer_display=model_infer_display,
            route_solve_display=route_solve_display,
            baseline_actual_time=baseline_sumo_travel_time,
            planned_actual_time=planned_sumo_travel_time,
            baseline_fallback_time=comparison_baseline_time,
            planned_fallback_time=comparison_planned_time,
        )
        m1.metric("Model Infer Time", _format_seconds_text(model_infer_display))
        m2.metric("Route Solve Time", _format_seconds_text(route_solve_display))
        m3.metric("avg_speed", f"{avg_speed:.1f} km/h" if planned_sumo_travel_time is not None else "N/A")
        m4.metric("约束符合率", f"{comply_rate:.0f} %")
        if not concise_mode:
            st.markdown(
                f"""
                <div class="info-strip">
                <b>baseline_estimated_time</b>：{_format_seconds_text(baseline_route_map_time)} &nbsp;|&nbsp;
                <b>planned_estimated_time</b>：{_format_seconds_text(planned_route_map_time)}<br>
                <b>baseline_sumo_travel_time</b>：{_format_seconds_text(baseline_sumo_travel_time)} &nbsp;|&nbsp;
                <b>planned_sumo_travel_time</b>：{_format_seconds_text(planned_sumo_travel_time)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sec-hdr">环境配置摘要</div>', unsafe_allow_html=True)
        env1, env2 = st.columns(2)
        with env1:
            _render_metric_panel(
                "TLS profile",
                display_payload["tls_profile"],
                f"scene_profile={display_payload['scene_profile']}",
                "#0F766E",
            )
        with env2:
            _render_metric_panel(
                "seed",
                str(display_payload["seed"]),
                _format_scene_time_context_text(display_payload),
                "#7C3AED",
            )
        st.markdown(
            f'<div class="info-strip"><b>reroute policy</b>：{display_payload["reroute_summary"]}<br><b>时间上下文</b>：{_format_scene_time_context_text(display_payload)}</div>',
            unsafe_allow_html=True,
        )

        env_debug_df = pd.DataFrame(
            [
                {"配置项": "TLS profile", "当前值": display_payload["tls_profile"]},
                {"配置项": "reroute policy", "当前值": display_payload["reroute_summary"]},
                {"配置项": "seed", "当前值": display_payload["seed"]},
                {"配置项": "Current Scene Time", "当前值": _format_seconds_text(display_payload.get("current_scene_time_s"))},
                {
                    "配置项": "warmup / evaluation window",
                    "当前值": (
                        f"{_format_seconds_text(display_payload.get('warmup_seconds'), precision=0)} / "
                        f"{_format_time_window_text(display_payload.get('evaluation_start_time'), display_payload.get('evaluation_end_time'))}"
                    ),
                },
            ]
        )

        chart_container = None
        if not concise_mode:
            chart_container = st.container()
        elif show_debug_details:
            chart_container = st.expander(
                "展开调试图表（速度曲线 / 路径甘特图 / 协议快照）",
                expanded=False,
            )
        if chart_container is not None:
            with chart_container:
                if not concise_mode:
                    st.markdown("**环境配置细表**")
                    _safe_table(env_debug_df)
                    if display_payload["protocol_snapshot"]:
                        st.markdown("**协议快照**")
                        proto_df = pd.DataFrame(
                            [
                                {"字段": key, "值": value}
                                for key, value in display_payload["protocol_snapshot"].items()
                            ]
                        )
                        _safe_table(proto_df)
                    if display_payload["sumo_debug_events"]:
                        st.markdown("**SUMO debug**")
                        _safe_table(pd.DataFrame(display_payload["sumo_debug_events"]))
                if display_payload["sumo_ok"] and speed_data:
                    s_col, g_col = st.columns([3, 2])
                    with s_col:
                        st.markdown('<div class="sec-hdr">车辆速度曲线</div>', unsafe_allow_html=True)
                        spd_fig = viz_engine.plot_speed_plotly(speed_data, config.get("STEP_LENGTH", 0.1))
                        spd_fig.update_layout(height=320)
                        st.plotly_chart(spd_fig, use_container_width=True)
                    with g_col:
                        st.markdown('<div class="sec-hdr">路径甘特图</div>', unsafe_allow_html=True)
                        tt_for_gantt = travel_time if travel_time > 0 else sum(weight_dict.get(e, 1.0) for e in path) * 3.0
                        gantt_fig = viz_engine.plot_path_gantt(path, weight_dict, tt_for_gantt, scene_weights=scene_weights)
                        if gantt_fig:
                            gantt_fig.update_layout(height=320)
                            st.plotly_chart(gantt_fig, use_container_width=True)
                else:
                    st.info(
                        "本次未形成有效速度曲线。"
                        f" 原因：{display_payload.get('sumo_speed_reason', display_payload.get('sumo_reason', 'unknown'))}"
                        f"；speed_status={display_payload.get('sumo_speed_status', 'unknown')}"
                        f"；samples={display_payload.get('sumo_speed_samples', 0)}"
                    )


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
    .result-panel{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:16px 18px;
        box-shadow:0 8px 18px rgba(15,23,42,.04);margin:6px 0 10px}
    .result-panel-title{font-size:.85rem;font-weight:700;color:#334155;margin-bottom:6px}
    .result-panel-value{font-size:1.9rem;font-weight:800;color:#0F172A;line-height:1.15}
    .result-panel-subtitle{font-size:.82rem;color:#475569;margin-top:6px}
    .group-strip{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px}
    .group-pill{background:#F8FAFC;border:1px solid #CBD5E1;border-radius:999px;padding:6px 10px;
        font-size:.78rem;color:#334155}
    .frozen-source-strip{margin-bottom:4px}
    .paper-table-wrap{overflow-x:auto;border:1px solid #CBD5E1;border-radius:12px;background:#FFF;
        box-shadow:0 2px 8px rgba(15,23,42,.06)}
    .paper-table{width:100%;border-collapse:separate;border-spacing:0;font-size:.9rem}
    .paper-table th{position:sticky;top:0;background:#1E3A8A;color:#FFFFFF;font-weight:700;
        text-align:left;padding:13px 12px;border-bottom:2px solid #1E40AF;white-space:nowrap;
        letter-spacing:.01em}
    .paper-table td{padding:11px 12px;border-bottom:1px solid #EEF2F7;vertical-align:middle;
        white-space:nowrap;color:#1E293B}
    .paper-table tbody tr:hover td{background:#F0F9FF}
    .paper-table .metric-best{font-weight:800;border-left:4px solid #16A34A;background:#F0FDF4;
        color:#14532D}
    .paper-table .metric-second{background:#FEF9C3;font-weight:600}
    .paper-row.mainline-row td{background:#DCFCE7;font-weight:600}
    .paper-row.mainline-row:hover td{background:#BBF7D0}
    .paper-row.extended-row td{background:#FFFBEB;color:#78350F;font-style:italic}
    .paper-row.fullmethod-row td{background:#DCFCE7;font-weight:600}
    .paper-row.ablation-focus-row td{background:#EFF6FF}
    .paper-delta{font-size:.76rem;font-weight:700;margin-left:6px}
    .paper-delta-positive{color:#B91C1C}
    .paper-delta-negative{color:#047857}
    .paper-delta-neutral{color:#64748B}
    .paper-badge{display:inline-block;border-radius:999px;padding:2px 9px;font-size:.72rem;font-weight:700;margin:1px 4px 1px 0}
    .badge-neutral{background:#E2E8F0;color:#334155}
    .badge-positive{background:#DCFCE7;color:#166534}
    .badge-warning{background:#FEF3C7;color:#92400E}
    .badge-main{background:#DBEAFE;color:#1D4ED8}
    </style>
    """, unsafe_allow_html=True)

    for k, v in [("results", {}), ("total_cost", 0.0), ("travel_time", 0.0),
                 ("last_raw", ""), ("last_wd", {}), ("last_raw_llm_weights", {}),
                 ("last_mode", "demo_mode"), ("last_fallback_used", False),
                 ("last_fallback_reason", "none"), ("last_path", None),
                 ("last_sparse_parse", {}), ("last_gat_meta", {}),
                 ("last_speed_data", []), ("last_pos_data", []),
                 ("last_scene_weights", {}), ("last_comply_rate", 0.0),
                 ("last_scene_weight_layers", {}), ("last_scene_raw_weights", {}),
                 ("last_scene_spillover_weights", {}),
                 ("last_baseline_path", []), ("last_baseline_estimated_time", None),
                 ("last_planned_estimated_time", None),
                 ("last_baseline_sumo_result", {}), ("last_baseline_sumo_travel_time", None),
                 ("last_dijkstra_ref", []), ("last_dijkstra_cost", 0.0),
                 ("last_result_signature", ""), ("active_request_signature", ""),
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
    sub = f"{MAINLINE_METHOD_NAME} 主链路 · stage4_fix · planner / SUMO / visualization · 规划次数: {run_cnt}"
    if best_t > 0:
        sub += f" · 最佳记录: {best_t:.1f}s"
    st.markdown('<div class="main-title">🚗 Sparse-LoRA-v2 交通路径规划可视化</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{sub}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-strip">
        <b>当前默认展示</b>：Production Showcase + Frozen Benchmark Tables<br>
        <b>主方法</b>：Sparse-LoRA-v2<br>
        <b>时间口径</b>：Planning Time = 模型推理 + 路径求解；Travel Time = SUMO 实际通行时间
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # 侧边栏
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1rem;font-weight:700;color:#1E3A8A;margin-bottom:8px">'
            '⚙️ 实验配置</div>', unsafe_allow_html=True)

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

        workspace_mode_label = st.radio(
            "展示工作区",
            ["Showcase Mode", "Benchmark Mode"],
            index=0,
            key="workspace_mode_label",
            horizontal=True,
            help="Showcase Mode：用于答辩演示，优先使用结构化 showcase 场景与稳定展示；Benchmark Mode：关闭 showcase 优化，只保留 benchmark/raw scene 口径。",
        )
        workspace_scene_mode = "showcase" if workspace_mode_label == "Showcase Mode" else "benchmark"
        if workspace_scene_mode == "showcase":
            st.caption("当前为 Showcase Mode：推荐场景会直接走结构化 baseline/planned 路径，不再依赖 parser。")
        else:
            st.caption("当前为 Benchmark Mode：不启用 showcase 场景优化，展示与 benchmark/raw scene 口径保持分离。")

        defense_mode = st.toggle(
            "Defense Mode",
            value=False,
            key="defense_mode",
            help="开启后只保留答辩最关键的信息：场景、主方法、sparse 摘要、最终路径、Planning / Travel Time、场景环境摘要和地图。",
        )
        if "defense_debug_open" not in st.session_state:
            st.session_state["defense_debug_open"] = False
        if defense_mode:
            if st.button("Show Debug Details", use_container_width=True, key="defense_debug_btn"):
                st.session_state["defense_debug_open"] = not st.session_state.get("defense_debug_open", False)
            show_debug_details = bool(st.session_state.get("defense_debug_open", False))
            display_mode = "答辩简洁模式"
            st.info("Defense Mode 已启用：主界面只保留答辩关键结论。")
            if show_debug_details:
                st.caption("Debug details 已展开；再次点击 `Show Debug Details` 可收起。")
        else:
            st.session_state["defense_debug_open"] = False
            show_debug_details = False
            display_mode = st.radio(
                "结果展示级别",
                ["答辩简洁模式", "调试详细模式"],
                key="result_display_mode",
                horizontal=True,
                help="答辩简洁模式：突出模型理解 / 规划 / 时间 / 环境四个结论块；调试详细模式：展开更细的路径和协议细节。切换后若需要刷新结果区，请重新运行规划。",
            )

        st.markdown('<div class="sec-hdr">🤖 默认主方法</div>', unsafe_allow_html=True)
        selected_model = MAINLINE_METHOD_NAME
        st.selectbox(
            "主界面默认展示方法",
            [MAINLINE_METHOD_NAME],
            index=0,
            disabled=True,
            key="mainline_method_lock",
            help="主界面默认展示链路锁定为 Sparse-LoRA-v2，上游冻结为 stage4_fix。",
        )

        _model_path_map = {
            MAINLINE_METHOD_NAME: MAINLINE_MODEL_PATH,
            "Qwen-LoRA (baseline)": "/root/autodl-tmp/model_merged_qwen",
            "R1-LoRA (baseline)": "/root/autodl-tmp/model_merged_r1",
            "Qwen-Raw (baseline)": "/root/autodl-tmp/Qwen2.5-1.5B-Instruct",
            "R1-Raw (baseline)": "/root/autodl-tmp/DeepSeek-R1-1.5B",
        }

        _gat_default_path = "/root/autodl-tmp/gat_model_v2_gated.pt"
        import os as _os
        _gat_trained = _os.path.exists(_gat_default_path)
        use_gat = False
        gat_alpha, gat_model_path = 0.85, _gat_default_path
        if run_mode != "experiment_mode":
            st.session_state["use_gat"] = False

        if defense_mode and not show_debug_details:
            st.caption("Defense Mode 已隐藏 exploratory model 切换、patch / GAT 试验入口和低层调试项。")
        else:
            with st.expander("🧪 高级选项（baseline / exploratory only）", expanded=False):
                st.caption("默认主链路始终为 Sparse-LoRA-v2。以下入口仅用于 appendix / 实验复核，不用于答辩主演示。")
                if run_mode == "experiment_mode":
                    selected_model = st.selectbox(
                        "扩展模型选择",
                        [
                            MAINLINE_METHOD_NAME,
                            "Qwen-LoRA (baseline)",
                            "R1-LoRA (baseline)",
                            "Qwen-Raw (baseline)",
                            "R1-Raw (baseline)",
                        ],
                        key="model_select",
                        help="Sparse-LoRA-v2 为默认主方法；其余模型仅作 baseline 对照。",
                    )
                    st.caption("patch SFT：negative results / exploratory only, not for defense main demo.")
                    st.caption("PPO：baseline only。当前 live planner UI 不提供 PPO 路径规划切换。")
                    st.markdown("**GAT（exploratory only, not mainline）**")
                    use_gat = st.toggle(
                        "启用 GAT 增强（exploratory only, not mainline）",
                        value=False,
                        key="use_gat",
                        help="开启后：LLM解析结果 → GAT图神经网络精炼 → A*规划。该入口仅用于附录/探索性分析，不进入默认主链路。",
                    )
                    if use_gat:
                        if not _gat_trained:
                            st.warning("⚠️ GAT尚未训练，开启会使用随机权重污染结果！\n请先运行：`python evaluate.py --train-gat`", icon="🚨")
                        gat_alpha = st.slider(
                            "混合系数 α（LLM权重占比）",
                            0.5, 1.0, 0.85, 0.05,
                            key="gat_alpha",
                            help="α=1: 纯LLM，α=0: 纯GAT。当前 UI 仅把它保留为 exploratory only 入口。",
                        )
                        gat_model_path = st.text_input(
                            "GAT模型路径",
                            value=_gat_default_path,
                            key="gat_path",
                        )
                    if _gat_trained:
                        st.caption("✅ GAT模型已就绪（exploratory / appendix-only）")
                    else:
                        st.caption("⚪ GAT未训练（exploratory / appendix-only）")
                else:
                    st.caption("切换 baseline / exploratory 模型与 GAT 开关需进入 Experiment Mode。")
                    st.caption("patch SFT：negative results / exploratory only, not for defense main demo.")
                    st.caption("PPO：baseline only。")
                    st.caption("GAT：exploratory only, not mainline。")

        config.config["MODEL_PATH"] = _model_path_map.get(selected_model, MAINLINE_MODEL_PATH)
        _is_sparse_ui = "Sparse" in selected_model

        st.markdown('<div class="sec-hdr">📝 场景输入</div>', unsafe_allow_html=True)
        traffic_constraint = st.text_area(
            "交通约束描述",
            placeholder="黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北施工封闭；红专路向东畅通无阻。",
            height=120, key="traffic_input",
        )

        col_s, col_e = st.columns(2)
        with col_s:
            start_edge = st.text_input("🟢 起点", value="R3C0_E", key="start_edge")
        with col_e:
            end_edge = st.text_input("🏁 终点", value="R3C3_E", key="end_edge")

        # 共享场景标签与快捷测试场景
        scenarios = get_ui_scenarios()
        scene_profile_options = list(list_scene_profiles())
        production_showcase_names = tuple(PRODUCTION_SHOWCASE_NAMES)
        calibration_showcase_specs = list_calibration_showcase_presets()
        structured_showcase_names = production_showcase_names + tuple(calibration_showcase_specs.keys())
        showcase_cases = select_showcase_scenarios(
            config.get("SUMO_NET_PATH"),
            structured_showcase_names,
        )
        production_showcase_cases = {
            name: showcase_cases[name]
            for name in production_showcase_names
            if name in showcase_cases
        }
        experimental_neutral_cases = {
            name: item
            for name, item in showcase_cases.items()
            if str(item.get("showcase_group") or "") == "experimental_neutral"
        }
        calibration_only_cases = {
            name: item
            for name, item in showcase_cases.items()
            if str(item.get("showcase_group") or "") == "calibration_debug"
        }
        best_showcase_case = select_best_showcase_scenario(
            config.get("SUMO_NET_PATH"),
            production_showcase_names,
        )
        best_showcase_name = str((best_showcase_case or {}).get("showcase_name") or "")
        production_default_showcase_name = (
            DEFAULT_PRODUCTION_SHOWCASE_NAME
            if DEFAULT_PRODUCTION_SHOWCASE_NAME in production_showcase_cases
            else best_showcase_name or next(iter(production_showcase_cases.keys()), "— 不使用 —")
        )
        default_scene_profile_name = config.get("DEFAULT_SCENE_PROFILE", "normal_baseline")
        default_manual_scene_profile = st.session_state.get(
            "manual_scene_profile_name",
            default_scene_profile_name,
        )
        if default_manual_scene_profile not in scene_profile_options:
            default_manual_scene_profile = default_scene_profile_name
            st.session_state["manual_scene_profile_name"] = default_manual_scene_profile
        scene_profile_binding_mode = st.session_state.get("scene_profile_binding_mode", "Auto Detect")
        manual_scene_profile_name = default_manual_scene_profile
        selected_showcase_name = "— 不使用 —"
        if "selected_showcase_name" not in st.session_state and workspace_scene_mode == "showcase":
            st.session_state["selected_showcase_name"] = production_default_showcase_name
        if workspace_scene_mode == "showcase" and (production_showcase_cases or experimental_neutral_cases or calibration_only_cases):
            current_showcase_selection = str(
                st.session_state.get("selected_showcase_name")
                or production_default_showcase_name
                or "— 不使用 —"
            )
            if current_showcase_selection not in showcase_cases and current_showcase_selection != "— 不使用 —":
                current_showcase_selection = production_default_showcase_name
                st.session_state["selected_showcase_name"] = current_showcase_selection
        with st.expander("🎯 Production Showcase", expanded=True):
            if workspace_scene_mode != "showcase":
                st.info("当前处于 Benchmark Mode：showcase 自动选点已暂停，不会覆盖 benchmark/raw scene 输入。")
            elif not production_showcase_cases:
                st.warning("当前未生成可用的 production showcase 场景。")
            else:
                if st.button("恢复默认 Production Showcase", use_container_width=True):
                    st.session_state["selected_showcase_name"] = production_default_showcase_name
                    st.session_state["production_showcase_selector"] = production_default_showcase_name
                production_selector_options = list(production_showcase_cases.keys())
                current_production_selector = (
                    current_showcase_selection
                    if current_showcase_selection in production_showcase_cases
                    else production_default_showcase_name
                )
                if "production_showcase_selector" not in st.session_state:
                    st.session_state["production_showcase_selector"] = current_production_selector
                if str(st.session_state.get("production_showcase_selector") or "") not in production_selector_options:
                    st.session_state["production_showcase_selector"] = current_production_selector
                selected_production_name = st.selectbox(
                    "Production Showcase 列表",
                    production_selector_options,
                    index=production_selector_options.index(
                        str(st.session_state.get("production_showcase_selector") or current_production_selector)
                    ),
                    key="production_showcase_selector",
                    format_func=lambda name: (
                        f"{production_showcase_cases[name].get('showcase_badge') or 'Positive Gain Showcase'} · "
                        f"{production_showcase_cases[name]['demo_label']} · "
                        f"节省时间 {float(production_showcase_cases[name].get('saving_seconds') or 0.0):.1f}s · "
                        f"节省比例 {float(production_showcase_cases[name].get('saving_ratio') or 0.0) * 100.0:.1f}% · "
                        f"{production_showcase_cases[name]['start_edge']} → {production_showcase_cases[name]['end_edge']}"
                    ),
                    help="Production Showcase 只包含已冻结的正收益案例。",
                )
                if st.button("应用当前 Production Showcase", use_container_width=True):
                    st.session_state["selected_showcase_name"] = selected_production_name
                if production_default_showcase_name:
                    st.caption(
                        f"默认 production showcase：{production_default_showcase_name} | "
                        f"Positive Gain Showcase | "
                        f"节省时间 {float((production_showcase_cases.get(production_default_showcase_name) or {}).get('saving_seconds') or 0.0):.1f}s | "
                        f"节省比例 {float((production_showcase_cases.get(production_default_showcase_name) or {}).get('saving_ratio') or 0.0) * 100.0:.1f}% | "
                        f"{(production_showcase_cases.get(production_default_showcase_name) or {}).get('start_edge')} → {(production_showcase_cases.get(production_default_showcase_name) or {}).get('end_edge')}"
                    )
                for showcase_name, item in production_showcase_cases.items():
                    st.markdown(
                        f"""
                        <div class="info-strip">
                        <b>{item.get('showcase_badge') or 'Positive Gain Showcase'}</b><br>
                        <b>{item['demo_label']}</b>：{item['start_edge']} → {item['end_edge']}<br>
                        <b>showcase_name</b>：{showcase_name}<br>
                        <b>推荐理由</b>：{item['display_reason']}<br>
                        <b>节省时间</b>：{float(item.get('saving_seconds') or 0.0):.1f} s<br>
                        <b>节省比例</b>：{float(item.get('saving_ratio') or 0.0) * 100.0:.1f}%<br>
                        <b>actual baseline / planned</b>：{float(item.get('baseline_time_s') or 0.0):.1f}s / {float(item.get('planned_time_s') or 0.0):.1f}s<br>
                        <b>affected_edges</b>：{", ".join(list(item.get("affected_edges") or [])[:6]) or "none"}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        with st.expander("🧪 Experimental / Neutral Cases", expanded=False):
            if workspace_scene_mode != "showcase":
                st.info("当前处于 Benchmark Mode：experimental / neutral showcase 不会覆盖 benchmark/raw scene 输入。")
            elif not experimental_neutral_cases:
                st.caption("当前没有可用的 neutral diagnostic showcase。")
            else:
                st.caption("这里仅保留中性诊断案例，不占默认 production 展示位。")
                for showcase_name, item in experimental_neutral_cases.items():
                    st.markdown(
                        f"""
                        <div class="info-strip">
                        <b>{item.get('showcase_badge', 'Neutral Diagnostic Case')}</b><br>
                        <b>{item.get('demo_label', showcase_name)}</b>：{item.get('start_edge', 'unknown')} → {item.get('end_edge', 'unknown')}<br>
                        <b>showcase_name</b>：{showcase_name}<br>
                        <b>状态</b>：当前规划与基线持平<br>
                        <b>说明</b>：{item.get('status_note', '但路径与脆弱走廊关系可解释。')}<br>
                        <b>actual baseline / planned</b>：{float(item.get('baseline_time_s') or 0.0):.1f}s / {float(item.get('planned_time_s') or 0.0):.1f}s
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"应用 {showcase_name}",
                        key=f"experimental_showcase_{showcase_name}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_showcase_name"] = showcase_name
        with st.expander("🛠️ Calibration / Hidden / Debug", expanded=False):
            if workspace_scene_mode != "showcase":
                st.info("当前处于 Benchmark Mode：calibration/debug showcase 不会覆盖 benchmark/raw scene 输入。")
            elif not calibration_only_cases:
                st.caption("当前没有 calibration-only showcase。")
            else:
                st.caption("这些入口只保留给 calibration / hidden / debug，不进入 production showcase 首页。")
                for showcase_name, item in calibration_only_cases.items():
                    st.markdown(
                        f"""
                        <div class="info-strip">
                        <b>{item.get('showcase_badge', 'Calibration Only')}</b><br>
                        <b>{item.get('demo_label', showcase_name)}</b>：{item.get('start_edge', 'unknown')} → {item.get('end_edge', 'unknown')}<br>
                        <b>showcase_name</b>：{showcase_name}<br>
                        <b>status</b>：{item.get('status_note', 'Calibration only')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"应用 {showcase_name}",
                        key=f"calibration_showcase_{showcase_name}",
                        use_container_width=True,
                    ):
                        st.session_state["selected_showcase_name"] = showcase_name
        selected_showcase_name = str(st.session_state.get("selected_showcase_name") or "— 不使用 —")
        with st.expander("⚡ 快捷测试场景"):
            sel = st.selectbox("选择场景", ["— 自定义 —"] + list(scenarios.keys()), key="preset_sel")
            if sel != "— 自定义 —":
                _s, _e, _con, _scene_type, _scene_profile_name = scenarios[sel]
                st.caption(
                    f"📍 起: {_s}  →  终: {_e} | 场景标签: {_scene_type} | 环境profile: {_scene_profile_name}"
                )
                st.info(_con[:120] + "…", icon="📝")
                st.caption(
                    "当前为显式预设：scene_profile 将锁定到该 preset，不再按文本内容重判。"
                )
            else:
                scene_profile_binding_mode = st.radio(
                    "Scene Profile Binding",
                    ["Auto Detect", "Manual Scene Profile"],
                    horizontal=True,
                    key="scene_profile_binding_mode",
                )
                if scene_profile_binding_mode == "Manual Scene Profile":
                    manual_scene_profile_name = st.selectbox(
                        "Manual Scene Profile",
                        scene_profile_options,
                        index=scene_profile_options.index(default_manual_scene_profile),
                        key="manual_scene_profile_name",
                    )
                    _manual_profile = get_scene_profile(manual_scene_profile_name)
                    st.caption(
                        f"手动锁定：scene_profile={_manual_profile.scene_name} | "
                        f"scene_type={_manual_profile.scene_type}"
                    )
                    st.info(
                        "Manual Scene Profile 模式下，scene_profile 将以你的选择为准，不再按文本内容重判。",
                        icon="🧭",
                    )
                else:
                    st.caption(
                        "Auto Detect：仅在自定义文本下按语义推断 scene_type / scene_profile；"
                        "包含“整体通行正常 / 无明显拥堵 / 无事故”等描述时将识别为 normal_baseline。"
                    )

        st.markdown('<div class="sec-hdr">⚙️ 规划算法</div>', unsafe_allow_html=True)
        if defense_mode and not show_debug_details:
            path_algo = st.session_state.get("path_algo", "A*（统一代价）")
            st.caption(f"Defense Mode 中隐藏低层算法切换；当前规划算法固定显示为：{path_algo}")
        else:
            path_algo = st.selectbox(
                "路径规划算法",
                ["A*（统一代价）", "Dijkstra", "Bellman-Ford"],
                key="path_algo",
            )

        run_button = st.button("🚀 一键运行规划", type="primary", use_container_width=True)

        # 快捷场景选中后覆盖运行参数
        scene_profile_locked = False
        scene_profile_selection_mode = "auto_detect"
        active_showcase_case = None
        if workspace_scene_mode == "showcase" and selected_showcase_name != "— 不使用 —":
            active_showcase_case = dict(showcase_cases.get(selected_showcase_name) or {})
            _recommended_profile = get_scene_profile(str(active_showcase_case.get("scene_profile") or "normal_baseline"))
            start_edge = str(active_showcase_case.get("start_edge") or start_edge)
            end_edge = str(active_showcase_case.get("end_edge") or end_edge)
            traffic_constraint = (
                active_showcase_case.get("display_reason")
                or getattr(_recommended_profile, "description", "")
                or traffic_constraint
            )
            _scene_type = _recommended_profile.scene_type
            _scene_profile_name = _recommended_profile.scene_name
            scene_profile_locked = True
            scene_profile_selection_mode = "structured_showcase"
            st.caption(
                f"当前使用 Showcase Mode 结构化案例：{active_showcase_case.get('showcase_name')} | "
                f"{start_edge} → {end_edge} | scene_profile={_scene_profile_name}"
            )
            st.caption("本次 Showcase 运行将直接使用 free-flow baseline + scene-aware shortest path，文本描述仅作展示，不参与真正路径生成。")
        elif sel != "— 自定义 —":
            _s, _e, _con, _scene_type, _scene_profile_name = scenarios[sel]
            start_edge = _s
            end_edge = _e
            traffic_constraint = _con
            scene_profile_locked = True
            scene_profile_selection_mode = "preset_locked"
        elif scene_profile_binding_mode == "Manual Scene Profile":
            _manual_profile = get_scene_profile(manual_scene_profile_name)
            _scene_type = _manual_profile.scene_type
            _scene_profile_name = _manual_profile.scene_name
            scene_profile_locked = True
            scene_profile_selection_mode = "manual_scene_profile"
        else:
            _scene_type = classify_scene_type(text=traffic_constraint)
            _scene_profile_name = resolve_scene_profile_name(
                scene_type=_scene_type,
                default_name=config.get("DEFAULT_SCENE_PROFILE", "normal_baseline"),
            )
            scene_profile_selection_mode = "auto_detect"
        scene_profile = resolve_scene_profile(
            scene_name=_scene_profile_name,
            scene_type=_scene_type,
            default_name=config.get("DEFAULT_SCENE_PROFILE", "normal_baseline"),
        )
        _scene_type = str(getattr(scene_profile, "scene_type", _scene_type) or _scene_type)
        _scene_profile_name = scene_profile.scene_name
        setattr(path_engine, "_last_scene_type", _scene_type)
        setattr(path_engine, "_last_scene_profile_name", _scene_profile_name)
        scene_mode = workspace_scene_mode
        if active_showcase_case:
            scene_weight_layers_preview = dict(active_showcase_case.get("scene_weight_layers") or {})
            if not scene_weight_layers_preview:
                scene_weight_layers_preview = build_environment_weight_layers(
                    LIVE_EDGE_IDS,
                    scene_profile,
                    enable_spillover=True,
                )
        else:
            scene_weight_layers_preview = build_environment_weight_layers(
                LIVE_EDGE_IDS,
                scene_profile,
                enable_spillover=(scene_mode == "showcase"),
            )
        scene_env_payload = _build_scene_env_payload(
            scene_profile,
            scene_weight_layers_preview,
            scene_mode=scene_mode,
        )
        current_request_signature = _build_request_signature(
            scene_profile_name=scene_env_payload["scene_name"],
            scene_type=scene_env_payload["scene_type"],
            workspace_scene_mode=scene_mode,
            start_edge=start_edge,
            end_edge=end_edge,
            traffic_constraint=traffic_constraint,
            path_algo=path_algo,
            selected_model=selected_model,
            use_gat=st.session_state.get("use_gat", False),
            gat_alpha=st.session_state.get("gat_alpha", 0.85),
            use_manual=st.session_state.get("use_manual", False),
            showcase_case_name=str((active_showcase_case or {}).get("showcase_name") or ""),
        )
        st.session_state["active_request_signature"] = current_request_signature
        result_context_dirty = bool(st.session_state.get("results")) and (
            current_request_signature != st.session_state.get("last_result_signature", "")
        )

        with st.expander("🧭 场景与环境配置", expanded=not defense_mode):
            st.markdown(
                """
                <div class="info-strip">
                <b>说明</b>：这些环境参数由场景配置固定控制，不直接受模型理解影响；模型只影响边权估计和路径选择。
                </div>
                """,
                unsafe_allow_html=True,
            )
            if scene_profile_locked:
                if scene_profile_selection_mode == "structured_showcase":
                    _binding_text = (
                        f"当前 scene_profile 由 Showcase Mode 结构化案例 `{active_showcase_case.get('showcase_name', '')}` 锁定为 "
                        f"`{scene_env_payload['scene_name']}`；起终点、受扰边、baseline/planned path 都直接来自结构化 showcase 数据，不再依赖 parser。"
                    )
                elif scene_profile_selection_mode == "auto_recommended":
                    _binding_text = (
                        f"当前 scene_profile 由自动推荐演示场景锁定为 `{scene_env_payload['scene_name']}`，"
                        "起终点将随推荐案例自动覆盖。"
                    )
                elif scene_profile_selection_mode == "preset_locked":
                    _binding_text = (
                        f"当前 scene_profile 由预设场景锁定为 `{scene_env_payload['scene_name']}`，"
                        "不会再按文本内容重判。"
                    )
                else:
                    _binding_text = (
                        f"当前 scene_profile 由 Manual Scene Profile 锁定为 `{scene_env_payload['scene_name']}`，"
                        "不会再按文本内容重判。"
                    )
            else:
                _binding_text = (
                    "当前 scene_profile 来源：Auto Detect。自由文本会按语义推断；"
                    "包含“整体通行正常 / 无明显拥堵 / 无事故”等描述时将识别为 `normal_baseline`。"
                )
            st.caption(_binding_text)
            s1, s2 = st.columns(2)
            s1.metric("scene_profile", scene_env_payload["scene_name"])
            s2.metric("scene_type", scene_env_payload["scene_type"])
            s3, s4 = st.columns(2)
            s3.metric("tls_profile_name", scene_env_payload["tls_profile_name"])
            s4.metric(
                "reroute",
                scene_env_payload["reroute_effective_status"],
                delta=(
                    f"period={scene_env_payload['reroute_period']}s · {scene_env_payload.get('reroute_reason', '')}"
                ),
                delta_color="off",
            )
            s5, s6 = st.columns(2)
            s5.metric("simulation_seed", str(scene_env_payload["simulation_seed"]))
            s6.metric("Warmup", _format_seconds_text(scene_env_payload.get("warmup_seconds"), precision=0))
            st.caption(
                f"Evaluation Window: {_format_time_window_text(scene_env_payload.get('evaluation_start_time'), scene_env_payload.get('evaluation_end_time'))}"
            )
            st.caption(f"Current Scene Time: {_format_seconds_text(scene_env_payload.get('current_scene_time_s'))}")
            st.caption(f"spillover summary: {scene_env_payload.get('spillover_summary', 'off')}")
            if defense_mode and not show_debug_details:
                st.markdown(
                    f"""
                    <div class="info-strip">
                    <b>场景环境摘要</b>：{_scene_env_summary_text(scene_env_payload)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("**speed_profile**")
                _safe_table(pd.DataFrame(scene_env_payload["speed_profile_rows"]))
                st.markdown("**incident / blocked / lane reduction 摘要**")
                st.markdown(
                    f"""
                    <div class="info-strip">
                    <b>incident</b>：{scene_env_payload['incident_summary']}<br>
                    <b>blocked</b>：{scene_env_payload['blocked_summary']}<br>
                    <b>lane reduction</b>：{scene_env_payload['lane_reduction_summary']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if result_context_dirty:
                st.warning("当前场景或输入已变化，上一轮缓存结果已停止作为当前场景展示。请重新运行以刷新结果。")
            else:
                st.caption("当前场景面板与当前输入参数同步；重新运行后结果区将写入同一场景上下文。")

        if defense_mode and not show_debug_details or display_mode == "答辩简洁模式":
            manual_str = st.session_state.get("manual_weights", "")
            use_manual = False
        else:
            with st.expander("🔧 手动权重（调试用）"):
                manual_str = st.text_area(
                    "权重（edge:w, …）",
                    value="R0C0_E:8.5, R0C1_E:9.2, C2R0_N:2.0, R3C0_E:6.0, C4R1_N:1.5, "
                          "R1C2_E:3.5, C0R2_N:7.0, R2C3_W:4.0",
                    height=80, key="manual_weights",
                )
                use_manual = st.checkbox("✅ 启用手动权重", key="use_manual")

        # 修正 6：模型原始输出栏：始终可见，未运行时显示"尚未运行"
        _short = selected_model.split("(")[0].strip()
        if display_mode != "答辩简洁模式" or show_debug_details:
            st.markdown(f'<div class="sec-hdr">🔍 模型原始输出 · {_short}</div>', unsafe_allow_html=True)
            with st.expander("查看原始输出", expanded=False):
                raw_wd = st.session_state.get("last_raw_llm_weights", {})
                raw = st.session_state.get("last_raw", "")
                last_mode = st.session_state.get("last_mode", config.get("RUN_MODE", "demo_mode"))
                last_fb_used = st.session_state.get("last_fallback_used", False)
                last_fb_reason = st.session_state.get("last_fallback_reason", "none")
                if result_context_dirty:
                    st.info("当前场景或输入已切换，上一轮缓存原始输出已隐藏。重新运行后这里会刷新为当前场景结果。")
                elif raw:
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

        if st.session_state["run_history"] and (display_mode != "答辩简洁模式" or show_debug_details):
            st.markdown('<div class="sec-hdr">📋 规划历史</div>', unsafe_allow_html=True)
            if result_context_dirty:
                st.caption("已切换场景或输入；以下历史记录属于之前的运行批次，仅作参考。")
            for h in reversed(st.session_state["run_history"][-5:]):
                st.markdown(f"""<div class="hist-row">
                    <span>#{h["id"]}  {h["start"]}→{h["end"]}</span>
                    <span style="color:#1D4ED8;font-weight:600">{h["travel_time"]:.1f}s</span>
                </div>""", unsafe_allow_html=True)
            if st.button("🗑 清除历史", use_container_width=True):
                st.session_state["run_history"] = []
                st.session_state["run_count"]   = 0

        export_disabled = (not bool(st.session_state.get("results"))) or result_context_dirty
        if st.button("📥 导出结果 JSON", use_container_width=True, disabled=export_disabled):
            if st.session_state["results"] and not result_context_dirty:
                fn = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                fp = export_engine.export_results(st.session_state["results"], fn)
                with open(fp) as f:
                    st.download_button("⬇️ 下载 JSON", f.read(), fn)
            else:
                st.warning("暂无结果")
        if result_context_dirty:
            st.caption("导出已临时禁用：当前场景或输入与上一轮结果不一致，需重新运行后再导出。")

        st.divider()
        st.caption(f"💡 路段ID: R{{行}}C{{列}}_{{E/W}}  C{{列}}R{{行}}_{{N/S}}  共{edge_total}条\n"
                   "农业路=R0, 黄河路=R3, 花园路=C4, 经六路=C2")

    _mode_badge = "Demo Mode（允许规则兜底）" if config.get("RUN_MODE", "demo_mode") == "demo_mode" else "Experiment Mode（禁用规则兜底）"
    st.markdown(f'<div class="info-strip">🧪 当前运行模式：<b>{_mode_badge}</b></div>', unsafe_allow_html=True)
    if defense_mode:
        st.markdown(
            '<div class="info-strip"><b>Defense Mode</b> 已启用：主界面仅保留答辩关键结论。需要更多字段时，点击 <b>Show Debug Details</b>。</div>',
            unsafe_allow_html=True,
        )
    _render_scope_banner(
        "Showcase vs Benchmark",
        [
            "Showcase 用于答辩展示当前冻结案例的地图、路径与收益结果。",
            "Benchmark 区固定读取 frozen main table / ablation / appendix artifacts，不被 showcase 页面覆盖。",
        ],
    )

    # 主内容区
    compare_ph = None
    adv_ph = None
    if defense_mode:
        if show_debug_details:
            tab1, tab2, tab_debug = st.tabs([
                "🎬 Showcase / Map & Summary", "🎬 Showcase / Metrics & Environment", "🔎 Debug / Appendix Details"
            ])
        else:
            tab1, tab2 = st.tabs([
                "🎬 Showcase / Map & Summary", "🎬 Showcase / Metrics & Environment"
            ])

        with tab1:
            st.markdown('<div class="sec-hdr">🛡️ Showcase Summary</div>', unsafe_allow_html=True)
            weight_ph = st.empty()
            path_ph = st.empty()
            metric_ph = st.empty()

        with tab2:
            st.markdown('<div class="sec-hdr">⏱️ Metrics & Environment</div>', unsafe_allow_html=True)
            st.caption("Showcase metrics and runtime summaries are colocated below the full-width map in the first tab.")

        if show_debug_details:
            with tab_debug:
                st.markdown('<div class="sec-hdr">🔎 Debug Details</div>', unsafe_allow_html=True)
                compare_ph = st.empty()
                adv_ph = st.empty()
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎬 Showcase / Map & Summary", "🎬 Showcase / Metrics & Environment", "📋 Benchmark / Main Table & Ablation", "🔎 Appendix / Advanced", "🗺️ Showcase / Network Overview"
        ])

        with tab5:
            st.markdown('<div class="sec-hdr">🗺️ 郑州金水区路网拓扑（真实底图 + 近似规整道路骨架）</div>',
                        unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("有向边总数", f"{edge_total} 条")
            c2.metric("交叉路口",   "30 个")
            c3.metric("路网面积",   "3.3×2.5 km")
            c4.metric("道路等级",   "3 级")
            try:
                net_preview, ec_preview = _load_net_cached(config.get("SUMO_NET_PATH"))
                wd_preview = st.session_state.get("last_wd") or {}
                scene_preview = st.session_state.get("last_scene_weights") or {}
                path_preview = st.session_state.get("last_path")
                base_preview = st.session_state.get("last_baseline_path") or []
                map_preview_weights = scene_preview or wd_preview
                preview_plan_time = (
                    _estimate_route_time_on_weights(net_preview, path_preview, map_preview_weights)
                    if path_preview else None
                )
                preview_base_time = (
                    _estimate_route_time_on_weights(net_preview, base_preview, map_preview_weights)
                    if base_preview else None
                )
                _render_dual_map_view(
                    selector_label="路网显示模式",
                    view_key="roadnet_overview_map_view",
                    folium_builder=lambda style, layer_visibility=None, basemap_style="nolabel_light": viz_engine.build_folium_network_map(
                        weight_dict=map_preview_weights or None,
                        raw_incident_weights=st.session_state.get("last_scene_raw_weights") or None,
                        spillover_weights=st.session_state.get("last_scene_spillover_weights") or None,
                        spillover_hops=(st.session_state.get("last_scene_weight_layers") or {}).get("spillover_hop_by_edge"),
                        gat_smoothed_weights=(st.session_state.get("last_wd") if st.session_state.get("gat_applied") else None),
                        overlay_stats=(st.session_state.get("last_scene_weight_layers") or {}).get("stats"),
                        path=path_preview,
                        focus_path=False,
                        baseline_path=base_preview,
                        planned_time_s=preview_plan_time,
                        baseline_time_s=preview_base_time,
                        map_style=style,
                        basemap_style=basemap_style,
                        layer_visibility=layer_visibility,
                    ),
                    plotly_builder=lambda style: viz_engine.plot_network_overview(
                        ec_preview,
                        map_preview_weights or None,
                        path_preview,
                        baseline_path=base_preview,
                        map_style=style,
                    ),
                    folium_component_key="roadnet_overview_folium",
                    folium_height=560,
                    plotly_height=560,
                    html_export_name="roadnet_overview_latest.html",
                    download_label="下载路网 HTML 地图",
                    default_layer_overrides={"gat_smoothed_layer": bool(st.session_state.get("gat_applied"))},
                )
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

        with tab1:
            st.markdown('<div class="sec-hdr">📊 Showcase Summary</div>', unsafe_allow_html=True)
            weight_ph = st.empty()
            path_ph = st.empty()
            metric_ph = st.empty()

        with tab2:
            st.markdown('<div class="sec-hdr">⏱️ Metrics & Environment</div>', unsafe_allow_html=True)
            st.caption("Showcase metrics and runtime summaries are colocated below the full-width map in the first tab.")

        with tab3:
            st.markdown('<div class="sec-hdr">📋 Frozen Main Table & Ablation</div>', unsafe_allow_html=True)
            compare_ph = st.empty()

        with tab4:
            st.markdown('<div class="sec-hdr">🔎 Appendix & Advanced Analysis</div>', unsafe_allow_html=True)
            adv_ph = st.empty()

    # 运行逻辑
    cached_display_ready = bool(st.session_state.get("results")) and not result_context_dirty
    if not run_button:
        if cached_display_ready:
            try:
                cached_record = dict(st.session_state.get("results") or {})
                cached_time_fields = normalize_eval_time_fields(cached_record)
                _, cached_edge_coords = _load_net_cached(config.get("SUMO_NET_PATH"))
                cached_weight_dict = dict(
                    cached_record.get("final_planning_weights")
                    or cached_record.get("weights")
                    or st.session_state.get("last_wd")
                    or {}
                )
                cached_path = list(cached_record.get("path") or st.session_state.get("last_path") or [])
                cached_total_cost = _safe_float(
                    cached_record.get("total_cost"),
                    st.session_state.get("total_cost", 0.0),
                )
                cached_travel_time = _safe_float(
                    cached_time_fields.get("travel_time_s"),
                    st.session_state.get("travel_time", 0.0),
                )
                cached_avg_speed = _safe_float(cached_record.get("avg_speed"), 0.0)
                cached_speed_data = list(st.session_state.get("last_speed_data") or [])
                cached_scene_weights = dict(st.session_state.get("last_scene_weights") or {})
                cached_baseline_path = list(
                    cached_record.get("baseline_path")
                    or st.session_state.get("last_baseline_path")
                    or []
                )
                cached_comply_rate = _safe_float(st.session_state.get("last_comply_rate"), 0.0)
                cached_algo = str(
                    cached_record.get("path_algo") or st.session_state.get("path_algo", "A*（统一代价）")
                )
                cached_display_payload = _build_result_display_payload(
                    cached_record,
                    edge_total=edge_total,
                    path_algo=cached_algo,
                    scene_weights=cached_scene_weights,
                )
                cached_severe = sum(1 for w in cached_weight_dict.values() if w >= 6.0)
                cached_medium = sum(1 for w in cached_weight_dict.values() if 3.0 <= w < 6.0)
                cached_clear = sum(1 for w in cached_weight_dict.values() if w < 3.0)

                st.info("已复用当前缓存结果，仅切换展示视图，不重复计算模型 / planner / SUMO。")
                _render_main_result_panels(
                    weight_ph=weight_ph,
                    path_ph=path_ph,
                    metric_ph=metric_ph,
                    display_payload=cached_display_payload,
                    display_mode=display_mode,
                    edge_total=edge_total,
                    weight_dict=cached_weight_dict,
                    path=cached_path,
                    edge_coords=cached_edge_coords,
                    severe=cached_severe,
                    medium=cached_medium,
                    clear=cached_clear,
                    travel_time=cached_travel_time,
                    avg_speed=cached_avg_speed,
                    comply_rate=cached_comply_rate,
                    baseline_path_ref=cached_baseline_path,
                    scene_weights=cached_scene_weights,
                    speed_data=cached_speed_data,
                    defense_mode=defense_mode,
                    show_debug_details=show_debug_details,
                    scene_env_payload=scene_env_payload,
                )
                if compare_ph is not None:
                    compare_ph.empty()
                    with compare_ph.container():
                        st.info("已复用缓存结果；冻结主表/消融表可直接查看，exploratory 图若需刷新可重新运行实验。")
                        _render_frozen_tables_workspace(
                            total_cost_value=float(cached_total_cost),
                            weight_dict_value=dict(cached_weight_dict or {}),
                            planning_time_value=float(
                                cached_time_fields.get("planning_time_s")
                                or cached_time_fields.get("infer_time_s")
                                or 0.0
                            ),
                            debug=show_debug_details,
                        )
                if adv_ph is not None:
                    adv_ph.empty()
                    with adv_ph.container():
                        st.info("已复用缓存结果；高级图表与地图切换不会重复计算核心模型逻辑。")
            except Exception as cache_exc:
                st.error(f"❌ 缓存结果恢复失败：{cache_exc}")
                st.code(traceback.format_exc())
        elif active_showcase_case:
            try:
                preview_record = _build_showcase_preview_record(
                    active_showcase_case,
                    scene_profile,
                    scene_mode=scene_mode,
                )
                preview_display_payload = _build_result_display_payload(
                    preview_record,
                    edge_total=edge_total,
                    path_algo=str(preview_record.get("path_algo") or "Showcase / Scene-aware Shortest Path"),
                    scene_weights=dict(active_showcase_case.get("scene_weights") or {}),
                )
                _, preview_edge_coords = _load_net_cached(config.get("SUMO_NET_PATH"))
                preview_weight_dict = dict(active_showcase_case.get("scene_weights") or {})
                preview_path = list(active_showcase_case.get("planned_path") or [])
                preview_baseline_path = list(active_showcase_case.get("baseline_path") or [])
                preview_severe = sum(1 for w in preview_weight_dict.values() if w >= 6.0)
                preview_medium = sum(1 for w in preview_weight_dict.values() if 3.0 <= w < 6.0)
                preview_clear = sum(1 for w in preview_weight_dict.values() if w < 3.0)
                _render_main_result_panels(
                    weight_ph=weight_ph,
                    path_ph=path_ph,
                    metric_ph=metric_ph,
                    display_payload=preview_display_payload,
                    display_mode=display_mode,
                    edge_total=edge_total,
                    weight_dict=preview_weight_dict,
                    path=preview_path,
                    edge_coords=preview_edge_coords,
                    severe=preview_severe,
                    medium=preview_medium,
                    clear=preview_clear,
                    travel_time=_safe_float(active_showcase_case.get("planned_time_s"), 0.0),
                    avg_speed=0.0,
                    comply_rate=100.0,
                    baseline_path_ref=preview_baseline_path,
                    scene_weights=preview_weight_dict,
                    speed_data=[],
                    defense_mode=defense_mode,
                    show_debug_details=show_debug_details,
                    scene_env_payload=scene_env_payload,
                )
                st.info("已加载默认 showcase 预览；如需重新执行 live planner / SUMO，可点击 `一键运行规划`。")
                if compare_ph is not None:
                    with compare_ph.container():
                        _render_frozen_tables_workspace(
                            total_cost_value=float(st.session_state.get("total_cost", 0.0) or 0.0),
                            weight_dict_value=dict(st.session_state.get("last_wd") or preview_weight_dict or {}),
                            planning_time_value=0.0,
                            debug=show_debug_details,
                        )
            except Exception as preview_exc:
                st.warning(f"默认 showcase 预览加载失败：{preview_exc}")
        elif compare_ph is not None:
            with compare_ph.container():
                st.info("当前尚未运行规划；冻结主表、消融表和附录表已可直接查看。")
                _render_frozen_tables_workspace(
                    total_cost_value=float(st.session_state.get("total_cost", 0.0) or 0.0),
                    weight_dict_value=dict(st.session_state.get("last_wd") or {}),
                    planning_time_value=0.0,
                    debug=show_debug_details,
                )
        return

    showcase_direct_mode = bool(active_showcase_case) and scene_mode == "showcase" and not (
        st.session_state.get("use_manual") and manual_str.strip()
    )

    if (not showcase_direct_mode) and not traffic_constraint.strip():
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
        status.text("🔄 正在准备规划输入..."); prog.progress(15)
        _effective_scene_type = str(getattr(scene_profile, "scene_type", _scene_type) or _scene_type)
        setattr(path_engine, "_last_scene_type", _effective_scene_type)
        setattr(path_engine, "_last_scene_profile_name", scene_profile.scene_name)

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
            _sparse_parse = {
                "scene_type": _scene_type,
                "anchors": [],
                "anchor_count": 0,
                "mapped_weights": dict(raw_llm_weights),
                "edge_confidence": {},
                "parse_confidence": 0.0,
                "conflicts": [],
                "protected_edges": [],
            }
            _gat_meta = {
                "gat_mode": "disabled",
                "high_conf_edges": [],
                "low_conf_edges": [],
                "missing_edges": [],
                "protected_edges": [],
            }
            _timing_info = {
                "init_ms": 0.0,
                "infer_ms": 0.0,
                "smooth_ms": 0.0,
                "total_ms": 0.0,
                "cold_start": False,
                "scene_type": _scene_type,
                "anchor_count": 0,
                "parse_confidence": 0.0,
                "gat_mode": "disabled",
                "high_conf_edges": [],
                "low_conf_edges": [],
                "missing_edges": [],
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
            st.session_state["last_sparse_parse"] = _sparse_parse
            st.session_state["last_gat_meta"] = _gat_meta
        elif showcase_direct_mode:
            struct_output = json.dumps(
                {
                    "showcase_name": active_showcase_case.get("showcase_name"),
                    "scene_profile": active_showcase_case.get("scene_profile"),
                    "start_edge": start_edge,
                    "end_edge": end_edge,
                    "affected_edges": list(active_showcase_case.get("affected_edges") or []),
                    "preferred_baseline_type": active_showcase_case.get("preferred_baseline_type"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            raw_llm_weights = {}
            final_planning_weights = dict(active_showcase_case.get("scene_weights") or {})
            infer_time = 0.0
            raw_output = "[Showcase structured case] parser bypassed; route generated from structured scene data."
            explicit_weight_dict = {}
            _llm_parsed = 0
            _gat_applied = False
            _gat_alpha_used = 0.0
            _mode_used = config.get("RUN_MODE", "demo_mode")
            _fallback_used = False
            _fallback_reason = "showcase_structured_direct"
            _sparse_parse = {
                "scene_type": _scene_type,
                "anchors": [],
                "anchor_count": 0,
                "mapped_weights": {},
                "edge_confidence": {},
                "parse_confidence": 1.0,
                "conflicts": [],
                "protected_edges": list(active_showcase_case.get("affected_edges") or []),
            }
            _gat_meta = {
                "gat_mode": "disabled",
                "high_conf_edges": [],
                "low_conf_edges": [],
                "missing_edges": [],
                "protected_edges": list(active_showcase_case.get("affected_edges") or []),
            }
            _timing_info = {
                "init_ms": 0.0,
                "infer_ms": 0.0,
                "smooth_ms": 0.0,
                "total_ms": 0.0,
                "cold_start": False,
                "scene_type": _scene_type,
                "anchor_count": 0,
                "parse_confidence": 1.0,
                "gat_mode": "disabled",
                "high_conf_edges": [],
                "low_conf_edges": [],
                "missing_edges": [],
                "showcase_structured_mode": True,
            }
            if not final_planning_weights:
                st.error("❌ Showcase 结构化场景未提供有效 scene weights。")
                prog.empty(); status.empty(); return
            weight_dict = final_planning_weights
            edge_metrics = compute_edge_metrics(
                final_weights=final_planning_weights,
                explicit_weights=explicit_weight_dict,
                all_edges=path_engine._ALL_EDGES,
                method_name="Showcase-Structured",
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
            st.session_state["last_sparse_parse"] = _sparse_parse
            st.session_state["last_gat_meta"] = _gat_meta
            st.info(
                "ℹ️ Showcase Mode 结构化案例已启用：本次直接使用 free-flow baseline + scene-aware shortest path，文本仅用于展示。"
            )
        else:
            # model_path 变化时缓存自动失效
            _cur_model_path = config.config.get("MODEL_PATH", "")
            _gw_result = path_engine.generate_weights(
                traffic_constraint, selected_model + "|" + _cur_model_path, model_manager,
                use_gat=st.session_state.get("use_gat", False),
                gat_model_path=st.session_state.get("gat_path", "/root/autodl-tmp/gat_model_v2_gated.pt"),
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
            _sparse_parse = getattr(path_engine, "_last_sparse_parse", {}) or {}
            _gat_meta = getattr(path_engine, "_last_gat_meta", {}) or {}
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
            st.session_state["last_sparse_parse"] = _sparse_parse
            st.session_state["last_gat_meta"] = _gat_meta
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

            if _is_sparse_mode:
                st.caption(f"🧩 语义锚点 {int(_timing_info.get('anchor_count', 0))} 个 | parse_confidence={float(_timing_info.get('parse_confidence', 0.0)):.2f} | scene_type={_timing_info.get('scene_type', _scene_type)}")

            if _gat_applied:
                st.caption(f"🔢 GAT实际混合系数 α = **{_gat_alpha_used:.2f}**（自适应调整，设定值={st.session_state.get('gat_alpha',0.85):.2f}）")
                st.caption(f"🧠 gate={_gat_meta.get('gat_mode', 'disabled')} | high_conf={len(_gat_meta.get('high_conf_edges', []))} | low_conf={len(_gat_meta.get('low_conf_edges', []))} | missing={len(_gat_meta.get('missing_edges', []))}")

        status.text("🗺️ 正在规划路径..."); prog.progress(35)
        net, edge_coords = _load_net_cached(config.get("SUMO_NET_PATH"))
        _algo = "Showcase / Scene-aware Shortest Path" if showcase_direct_mode else st.session_state.get("path_algo", "A*（统一代价）")
        _route_solve_t0 = time.perf_counter()
        if showcase_direct_mode:
            path = list(active_showcase_case.get("planned_path") or [])
            total_cost = _safe_float(
                (active_showcase_case.get("debug") or {}).get("planned_selection_cost_s"),
                active_showcase_case.get("planned_time_s") or 0.0,
            )
            if not path:
                path, total_cost = path_engine.dijkstra_route(net, start_edge, end_edge, weight_dict)
            st.info("ℹ️ Showcase 结构化场景：planned_path 直接来自 scene-aware shortest path，未经过 parser。")
        elif "Dijkstra" in _algo:
            path, total_cost = path_engine.dijkstra_route(net, start_edge, end_edge, weight_dict)
            st.info("ℹ️ Dijkstra模式：与A*使用相同预计通行时间代价，但不使用启发函数。")
        elif "Bellman-Ford" in _algo:
            path, total_cost = path_engine.bellman_ford_route(net, start_edge, end_edge, weight_dict)
            st.info("ℹ️ Bellman-Ford模式：与A*/Dijkstra使用相同边代价，差异仅在搜索/松弛策略。")
        else:
            path, total_cost = path_engine.astar_route(net, start_edge, end_edge, weight_dict)
        route_solve_time_s = time.perf_counter() - _route_solve_t0
        planning_time_s = float(infer_time or 0.0) + float(route_solve_time_s)
        planning_time_ms = planning_time_s * 1000.0
        eval_protocol = build_sim_eval_protocol(scene_profile)
        st.session_state["total_cost"] = total_cost
        if not path:
            st.error("❌ 未找到有效路径，请检查权重路段ID与路网是否一致")
            prog.empty(); status.empty(); return

        _scene_mode = scene_mode
        if showcase_direct_mode:
            _scene_weight_layers = dict(active_showcase_case.get("scene_weight_layers") or {})
            if not _scene_weight_layers:
                _scene_weight_layers = build_environment_weight_layers(
                    LIVE_EDGE_IDS,
                    scene_profile,
                    enable_spillover=True,
                )
        else:
            _scene_weight_layers = build_environment_weight_layers(
                LIVE_EDGE_IDS,
                scene_profile,
                enable_spillover=(_scene_mode == "showcase"),
            )
        _scene_weights = dict(_scene_weight_layers.get("weights") or {})
        _scene_raw_weights = dict(_scene_weight_layers.get("raw_only_weights") or {})
        _scene_spillover_weights = dict(_scene_weight_layers.get("spillover_weights") or {})
        _scene_spillover_hops = dict(_scene_weight_layers.get("spillover_hop_by_edge") or {})
        if showcase_direct_mode:
            _baseline_path = list(active_showcase_case.get("baseline_path") or [])
            _baseline_freeflow_cost = _safe_float(
                (active_showcase_case.get("debug") or {}).get("baseline_selection_cost_s"),
                active_showcase_case.get("baseline_time_s") or 0.0,
            )
            if not _baseline_path:
                _baseline_path, _baseline_freeflow_cost = path_engine.free_flow_route(net, start_edge, end_edge)
        else:
            _baseline_path, _baseline_freeflow_cost = path_engine.free_flow_route(net, start_edge, end_edge)
        _baseline_overlap_ratio = _edge_overlap_ratio(_baseline_path, path)
        baseline_estimated_time_s = (
            _estimate_route_time_on_weights(net, _baseline_path, _scene_weights) if _baseline_path else None
        )
        planned_estimated_time_s = (
            _estimate_route_time_on_weights(net, path, _scene_weights) if path else None
        )
        _baseline_selection_freeflow_time_s = (
            _estimate_route_time_with_cost_fn(net, _baseline_path, edge_freeflow_time) if _baseline_path else None
        )
        if not _baseline_path:
            st.warning("free-flow baseline_path 未生成，收益对比将仅显示规划路线的 SUMO / 估算结果。")

        status.text("🚗 正在运行SUMO仿真（规划路线）..."); prog.progress(60)
        simulation_result = sumo_engine.run_simulation(
            tuple(path),
            scene_profile,
            tuple(sorted(_scene_weights.items())),
            scene_weight_layer_payload=_scene_weight_layers,
            scene_mode=_scene_mode,
        )
        travel_time = _safe_float(simulation_result.get("travel_time"), 0.0)
        avg_speed = _safe_float(simulation_result.get("avg_speed"), 0.0)
        speed_data = list(simulation_result.get("speed_data") or [])
        pos_data = list(simulation_result.get("pos_data") or [])
        if not simulation_result.get("ok"):
            st.warning(
                "SUMO 未完成有效发车/执行。"
                f" stage={simulation_result.get('stage', 'unknown')}，"
                f" reason={simulation_result.get('reason', 'unknown')}"
            )
        st.caption(
            f"🔵 SUMO环境来自 scene profile `{scene_profile.scene_name}`，模型只影响边代价与路径选择；"
            "TLS、速度、封闭、流量、seed 不受模型输出控制"
        )
        if _scene_weight_layers.get("spillover_enabled"):
            _spill_stats = _scene_weight_layers.get("stats") or {}
            _runtime_spill = dict(simulation_result.get("runtime_spillover_summary") or {})
            st.caption(
                f"🟠 Showcase spillover 已启用：raw={int(_spill_stats.get('raw_incident_count', 0))} 条，"
                f"1-hop={int(_spill_stats.get('spillover_1hop_count', 0))} 条，"
                f"2-hop={int(_spill_stats.get('spillover_2hop_count', 0))} 条；"
                f"SUMO runtime spillover={'ON' if _runtime_spill.get('runtime_spillover_enabled') else 'OFF'}。"
            )
        else:
            st.caption("⚪ Benchmark/raw scene：未启用 spillover propagation，主结果保持原始 scene profile 扰动。")
        st.session_state["travel_time"] = travel_time

        baseline_simulation_result = {}
        baseline_sumo_travel_time_s = None
        if _baseline_path:
            status.text("🚗 正在运行SUMO仿真（baseline路线）..."); prog.progress(72)
            baseline_simulation_result = sumo_engine.run_simulation(
                tuple(_baseline_path),
                scene_profile,
                tuple(sorted(_scene_weights.items())),
                scene_weight_layer_payload=_scene_weight_layers,
                scene_mode=_scene_mode,
            )
            if baseline_simulation_result.get("ok"):
                baseline_sumo_travel_time_s = float(baseline_simulation_result.get("travel_time") or 0.0)
            else:
                st.warning(
                    "baseline 路线的 SUMO 未完成有效发车/执行。"
                    f" stage={baseline_simulation_result.get('stage', 'unknown')}，"
                    f" reason={baseline_simulation_result.get('reason', 'unknown')}"
                )

        # 提前计算 Dijkstra 参考路径，供 Tab2 / Tab4 复用。
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

        if baseline_sumo_travel_time_s is not None and simulation_result.get("ok"):
            _whether_case_is_positive_gain = bool(float(travel_time) + 1e-9 < float(baseline_sumo_travel_time_s))
        elif planned_estimated_time_s is not None and baseline_estimated_time_s is not None:
            _whether_case_is_positive_gain = bool(float(planned_estimated_time_s) + 1e-9 < float(baseline_estimated_time_s))
        else:
            _whether_case_is_positive_gain = None

        prog.progress(85)
        _detected_scene_type = str(_timing_info.get("scene_type", _effective_scene_type) or _effective_scene_type)
        st.session_state["results"] = {
            "timestamp":   datetime.now().isoformat(),
            "model":       selected_model,
            "path_algo":   _algo,
            "mode":        _mode_used,
            "scene_type":  _effective_scene_type,
            "detected_scene_type": _detected_scene_type,
            "scene_profile_selection_mode": scene_profile_selection_mode,
            "scene_profile_locked": scene_profile_locked,
            "fallback_used": _fallback_used,
            "fallback_reason": _fallback_reason,
            "constraint":  traffic_constraint,
            "scene_profile": scene_profile.scene_name,
            "scene_profile_config": scene_profile_to_dict(scene_profile),
            "scene_mode": _scene_mode,
            "workspace_scene_mode": scene_mode,
            "showcase_structured_mode": showcase_direct_mode,
            "showcase_case_name": str((active_showcase_case or {}).get("showcase_name") or ""),
            "showcase_display_reason": str((active_showcase_case or {}).get("display_reason") or ""),
            "showcase_affected_edges": list((active_showcase_case or {}).get("affected_edges") or []),
            "showcase_saving_seconds": (active_showcase_case or {}).get("saving_seconds"),
            "showcase_saving_ratio": (active_showcase_case or {}).get("saving_ratio"),
            "showcase_preferred_baseline_type": str((active_showcase_case or {}).get("preferred_baseline_type") or ""),
            "scene_weight_layers": _scene_weight_layers,
            "scene_raw_weights": _scene_raw_weights,
            "scene_spillover_weights": _scene_spillover_weights,
            "scene_spillover_hops": _scene_spillover_hops,
            "sim_eval_protocol": protocol_to_dict(eval_protocol),
            "scene_tls_profile": scene_profile.tls_profile_name,
            "scene_evaluation_window": [
                int(scene_profile.evaluation_start_time),
                int(scene_profile.evaluation_end_time),
            ],
            "scene_reroute_policy": build_reroute_policy(scene_profile),
            "sumo_result": simulation_result,
            "baseline_sumo_result": baseline_simulation_result,
            "sumo_runtime_spillover_summary": dict(simulation_result.get("runtime_spillover_summary") or {}),
            "baseline_sumo_runtime_spillover_summary": dict(baseline_simulation_result.get("runtime_spillover_summary") or {}),
            "start":       start_edge, "end": end_edge,
            "weights":     final_planning_weights,
            "raw_llm_weights": raw_llm_weights,
            "final_planning_weights": final_planning_weights,
            "path": path,
            "planned_path": path,
            "planned_path_source": "scene_aware_shortest_path" if showcase_direct_mode else "parser",
            "baseline_path": _baseline_path,
            "baseline_overlap_ratio": _baseline_overlap_ratio,
            "total_cost":  total_cost,
            "planning_time_s": planning_time_s,
            "planning_time_ms": planning_time_ms,
            "model_infer_time_s": float(infer_time or 0.0),
            "route_solve_time_s": route_solve_time_s,
            # 旧字段只给早期 UI/读取脚本兼容用。
            # 论文和答辩统计统一看上面的 protocol 字段。
            "travel_time": travel_time,
            "travel_time_s": travel_time,
            "travel_time_ms": float(travel_time) * 1000.0,
            "planned_sumo_travel_time_s": float(travel_time) if simulation_result.get("ok") else None,
            "baseline_sumo_travel_time_s": baseline_sumo_travel_time_s,
            "planned_estimated_time_s": planned_estimated_time_s,
            "baseline_estimated_time_s": baseline_estimated_time_s,
            "baseline_selection_freeflow_time_s": _baseline_selection_freeflow_time_s,
            "baseline_freeflow_route_cost_s": (
                float(_baseline_freeflow_cost) if math.isfinite(float(_baseline_freeflow_cost)) else None
            ),
            "baseline_path_source": (
                str((active_showcase_case or {}).get("preferred_baseline_type") or "free_flow_shortest_path")
            ),
            "whether_case_is_positive_gain": _whether_case_is_positive_gain,
            "final_route_after_safe_route_build": {
                "planned": list(simulation_result.get("final_route_edges") or []),
                "baseline": list(baseline_simulation_result.get("final_route_edges") or []),
            },
            "llm_parsed": _llm_parsed,
            "parsed_edges": edge_metrics["parsed_edges"],
            "parsed_edge_ratio": edge_metrics["parsed_edge_ratio"],
            "signal_edges": edge_metrics["signal_edges"],
            "signal_ratio": edge_metrics["signal_ratio"],
            "default_edges": edge_metrics["default_edges"],
            "avg_speed":   avg_speed,
            "time_loss_s": simulation_result.get("time_loss_s"),
            "waiting_time_s": simulation_result.get("waiting_time_s"),
            "stop_count": simulation_result.get("stop_count"),
            "depart_delay_s": simulation_result.get("depart_delay_s"),
            "route_length_edges": simulation_result.get("route_length_edges"),
            "affected_edge_overlap_ratio": simulation_result.get("affected_edge_overlap_ratio"),
            "baseline_time_loss_s": baseline_simulation_result.get("time_loss_s"),
            "baseline_waiting_time_s": baseline_simulation_result.get("waiting_time_s"),
            "baseline_stop_count": baseline_simulation_result.get("stop_count"),
            "baseline_depart_delay_s": baseline_simulation_result.get("depart_delay_s"),
            "baseline_route_length_edges": baseline_simulation_result.get("route_length_edges"),
            "baseline_affected_edge_overlap_ratio": baseline_simulation_result.get("affected_edge_overlap_ratio"),
            # 旧字段，保留给早期读取脚本。
            "infer_time":  infer_time,
            "timing_info": _timing_info,
            "sparse_parse": _sparse_parse,
            "gat_meta": _gat_meta,
            "gat_applied": _gat_applied,
            "gat_alpha_used": _gat_alpha_used,
        }
        st.session_state["last_wd"]   = weight_dict
        st.session_state["last_path"] = path
        st.session_state["last_baseline_path"] = list(_baseline_path or [])
        st.session_state["last_baseline_estimated_time"] = baseline_estimated_time_s
        st.session_state["last_planned_estimated_time"] = planned_estimated_time_s
        st.session_state["last_scene_weights"] = dict(_scene_weights)
        st.session_state["last_scene_weight_layers"] = dict(_scene_weight_layers)
        st.session_state["last_scene_raw_weights"] = dict(_scene_raw_weights)
        st.session_state["last_scene_spillover_weights"] = dict(_scene_spillover_weights)
        st.session_state["last_speed_data"] = list(speed_data or [])
        st.session_state["last_pos_data"] = list(pos_data or [])
        st.session_state["last_sumo_result"] = dict(simulation_result or {})
        st.session_state["last_baseline_sumo_result"] = dict(baseline_simulation_result or {})
        st.session_state["last_baseline_sumo_travel_time"] = baseline_sumo_travel_time_s
        st.session_state["last_comply_rate"] = float(_comply_rate)
        st.session_state["last_dijkstra_ref"] = list(_path_dijk_ref or [])
        st.session_state["last_dijkstra_cost"] = float(_cost_dijk_ref or 0.0)
        st.session_state["last_result_signature"] = current_request_signature
        try:
            st.session_state["last_runtime_audit_path"] = _write_runtime_audit_metrics(st.session_state["results"])
        except Exception as exc:
            logging.warning("Failed to write runtime audit metrics: %s", exc)

        prog.progress(100); status.text("✅ 完成！")
        time.sleep(0.4); prog.empty(); status.empty()

        st.session_state["run_count"] += 1
        st.session_state["run_history"].append({
            "id": st.session_state["run_count"],
            "start": start_edge, "end": end_edge,
            "travel_time": travel_time,
            "travel_time_s": travel_time,
            "planning_time_s": planning_time_s,
            "path_cost": total_cost,
            "edges_parsed": _llm_parsed,
        })

        severe = sum(1 for w in weight_dict.values() if w >= 6.0)
        medium = sum(1 for w in weight_dict.values() if 3.0 <= w < 6.0)
        clear  = sum(1 for w in weight_dict.values() if w < 3.0)
        display_payload = _build_result_display_payload(
            st.session_state["results"],
            edge_total=edge_total,
            path_algo=_algo,
            scene_weights=_scene_weights,
        )
        _render_main_result_panels(
            weight_ph=weight_ph,
            path_ph=path_ph,
            metric_ph=metric_ph,
            display_payload=display_payload,
            display_mode=display_mode,
            edge_total=edge_total,
            weight_dict=weight_dict,
            path=path,
            edge_coords=edge_coords,
            severe=severe,
            medium=medium,
            clear=clear,
            travel_time=travel_time,
            avg_speed=avg_speed,
            comply_rate=_comply_rate,
            baseline_path_ref=_baseline_path,
            scene_weights=_scene_weights,
            speed_data=speed_data,
            defense_mode=defense_mode,
            show_debug_details=show_debug_details,
            scene_env_payload=scene_env_payload,
        )

        # Tab3：真实对比实验
        if compare_ph is not None:
            compare_ph.empty()
        if compare_ph is not None:
          with compare_ph.container():
            _render_frozen_tables_workspace(
                total_cost_value=float(total_cost),
                weight_dict_value=dict(weight_dict or {}),
                planning_time_value=float(planning_time_s),
                debug=show_debug_details,
            )
        # Tab4
        if adv_ph is not None:
            adv_ph.empty()
        if adv_ph is not None:
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
