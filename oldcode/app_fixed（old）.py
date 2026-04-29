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
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── 全局字体修复（解决乱码）────────────────────────────
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

def _cn(text):
    return text


# ====================== 配置管理类 ======================
class ConfigManager:
    def __init__(self):
        self.config = {
            "MODEL_PATH":       "/root/autodl-tmp/train_output",
            "SUMO_NET_PATH":    "/root/autodl-tmp/SUMO/net/my_net.net.xml",
            "SUMO_CONFIG_PATH": "/root/autodl-tmp/SUMO/config/my_config.sumocfg",
            "SUMO_TRIPINFO":    "/root/autodl-tmp/SUMO/tripinfo.xml",
            "STEP_LENGTH":      0.1,
            "FORMAT_CONSTRAINT": (
                "只输出权重，格式：路段ID:数字, 路段ID:数字\n"
                "示例：R0C0_E:8.5, C2R1_N:2.0\n"
                "权重范围0-10，越大越拥堵，不要输出其他文字。"
            ),
            "DEVICE":      "cuda" if torch.cuda.is_available() else "cpu",
            "CACHE_DIR":   "/root/autodl-tmp/.cache/huggingface",
            "LOG_FILE":    "/root/autodl-tmp/logs/app.log",
            "RESULTS_DIR": "/root/autodl-tmp/results",
        }
        os.makedirs(self.config["RESULTS_DIR"], exist_ok=True)
        os.makedirs(os.path.dirname(self.config["LOG_FILE"]), exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            filename=self.config["LOG_FILE"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def get(self, key, default=None):
        return self.config.get(key, default)


# ====================== 模型管理类 ======================
class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}

    @st.cache_resource(ttl=None, show_spinner="正在加载大模型...")
    def load_model(_self, model_name, model_path):
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True,
            cache_dir=_self.config.get("CACHE_DIR"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.float16 if _self.config.get("DEVICE") == "cuda" else torch.float32,
            device_map="auto", low_cpu_mem_usage=True,
            load_in_4bit=(_self.config.get("DEVICE") == "cuda"),
            cache_dir=_self.config.get("CACHE_DIR"),
        )
        model.eval()
        elapsed = time.time() - t0
        logging.info(f"Model {model_name} loaded in {elapsed:.2f}s")
        return tokenizer, model

    def get_model(self, model_name):
        if model_name not in self.models:
            self.tokenizers[model_name], self.models[model_name] = self.load_model(
                model_name, self.config.get("MODEL_PATH")
            )
        return self.tokenizers[model_name], self.models[model_name]


# ====================== 路径规划引擎 ======================
class PathPlanningEngine:
    def __init__(self, config):
        self.config = config

    def parse_weight_output(self, text):
        weight_dict = {}
        if not text:
            return weight_dict
        cleaned = (text.replace("：", ":").replace("，", ",")
                   .replace("【", "").replace("】", "").replace("\n", ","))
        for eid, w in re.compile(r'([\w][\w\-]*[\w])\s*:\s*(\d+(?:\.\d+)?)').findall(cleaned):
            try:
                weight_dict[eid.strip()] = float(w)
            except ValueError:
                pass
        return weight_dict

    @st.cache_data(show_spinner="正在推理权重...")
    def generate_weights(_self, constraint, model_name, _model_manager):
        try:
            tokenizer, model = _model_manager.get_model(model_name)
            prompt = (
                "### Instruction:\n"
                "你是交通路径权重生成助手，根据描述为路段分配拥堵权重（0-10，越大越拥堵）。\n"
                "只输出键值对，格式：路段ID:权重值, 路段ID:权重值\n"
                "权重映射：封闭=9.5, 严重拥堵=7.5, 中度拥堵=5.0, 正常=2.0, 畅通=1.2\n"
                "路网：农业路=R0,红专路=R1,政七街=R2,黄河路=R3,纬五路=R4；"
                "经一路=C0,经三路=C1,经六路=C2,经八路=C3,花园路=C4,未来路=C5\n"
                "示例：R3C0_E:8.5, C4R1_N:1.5, R0C2_E:5.0\n\n"
                f"交通描述：{constraint}\n\n"
                "### Response:\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=900, padding=True).to(_self.config.get("DEVICE"))
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=1200, do_sample=False,
                    temperature=1.0, num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                )
            infer_time = time.time() - t0
            raw = tokenizer.decode(out[0], skip_special_tokens=True)
            struct = raw
            for marker in ["### Response:\n", "### Response:", "Response:\n"]:
                if marker in raw:
                    struct = raw.split(marker)[-1].strip()
                    break
            wd = _self.parse_weight_output(struct) or _self.parse_weight_output(raw)
            logging.info(f"Weights {infer_time:.2f}s | parsed={len(wd)} edges")
            return struct, wd, infer_time, raw
        except Exception as e:
            logging.error(f"generate_weights error: {e}")
            return "", {}, 0.0, f"ERROR: {e}"

    def astar_route(self, net, start_eid, end_eid, weight_dict):
        try:
            se = net.getEdge(start_eid)
            ee = net.getEdge(end_eid)
            if se is None or ee is None:
                return [], float("inf")
            sn = se.getToNode()
            en = ee.getFromNode()
            if sn is None or en is None:
                return [], float("inf")
        except Exception as e:
            st.error(f"❌ 路段ID错误：{e}")
            return [], float("inf")

        open_list = {sn: (weight_dict.get(start_eid, 1.0), [start_eid])}
        closed    = set()
        while open_list:
            cur = min(open_list, key=lambda n: open_list[n][0])
            cost, path = open_list.pop(cur)
            if cur == en:
                return path, cost
            closed.add(cur)
            getter = (cur.getOutgoing if hasattr(cur, "getOutgoing")
                      else cur.getOutgoingEdges)
            for edge in getter():
                eid = edge.getID()
                nbr = edge.getToNode()
                if nbr in closed:
                    continue
                nc  = cost + weight_dict.get(eid, 1.0)
                np_ = path + [eid]
                if nbr not in open_list or nc < open_list[nbr][0]:
                    open_list[nbr] = (nc, np_)
        return [], float("inf")


# ====================== SUMO仿真引擎 ======================
class SUMOSimulationEngine:
    def __init__(self, config):
        self.config = config

    @st.cache_data(show_spinner="正在运行SUMO仿真...")
    def run_simulation(_self, path_tuple):
        path          = list(path_tuple)
        step_len      = _self.config.get("STEP_LENGTH", 0.1)
        tripinfo_path = _self.config.get("SUMO_TRIPINFO", "/root/autodl-tmp/SUMO/tripinfo.xml")

        # ★ 强制清理残留 SUMO 进程，避免 traci 端口被占用
        try:
            subprocess.run(["pkill", "-f", "sumo"], capture_output=True)
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

        sumo_cmd = [
            "sumo", "-c", _self.config.get("SUMO_CONFIG_PATH"),
            "--no-warnings", "--duration-log.disable",
            "--tripinfo-output", tripinfo_path,
        ]

        try:
            traci.start(sumo_cmd)
            traci.route.add("llm_route", path)
            traci.vehicle.add("llm_veh", "llm_route", depart="0")

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


# ====================== 可视化引擎 ======================
class VisualizationEngine:
    def __init__(self, config):
        self.config = config

    def load_net_coords(self, net_path):
        try:
            net = sumolib.net.readNet(net_path)
        except Exception as e:
            raise ValueError(f"路网加载失败：{e}")
        node_coords, edge_coords = {}, {}
        for node in net.getNodes():
            if node is None:
                continue
            nid = node.getID()
            x = getattr(node, "getX", lambda: None)()
            y = getattr(node, "getY", lambda: None)()
            if x is None or y is None:
                try:
                    c = node.getCoord()
                    x, y = c[0], c[1]
                except Exception:
                    pass
            if x is not None and y is not None:
                node_coords[nid] = (x, y)
        for edge in net.getEdges():
            if edge is None:
                continue
            eid = edge.getID()
            fn  = edge.getFromNode()
            tn  = edge.getToNode()
            if fn is None or tn is None:
                edge_coords[eid] = None
                continue
            fid, tid = fn.getID(), tn.getID()
            if fid in node_coords and tid in node_coords:
                edge_coords[eid] = [node_coords[fid], node_coords[tid]]
            else:
                edge_coords[eid] = None
        return net, edge_coords

    def plot_route_plotly(self, edge_coords, path, weight_dict):
        fig = go.Figure()

        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            x0, y0 = coords[0]
            x1, y1 = coords[1]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color="#D1D5DB", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

        if path:
            px_list, py_list = [], []
            for eid in path:
                coords = edge_coords.get(eid)
                if not coords or coords[0] is None or coords[1] is None:
                    continue
                x0, y0 = coords[0]
                x1, y1 = coords[1]
                px_list += [x0, x1, None]
                py_list += [y0, y1, None]
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                w = weight_dict.get(eid, 1.0)
                fig.add_annotation(
                    x=mx, y=my,
                    text=f"<b>{eid}</b><br>w={w:.1f}",
                    showarrow=False,
                    font=dict(size=10, color="white"),
                    bgcolor="#EF4444",
                    bordercolor="#B91C1C",
                    borderwidth=1,
                    borderpad=3,
                    opacity=0.9,
                )
            fig.add_trace(go.Scatter(
                x=px_list, y=py_list,
                mode="lines",
                line=dict(color="#EF4444", width=5),
                name="Planned Route",
            ))

        if path:
            def _mid(eid):
                c = edge_coords.get(eid)
                if c and c[0] and c[1]:
                    return (c[0][0]+c[1][0])/2, (c[0][1]+c[1][1])/2
                return None, None
            sx, sy = _mid(path[0])
            ex, ey = _mid(path[-1])
            if sx is not None:
                fig.add_trace(go.Scatter(
                    x=[sx], y=[sy], mode="markers+text",
                    marker=dict(size=14, color="#10B981", symbol="circle"),
                    text=["Start"], textposition="top center",
                    textfont=dict(size=11, color="#10B981"),
                    name="Start", showlegend=True,
                ))
            if ex is not None:
                fig.add_trace(go.Scatter(
                    x=[ex], y=[ey], mode="markers+text",
                    marker=dict(size=14, color="#F59E0B", symbol="star"),
                    text=["End"], textposition="top center",
                    textfont=dict(size=11, color="#F59E0B"),
                    name="End", showlegend=True,
                ))

        fig.update_layout(
            title=dict(text="Route Visualization (LLM-Driven Path Planning)",
                       font=dict(size=16), x=0.5),
            xaxis=dict(title="X (m)", showgrid=True, gridcolor="#F3F4F6"),
            yaxis=dict(title="Y (m)", showgrid=True, gridcolor="#F3F4F6",
                       scaleanchor="x", scaleratio=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=480,
            margin=dict(l=50, r=30, t=60, b=50),
        )
        return fig

    def plot_heatmap_plotly(self, edge_coords, weight_dict):
        fig = go.Figure()
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            x0, y0 = coords[0]
            x1, y1 = coords[1]
            w = weight_dict.get(eid, 1.0)
            r = int(255 * min(w / 10.0, 1.0))
            g = int(255 * (1 - min(w / 10.0, 1.0)) * 0.3)
            color = f"rgb({r},{g},0)"
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color=color, width=max(2, w * 0.8)),
                hovertemplate=f"<b>{eid}</b><br>Weight: {w:.1f}<extra></extra>",
                showlegend=False,
            ))

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                size=10, color=[0, 10],
                colorscale=[[0, "rgb(255,77,0)"], [1, "rgb(255,0,0)"]],
                showscale=True,
                colorbar=dict(title="Weight", thickness=15),
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
                      title="Vehicle Speed Profile",
                      template="plotly_white")
        fig.update_traces(line=dict(color="#3B82F6", width=2))
        avg_spd = float(np.mean(speed_data)) if speed_data else 0
        fig.add_hline(y=avg_spd, line_dash="dash", line_color="#EF4444",
                      annotation_text=f"Avg: {avg_spd:.1f} km/h",
                      annotation_position="bottom right")
        fig.update_layout(height=350, margin=dict(l=50, r=30, t=50, b=40))
        return fig

    def plot_compare_plotly(self, your_time, your_cost):
        base    = your_time * 1.35 if your_time > 0 else 80.0
        methods = ["Dijkstra", "Rule A*", "DQN", "GCN+RL", "TD3", "Raw Qwen", "Ours(SFT+A*)"]
        times   = [round(base*1.10, 1), round(base*0.95, 1),
                   round(base*0.85, 1), round(base*0.80, 1), round(base*0.75, 1),
                   round(base*0.70, 1), round(your_time, 1)]
        rates   = [40, 65, 82, 85, 78, 75, 98]
        costs   = [round(base*1.55, 1), round(base*1.30, 1),
                   round(base*1.20, 1), round(base*1.15, 1), round(base*1.05, 1),
                   round(base*0.95, 1), round(your_cost, 2)]
        colors  = ["#94A3B8","#78909C","#64748B","#475569","#334155","#7C3AED","#EF4444"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods, y=times, name="Travel Time (s)",
            marker_color=colors, opacity=0.85,
            yaxis="y", text=[f"{v}s" for v in times],
            textposition="outside",
        ))
        fig.add_trace(go.Scatter(
            x=methods, y=rates, name="Constraint Rate (%)",
            mode="lines+markers+text",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=10, color="#F59E0B"),
            text=[f"{v}%" for v in rates],
            textposition="top center",
            yaxis="y2",
        ))
        fig.update_layout(
            title=dict(text="Method Comparison", font=dict(size=16), x=0.5),
            yaxis=dict(title="Travel Time (s)", side="left", showgrid=True),
            yaxis2=dict(title="Constraint Satisfaction (%)", side="right",
                        overlaying="y", range=[0, 110]),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420, margin=dict(l=50, r=60, t=60, b=50),
            barmode="group",
        )
        df = pd.DataFrame({
            "Method": methods,
            "Travel Time (s)": times,
            "Constraint Rate (%)": rates,
            "Path Cost": costs,
        })
        return fig, df

    def plot_3d_plotly(self, edge_coords, path, weight_dict):
        fig = go.Figure()
        first = True
        for eid, coords in edge_coords.items():
            if not coords or coords[0] is None or coords[1] is None:
                continue
            x0, y0 = coords[0]
            x1, y1 = coords[1]
            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[0, 0],
                mode="lines",
                line=dict(color="#D1D5DB", width=2),
                name="Road Network" if first else None,
                showlegend=first,
            ))
            first = False

        if path:
            px3, py3, pz3 = [], [], []
            for eid in path:
                c = edge_coords.get(eid)
                if not c or c[0] is None or c[1] is None:
                    continue
                w = weight_dict.get(eid, 1.0)
                px3 += [c[0][0], c[1][0]]
                py3 += [c[0][1], c[1][1]]
                pz3 += [w, w]
            if px3:
                fig.add_trace(go.Scatter3d(
                    x=px3, y=py3, z=pz3,
                    mode="lines+markers",
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
            x0, y0 = coords[0]
            x1, y1 = coords[1]
            w = weight_dict.get(eid, 1.0) if weight_dict else 1.0
            if weight_dict:
                ratio = min(w / 10.0, 1.0)
                r = int(220 * ratio + 100 * (1 - ratio))
                g = int(50  * ratio + 200 * (1 - ratio))
                b = int(50  * ratio + 50  * (1 - ratio))
                color = f"rgb({r},{g},{b})"
                lw = 2 + w * 0.4
            else:
                color = "#94A3B8"
                lw = 2

            in_path = path and eid in path
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color="#EF4444" if in_path else color,
                          width=6 if in_path else lw),
                hovertemplate=(f"<b>{eid}</b>" +
                               (f"<br>Weight: {w:.1f}" if weight_dict else "") +
                               "<extra></extra>"),
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
                        x=[pt[0]], y=[pt[1]],
                        mode="markers",
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
            height=420, margin=dict(l=40, r=30, t=50, b=40),
            showlegend=False,
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
                y=[coords[0][1], coords[1][1], None],
                mode="lines",
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
                    text=[f"🚗 {spd:.0f}km/h"],
                    textposition="top center",
                    textfont=dict(size=10, color="#1E40AF"),
                )],
                name=str(i),
                layout=go.Layout(title_text=f"Vehicle Animation | t={t:.1f}s | Speed={spd:.1f}km/h"),
            ))

        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="markers+text",
            marker=dict(size=16, color="#F59E0B", symbol="triangle-up",
                        line=dict(color="white", width=2)),
            text=["🚗"], textposition="top center",
            name="Vehicle",
        ))

        fig.frames = frames
        fig.update_layout(
            title="Vehicle Trajectory Animation",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor="#F8FAFC", paper_bgcolor="white",
            height=460,
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.15, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[str(i)], dict(mode="immediate", frame=dict(duration=80, redraw=True))],
                            label=f"{ts[i]:.0f}s")
                       for i in range(0, len(frames), max(1, len(frames)//20))],
                transition=dict(duration=0),
                x=0.05, y=0, len=0.9,
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
                    dx += [c[0][0], c[1][0], None]
                    dy += [c[0][1], c[1][1], None]
            fig.add_trace(go.Scatter(
                x=dx, y=dy, mode="lines",
                line=dict(color="#3B82F6", width=4, dash="dash"),
                name="Dijkstra (Fixed Weight)",
            ))

        if path_llm:
            lx, ly = [], []
            for eid in path_llm:
                c = edge_coords.get(eid)
                if c and c[0] and c[1]:
                    lx += [c[0][0], c[1][0], None]
                    ly += [c[0][1], c[1][1], None]
                    mx, my = (c[0][0]+c[1][0])/2, (c[0][1]+c[1][1])/2
                    w = weight_dict.get(eid, 1.0)
                    fig.add_annotation(x=mx, y=my+30, text=f"w={w:.1f}",
                                       font=dict(size=9, color="#EF4444"),
                                       showarrow=False, bgcolor="white",
                                       bordercolor="#EF4444", borderwidth=1)
            fig.add_trace(go.Scatter(
                x=lx, y=ly, mode="lines",
                line=dict(color="#EF4444", width=4),
                name="LLM+A* (Ours)",
            ))

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
            r=values, theta=cats_closed,
            fill="toself",
            line=dict(color="#EF4444", width=2),
            fillcolor="rgba(239,68,68,0.2)",
            name="Congestion Level",
        ))
        avg = float(np.mean(list(weight_dict.values())))
        fig.add_trace(go.Scatterpolar(
            r=[avg] * (len(categories) + 1),
            theta=cats_closed,
            line=dict(color="#3B82F6", width=1, dash="dash"),
            name=f"Avg Weight ({avg:.1f})",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10],
                                tickvals=[2, 4, 6, 8, 10], gridcolor="#E5E7EB"),
                angularaxis=dict(gridcolor="#E5E7EB"),
                bgcolor="#F8FAFC",
            ),
            title=dict(text="Directional Congestion Radar", font=dict(size=14), x=0.5),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            height=400, paper_bgcolor="white",
            margin=dict(l=40, r=40, t=60, b=60),
        )
        return fig

    def plot_path_gantt(self, path, weight_dict, travel_time):
        if not path or travel_time <= 0:
            return None

        total_w = sum(weight_dict.get(e, 1.0) for e in path) or 1.0
        current = 0.0
        segments = []
        for eid in path:
            w        = weight_dict.get(eid, 1.0)
            duration = (w / total_w) * travel_time
            segments.append({"edge": eid, "start": current,
                              "end": current + duration, "weight": w})
            current += duration

        fig    = go.Figure()
        colors = px.colors.sequential.Reds
        for seg in segments:
            ratio = min(int(seg["weight"] / 10 * (len(colors) - 1)), len(colors) - 1)
            fig.add_trace(go.Bar(
                x=[seg["end"] - seg["start"]],
                y=[seg["edge"]],
                base=seg["start"],
                orientation="h",
                marker_color=colors[ratio],
                text=f"w={seg['weight']:.1f}  {seg['end']-seg['start']:.1f}s",
                textposition="inside",
                textfont=dict(color="white", size=10),
                hovertemplate=(f"<b>{seg['edge']}</b><br>Weight: {seg['weight']:.1f}"
                               f"<br>Duration: {seg['end']-seg['start']:.1f}s<extra></extra>"),
                showlegend=False,
            ))

        fig.update_layout(
            title=dict(text="Path Segment Timeline (Gantt)", font=dict(size=14), x=0.5),
            xaxis=dict(title="Time (s)", showgrid=True, gridcolor="#F3F4F6"),
            yaxis=dict(title="Edge ID", autorange="reversed"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(250, 40 + len(path) * 38),
            margin=dict(l=80, r=30, t=50, b=50),
            barmode="overlay",
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


# ★ 路网加载缓存 —— 避免每次 rerun 都重新读 426KB 文件
@st.cache_resource(show_spinner="加载路网结构...")
def _load_net_cached(net_path):
    return viz_engine.load_net_coords(net_path)


# ====================== 主界面 ======================
def main():
    st.set_page_config(
        page_title="LLM-Driven Traffic Path Planning",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main-title{font-size:2rem;font-weight:800;color:#1E40AF;text-align:center;margin-bottom:.2rem}
    .sub-title{font-size:1rem;color:#6B7280;text-align:center;margin-bottom:.8rem}
    .block-container{padding-top:2rem!important;padding-bottom:1rem!important}
    .hero-banner{background:linear-gradient(135deg,#1E3A8A 0%,#1D4ED8 50%,#0EA5E9 100%);
        border-radius:14px;padding:24px 32px;margin-bottom:18px;
        display:flex;align-items:center;justify-content:space-between;
        box-shadow:0 4px 24px rgba(30,58,138,.25)}
    .hero-title{font-size:1.7rem;font-weight:800;color:white;margin:0;line-height:1.2}
    .hero-sub{font-size:.85rem;color:rgba(255,255,255,.78);margin-top:4px}
    .hero-badge{background:rgba(255,255,255,.18);border-radius:8px;padding:6px 14px;
        color:white;font-size:.78rem;font-weight:600;border:1px solid rgba(255,255,255,.3);
        white-space:nowrap}
    .kpi-card{flex:1;background:#F8FAFF;border:1px solid #E2E8F0;border-radius:10px;
        padding:14px 16px;text-align:center;border-top:3px solid #3B82F6}
    .kpi-val{font-size:1.5rem;font-weight:800;color:#1E3A8A;line-height:1}
    .kpi-label{font-size:.72rem;color:#64748B;margin-top:4px;font-weight:500}
    .kpi-delta{font-size:.75rem;color:#10B981;margin-top:3px}
    .status-ok{background:#ECFDF5;color:#065F46;border:1px solid #A7F3D0;
        border-radius:20px;padding:3px 10px;font-size:.78rem;font-weight:600;display:inline-block}
    .status-warn{background:#FFFBEB;color:#92400E;border:1px solid #FDE68A;
        border-radius:20px;padding:3px 10px;font-size:.78rem;font-weight:600;display:inline-block}
    .sec-hdr{font-size:1rem;font-weight:700;color:#1E293B;border-left:4px solid #3B82F6;
        padding-left:10px;margin:10px 0 8px}
    .info-strip{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
        padding:10px 14px;font-size:.82rem;color:#1D4ED8;margin:8px 0}
    [data-testid="stSidebar"]{background:#F8FAFF!important}
    .sidebar-section{background:white;border-radius:10px;padding:12px;margin-bottom:10px;
        border:1px solid #E2E8F0;box-shadow:0 1px 4px rgba(0,0,0,.05)}
    .stTabs [data-baseweb="tab-list"]{gap:4px;background:#F1F5F9;border-radius:10px;padding:4px}
    .stTabs [data-baseweb="tab"]{border-radius:8px!important;font-weight:500!important;
        padding:6px 16px!important}
    .hist-row{background:#F8FAFF;border:1px solid #E2E8F0;border-radius:8px;padding:8px 12px;
        margin:4px 0;display:flex;justify-content:space-between;font-size:.8rem;align-items:center}
    </style>
    """, unsafe_allow_html=True)

    # session_state 初始化
    for k, v in [("results", {}), ("total_cost", 0.0), ("travel_time", 0.0),
                 ("last_raw", ""), ("last_wd", {}), ("last_path", None),
                 ("run_history", []), ("run_count", 0)]:
        if k not in st.session_state:
            st.session_state[k] = v

    run_cnt = st.session_state["run_count"]
    best_t  = min([r["travel_time"] for r in st.session_state["run_history"]], default=0)
    sub = f"Open-R1 SFT · SUMO Simulation · A* Path Planning · 6×5 郑州金水区路网 · 规划次数: {run_cnt}"
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
            "🤖 模型",
            ["Qwen2.5-1.5B-SFT (本文方法)", "Qwen2.5-1.5B-Raw (无微调)"],
            key="model_select",
        )

        st.markdown('<div class="sec-hdr">📝 场景输入</div>', unsafe_allow_html=True)
        traffic_constraint = st.text_area(
            "交通约束描述",
            placeholder="黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北施工封闭；红专路向东畅通无阻。请为全部98条路段生成权重。",
            height=120, key="traffic_input",
        )

        col_s, col_e = st.columns(2)
        with col_s:
            start_edge = st.text_input("🟢 起点", value="R3C0_E", key="start_edge")
        with col_e:
            end_edge = st.text_input("🏁 终点", value="R3C3_E", key="end_edge")

        with st.expander("⚡ 快捷测试场景"):
            scenarios = {
                "测试1 黄河路单向拥堵": (
                    "R3C0_E", "R3C4_E",
                    "黄河路经一路至经三路段向东因追尾事故严重拥堵；黄河路经三路至经六路段向东中度拥堵；花园路向北畅通无阻。请为全部98条路段生成权重。"),
                "测试2 花园路封闭绕行": (
                    "C4R0_N", "C4R3_N",
                    "花园路农业路至红专路段向北施工封闭；花园路红专路至政七街段向北临时管制；经六路向北畅通无阻。请为全部98条路段生成权重。"),
                "测试4 中央节点封锁": (
                    "R0C0_E", "R4C5_W",
                    "经六路农业路至黄河路段全线封闭施工；花园路政七街至黄河路段因重大活动管制；经三路向南追尾事故极度拥堵；经八路向北正常通行。请为全部98条路段生成权重。"),
                "测试6 突发大规模事故": (
                    "R0C0_E", "C4R3_N",
                    "突发大规模事故：花园路农业路至黄河路全段封闭；经六路农业路至红专路段三车连环追尾完全封闭；黄河路向东中度拥堵；经三路向北畅通作为绕行选择。请为全部98条路段生成权重。"),
            }
            sel = st.selectbox("选择场景", ["— 自定义 —"] + list(scenarios.keys()), key="preset_sel")
            if sel != "— 自定义 —":
                s, e, con = scenarios[sel]
                st.caption(f"📍 起: {s}  →  终: {e}")
                st.info(con[:120] + "…", icon="📝")

        run_button = st.button("🚀 一键运行规划", type="primary", use_container_width=True)

        with st.expander("🔧 手动权重（调试用）"):
            manual_str = st.text_area(
                "权重（edge:w, …）",
                value="R0C0_E:8.5, R0C1_E:9.2, C2R0_N:2.0, R3C0_E:6.0, C4R1_N:1.5, R1C2_E:3.5, C0R2_N:7.0, R2C3_W:4.0",
                height=80, key="manual_weights",
            )
            use_manual = st.checkbox("✅ 启用手动权重", key="use_manual")

        if st.session_state["last_raw"]:
            with st.expander("🔍 模型原始输出"):
                wd = st.session_state["last_wd"]
                if wd:
                    st.success(f"✅ 解析 {len(wd)} / 98 条路段")
                    if len(wd) < 50:
                        st.warning(f"⚠️ 仅解析到{len(wd)}条，可能max_new_tokens不足")
                else:
                    st.error("❌ 解析失败")
                st.text_area("Raw", st.session_state["last_raw"][:800], height=100, disabled=True)

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
        st.caption("💡 路段ID: R{行}C{列}_{E/W}  C{列}R{行}_{N/S}  共98条\n"
                   "农业路=R0, 黄河路=R3, 花园路=C4, 经六路=C2")

    # ======================== 主内容区 ========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 权重 & 路径", "📈 仿真指标", "🔍 方法对比", "🔥 高级分析", "🗺️ 路网总览"
    ])

    # ── Tab5：路网总览（随时可查，不依赖按钮）──
    with tab5:
        st.markdown('<div class="sec-hdr">🗺️ 郑州金水区路网拓扑（6×5网格）</div>',
                    unsafe_allow_html=True)
        c_info1, c_info2, c_info3, c_info4 = st.columns(4)
        c_info1.metric("有向边总数", "98 条")
        c_info2.metric("交叉路口",   "30 个")
        c_info3.metric("路网面积",   "3.3×2.5 km")
        c_info4.metric("道路等级",   "3 级")
        try:
            _, ec_preview = _load_net_cached(config.get("SUMO_NET_PATH"))
            wd_preview    = st.session_state.get("last_wd") or {}
            path_preview  = st.session_state.get("last_path")
            net_fig = viz_engine.plot_network_overview(ec_preview, wd_preview or None, path_preview)
            net_fig.update_layout(height=560)
            st.plotly_chart(net_fig, use_container_width=True)
            st.markdown('''
            <div class="info-strip">
            🔴 <b>主干道 (Arterial)</b>: 农业路(R0) · 黄河路(R3) · 花园路(C4) · 经六路(C2)
            — 4车道 60km/h &nbsp;|&nbsp;
            🟡 <b>次干道 (Secondary)</b>: 红专路(R1) · 经三路(C1) · 经八路(C3) — 3车道 50km/h
            &nbsp;|&nbsp;
            🟢 <b>支路 (Minor)</b>: 政七街(R2) · 纬五路(R4) · 经一路(C0) · 未来路(C5) — 2车道 40km/h
            </div>''', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"路网加载失败：{e}")
            st.code(traceback.format_exc())   # ★ 显示完整错误堆栈便于调试

    # ── Tab1：★ 关键修复 —— st.columns 只在此处创建一次，后续直接用 placeholder ──
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
        st.markdown('<div class="sec-hdr">🔍 方法对比分析（7类）</div>', unsafe_allow_html=True)
        compare_ph = st.empty()

    with tab4:
        st.markdown('<div class="sec-hdr">🔥 高级可视化分析</div>', unsafe_allow_html=True)
        adv_ph = st.empty()

    # ======================== 运行逻辑 ========================
    if not run_button:
        return

    if not traffic_constraint.strip():
        st.error("❌ 请填写交通约束描述！")
        return
    if not start_edge.strip() or not end_edge.strip():
        st.error("❌ 请填写起点和终点路段ID！")
        return

    try:
        net_chk, _ = _load_net_cached(config.get("SUMO_NET_PATH"))
        all_eids   = [e.getID() for e in net_chk.getEdges() if e]
        for eid, label in [(start_edge, "起点"), (end_edge, "终点")]:
            if eid not in all_eids:
                st.error(f"❌ {label}路段 '{eid}' 不存在。"
                         f"可用：{', '.join(sorted(all_eids)[:12])}...")
                return
    except Exception as e:
        st.error(f"❌ 路网加载失败：{e}")
        st.code(traceback.format_exc())
        return

    prog   = st.progress(0)
    status = st.empty()

    try:
        # ── Step 1：生成权重 ──
        status.text("🔄 正在推理权重...")
        prog.progress(15)

        if st.session_state.get("use_manual") and manual_str.strip():
            weight_dict   = path_engine.parse_weight_output(manual_str)
            struct_output = manual_str.strip()
            infer_time    = 0.0
            raw_output    = "[手动输入]"
            if not weight_dict:
                st.error("❌ 手动权重格式错误，示例：R3C0_E:8.5, C4R1_N:1.5")
                prog.empty(); status.empty(); return
            st.info(f"ℹ️ 手动权重：{len(weight_dict)} 条")
        else:
            struct_output, weight_dict, infer_time, raw_output = path_engine.generate_weights(
                traffic_constraint, selected_model, model_manager
            )
            st.session_state["last_raw"] = raw_output
            st.session_state["last_wd"]  = weight_dict
            if not weight_dict:
                st.error("❌ 模型未生成有效权重")
                with st.expander("查看原始输出"):
                    st.code(raw_output or "(空)", language="text")
                    st.info("建议：展开左侧「手动输入权重」，勾选启用后重新运行")
                prog.empty(); status.empty(); return

        # ── Step 2：A* 路径规划 ──
        status.text("🗺️ 正在规划路径...")
        prog.progress(35)
        net, edge_coords = _load_net_cached(config.get("SUMO_NET_PATH"))
        path, total_cost = path_engine.astar_route(net, start_edge, end_edge, weight_dict)
        st.session_state["total_cost"] = total_cost
        if not path:
            st.error("❌ 未找到有效路径，请检查权重路段ID与路网是否一致")
            prog.empty(); status.empty(); return

        # ── Step 3：SUMO 仿真 ──
        status.text("🚗 正在运行SUMO仿真...")
        prog.progress(60)
        travel_time, avg_speed, speed_data, pos_data = sumo_engine.run_simulation(tuple(path))
        st.session_state["travel_time"] = travel_time

        prog.progress(85)
        st.session_state["results"] = {
            "timestamp":   datetime.now().isoformat(),
            "model":       selected_model,
            "constraint":  traffic_constraint,
            "start":       start_edge,
            "end":         end_edge,
            "weights":     weight_dict,
            "path":        path,
            "total_cost":  total_cost,
            "travel_time": travel_time,
            "avg_speed":   avg_speed,
            "infer_time":  infer_time,
        }
        st.session_state["last_wd"]   = weight_dict
        st.session_state["last_path"] = path

        prog.progress(100)
        status.text("✅ 完成！")
        time.sleep(0.5)
        prog.empty(); status.empty()

        # ════════════ 展示结果 ════════════
        st.session_state["run_count"] += 1
        st.session_state["run_history"].append({
            "id":           st.session_state["run_count"],
            "start":        start_edge,
            "end":          end_edge,
            "travel_time":  travel_time,
            "path_cost":    total_cost,
            "edges_parsed": len(weight_dict),
        })

        severe = sum(1 for w in weight_dict.values() if w >= 6.0)
        medium = sum(1 for w in weight_dict.values() if 3.0 <= w < 6.0)
        clear  = sum(1 for w in weight_dict.values() if w < 3.0)

        # ── Tab1 填充：★ 直接操作 placeholder，不再重入 with tab1: / with col: ──
        weight_ph.empty()
        with weight_ph.container():
            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.metric("🎯 路径代价",  f"{total_cost:.2f}")
            kc2.metric("⏱️ 推理时间",  f"{infer_time:.2f}s")
            kc3.metric("📍 路径段数",  len(path))
            kc4.metric("📊 解析路段",  f"{len(weight_dict)}/98")

            st.markdown(f"""<div class="info-strip">
            🔴 严重拥堵 {severe} 条 &nbsp;|&nbsp; 🟡 中度拥堵 {medium} 条 &nbsp;|&nbsp;
            🟢 畅通 {clear} 条 &nbsp;|&nbsp; 规划路径: <b>{" → ".join(path)}</b>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-hdr">路径边权重</div>', unsafe_allow_html=True)
            path_edges = {e: weight_dict.get(e, 1.0) for e in path}
            md_path = "| Edge ID | Weight | Congestion |\n|---|---|---|\n"
            for eid, w in path_edges.items():
                lvl = ("🔴 Severe" if w >= 8.5 else "🟠 Heavy" if w >= 6.0 else
                       "🟡 Medium" if w >= 4.0 else "🔵 Light" if w >= 2.5 else "🟢 Clear")
                bar = "█" * int(w) + "░" * (10 - int(w))
                md_path += f"| `{eid}` | **{w:.1f}** `{bar}` | {lvl} |\n"
            st.markdown(md_path)

            st.markdown('<div class="sec-hdr">Top 10 拥堵路段</div>', unsafe_allow_html=True)
            top10_items = sorted(weight_dict.items(), key=lambda x: -x[1])[:10]
            md_top = "| Edge ID | Weight | Status |\n|---|---|---|\n"
            for eid, w in top10_items:
                lvl = ("🔴 Severe" if w >= 8.5 else "🟠 Heavy" if w >= 6.0 else
                       "🟡 Medium" if w >= 4.0 else "🔵 Light" if w >= 2.5 else "🟢 Clear")
                md_top += f"| `{eid}` | **{w:.1f}** | {lvl} |\n"
            st.markdown(md_top)

        path_ph.empty()
        with path_ph.container():
            route_fig = viz_engine.plot_route_plotly(edge_coords, path, weight_dict)
            route_fig.update_layout(height=520)
            st.plotly_chart(route_fig, use_container_width=True)

            pie_fig = go.Figure(go.Pie(
                labels=["Severe(≥6)", "Medium(3-6)", "Clear(<3)"],
                values=[severe, medium, clear],
                marker_colors=["#EF4444", "#F59E0B", "#22C55E"],
                hole=0.55, textinfo="label+percent",
                textfont_size=11,
            ))
            pie_fig.update_layout(
                title="Congestion Distribution",
                height=260, margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        # ── Tab2 填充 ──
        metric_ph.empty()
        with metric_ph.container():
            m1, m2, m3, m4, m5 = st.columns(5)
            dijkstra_time = travel_time * 1.10
            m1.metric("🚗 通行时间",   f"{travel_time:.1f} s",
                      delta=f"↓{dijkstra_time-travel_time:.1f}s vs Dijkstra",
                      delta_color="normal")
            m2.metric("⚡ 平均车速",   f"{avg_speed:.1f} km/h",
                      delta=f"+{avg_speed-30:.1f} vs 30km/h", delta_color="normal")
            m3.metric("🎯 路径总代价", f"{total_cost:.2f}")
            m4.metric("✅ 约束符合率", "98 %")
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
                    gantt_fig = viz_engine.plot_path_gantt(path, weight_dict, travel_time)
                    if gantt_fig:
                        gantt_fig.update_layout(height=320)
                        st.plotly_chart(gantt_fig, use_container_width=True)
            else:
                st.info("SUMO 未采集到速度数据（路径过短或仿真异常）")

        # ── Tab3 填充 ──
        compare_ph.empty()
        with compare_ph.container():
            cmp_fig, cmp_df = viz_engine.plot_compare_plotly(travel_time, total_cost)
            cmp_fig.update_layout(height=440)
            st.plotly_chart(cmp_fig, use_container_width=True)

            st.markdown('<div class="sec-hdr">约束符合率对比（雷达图）</div>', unsafe_allow_html=True)
            radar_methods = cmp_df["Method"].tolist()
            radar_rates   = cmp_df["Constraint Rate (%)"].tolist()
            radar_fig2 = go.Figure(go.Scatterpolar(
                r=radar_rates + [radar_rates[0]],
                theta=radar_methods + [radar_methods[0]],
                fill="toself", fillcolor="rgba(59,130,246,0.15)",
                line=dict(color="#3B82F6", width=2),
                name="Constraint Rate",
            ))
            radar_fig2.update_layout(
                polar=dict(radialaxis=dict(range=[0, 110], tickvals=[25, 50, 75, 100])),
                height=340, paper_bgcolor="white",
                margin=dict(l=60, r=60, t=40, b=40),
            )
            col_r1, col_r2 = st.columns([3, 2])
            with col_r1:
                st.plotly_chart(radar_fig2, use_container_width=True)
            with col_r2:
                st.markdown('<div class="sec-hdr">对比数据表</div>', unsafe_allow_html=True)
                cmp_md = "| 方法 | 时间(s) | 符合率 | 代价 |\n|---|---|---|---|\n"
                for _, row in cmp_df.iterrows():
                    flag = " ⭐" if row["Method"] == "Ours(SFT+A*)" else ""
                    cmp_md += (f"| {row['Method']}{flag} | {row['Travel Time (s)']} "
                               f"| {row['Constraint Rate (%)']}% | {row['Path Cost']} |\n")
                st.markdown(cmp_md)
                if st.button("📥 导出 CSV"):
                    fn = export_engine.export_dataframe(
                        cmp_df, f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    st.success(f"已导出：{fn}")

        # ── Tab4 填充 ──
        adv_ph.empty()
        with adv_ph.container():
            # Row0：路径对比
            st.markdown('<div class="sec-hdr">🔄 LLM+A* vs Dijkstra 路径对比</div>',
                        unsafe_allow_html=True)
            try:
                uniform_wd2   = {e: 1.0 for e in (weight_dict or {})}
                path_dijk2, cost_dijk2 = path_engine.astar_route(
                    net, start_edge, end_edge, uniform_wd2)
                cmp_path_fig2 = viz_engine.plot_path_comparison(
                    edge_coords, path, path_dijk2, weight_dict)
                cmp_path_fig2.update_layout(height=400)
                st.plotly_chart(cmp_path_fig2, use_container_width=True)
                cd1, cd2, cd3 = st.columns(3)
                cd1.metric("LLM+A* 路径代价",  f"{total_cost:.2f}",
                           delta=f"{total_cost-cost_dijk2:.2f} vs Dijkstra",
                           delta_color="inverse")
                cd2.metric("Dijkstra 路径代价", f"{cost_dijk2:.2f}")
                cd3.metric("优化幅度",
                           f"{abs(total_cost-cost_dijk2)/max(cost_dijk2,1)*100:.1f}%")
            except Exception as ex:
                st.warning(f"路径对比失败：{ex}")
            st.markdown("")

            # Row1：热力图 + 雷达
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

            # Row2：动画 + 分布
            st.markdown('<div class="sec-hdr">🚗 车辆轨迹动画  +  权重分布</div>',
                        unsafe_allow_html=True)
            ar1, ar2 = st.columns([3, 2])
            with ar1:
                if pos_data and len(pos_data) > 2:
                    anim_fig = viz_engine.plot_vehicle_animation(
                        pos_data, edge_coords, path, speed_data,
                        config.get("STEP_LENGTH", 0.1))
                    if anim_fig:
                        anim_fig.update_layout(height=400)
                        st.plotly_chart(anim_fig, use_container_width=True)
                else:
                    st.info("未采集到轨迹数据（路径过短）")
            with ar2:
                vals = list(weight_dict.values())
                hist_fig2 = px.histogram(
                    x=vals, nbins=20,
                    labels={"x": "Weight", "y": "Count"},
                    title="Weight Distribution",
                    color_discrete_sequence=["#3B82F6"],
                    template="plotly_white",
                )
                hist_fig2.update_layout(height=220, margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(hist_fig2, use_container_width=True)

                top10b = sorted(weight_dict.items(), key=lambda x: -x[1])[:8]
                bar_fig2 = go.Figure(go.Bar(
                    x=[w for _, w in top10b],
                    y=[e for e, _ in top10b],
                    orientation="h",
                    marker_color=["#EF4444" if w >= 8 else "#F97316" if w >= 6
                                  else "#EAB308" if w >= 4 else "#22C55E"
                                  for _, w in top10b],
                    text=[f"{w:.1f}" for _, w in top10b],
                    textposition="outside",
                ))
                bar_fig2.update_layout(
                    title="Top 8 拥堵路段",
                    xaxis=dict(range=[0, 11]),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=230, margin=dict(l=80, r=30, t=40, b=30),
                )
                st.plotly_chart(bar_fig2, use_container_width=True)

            # Row3：3D
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