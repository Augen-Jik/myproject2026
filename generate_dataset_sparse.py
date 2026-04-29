#!/usr/bin/env python3
"""Generate Sparse-LoRA-v2 datasets on the fixed 96-edge SUMO road network."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CODE_DIR = "/root/autodl-tmp/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from roadnet_meta import (  # noqa: E402
    COL_NAMES,
    ROW_NAMES,
    NORMAL_DEFAULT_WEIGHT,
    load_roadnet_meta,
)
from scenarios import classify_scene_type, scene_bucket  # noqa: E402
from sparse_utils import (  # noqa: E402
    build_anchor_response,
    format_sparse_prompt,
    level_to_weight,
    parse_sparse_output_bundle,
    validate_sparse_response,
)


LEVEL_DESCS = {
    "封闭": ["施工全封闭", "临时封闭管制", "事故处置封闭"],
    "极度拥堵": ["多车事故导致极度拥堵", "车流完全淤积"],
    "严重拥堵": ["追尾事故严重拥堵", "车流积压严重拥堵", "排队长度明显增加"],
    "中度拥堵": ["中度拥堵缓行", "排队通行效率下降", "车流较大缓行"],
    "轻度拥堵": ["轻度拥堵", "车辆较多但仍可缓慢通过"],
    "畅通": ["畅通无阻", "车流顺畅"],
    "正常": ["保持正常通行", "交通基本正常"],
}

GLOBAL_SCENE_WEIGHTS = {
    "simple_local": 0.25,
    "directional_asymmetry": 0.20,
    "core_blockage": 0.25,
    "compound_disaster": 0.12,
    "temporal_switch": 0.09,
    "propagation_range": 0.09,
}

GLOBAL_LENGTH_WEIGHTS = {
    "short": 0.35,
    "medium": 0.35,
    "long": 0.15,
    "noisy_long": 0.15,
}

STAGE_POLICIES = {
    "stage1": {
        "scene_weights": {
            "simple_local": 0.56,
            "directional_asymmetry": 0.44,
        },
        "length_weights": {
            "short": 0.46,
            "medium": 0.44,
            "long": 0.07,
            "noisy_long": 0.03,
        },
        "train_size": 2000,
        "eval_size": 260,
    },
    "stage2": {
        "scene_weights": dict(GLOBAL_SCENE_WEIGHTS),
        "length_weights": {
            "short": 0.24,
            "medium": 0.41,
            "long": 0.19,
            "noisy_long": 0.16,
        },
        "train_size": 4200,
        "eval_size": 420,
    },
    "stage3": {
        "scene_weights": {
            "core_blockage": 0.24,
            "compound_disaster": 0.30,
            "temporal_switch": 0.22,
            "propagation_range": 0.24,
        },
        "length_weights": {
            "short": 0.00,
            "medium": 0.22,
            "long": 0.38,
            "noisy_long": 0.40,
        },
        "train_size": 1200,
        "eval_size": 180,
    },
}

VALIDATION_POLICIES = {
    "val_normal": {
        "scene_weights": {"simple_local": 0.75, "directional_asymmetry": 0.25},
        "length_weights": {"short": 0.45, "medium": 0.45, "long": 0.08, "noisy_long": 0.02},
        "size": 180,
    },
    "val_special": {
        "scene_weights": {
            "core_blockage": 0.30,
            "directional_asymmetry": 0.20,
            "compound_disaster": 0.18,
            "temporal_switch": 0.16,
            "propagation_range": 0.16,
        },
        "length_weights": {"short": 0.10, "medium": 0.42, "long": 0.24, "noisy_long": 0.24},
        "size": 180,
    },
    "val_anti_truncation": {
        "scene_weights": {
            "compound_disaster": 0.32,
            "temporal_switch": 0.34,
            "propagation_range": 0.34,
        },
        "length_weights": {"short": 0.00, "medium": 0.08, "long": 0.42, "noisy_long": 0.50},
        "size": 180,
    },
}


class SparseDatasetBuilder:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.meta = load_roadnet_meta()
        self.edge_ids = list(self.meta.edge_ids)
        self.edge_meta = self.meta.edge_meta
        self.neighbors = self.meta.line_graph_neighbors
        self.valid_edges = set(self.edge_ids)
        self.length_noise = [
            "气象部门提示傍晚仍有小雨，但目前并未形成新增积水。",
            "交警提醒周边停车场正在分流车辆，不过主路仍可通行。",
            "导航播报中还提到相邻商圈有出库车流，但不直接改变本段主判断。",
            "有市民反映支路口偶尔压车，不过主约束仍以当前干道情况为准。",
        ]
        self.scene_prefix = {
            "simple_local": "常态或局部短时事件",
            "directional_asymmetry": "方向不对称约束",
            "core_blockage": "核心通道阻断",
            "compound_disaster": "多事件复合灾害",
            "temporal_switch": "时段切换约束",
            "propagation_range": "传播范围与外溢",
        }

    def _choice(self, mapping: dict[str, float]) -> str:
        keys = list(mapping)
        weights = [mapping[key] for key in keys]
        return self.rng.choices(keys, weights=weights, k=1)[0]

    def _pick_edge(self, *, axis: str | None = None, dir_code: str | None = None, preferred_roads: list[str] | None = None) -> tuple[str, dict]:
        candidates = []
        for eid in self.edge_ids:
            meta = self.edge_meta[eid]
            if axis and meta["axis"] != axis:
                continue
            if dir_code and meta["dir_code"] != dir_code:
                continue
            if preferred_roads and meta["road"] not in preferred_roads:
                continue
            candidates.append(eid)
        if not candidates:
            candidates = list(self.edge_ids)
        eid = self.rng.choice(candidates)
        return eid, self.edge_meta[eid]

    def _range_label(self, meta: dict, scope: str) -> tuple[str, list[str]]:
        if scope == "full_line":
            label = "全线"
            edges = [eid for eid in self.edge_ids if self.edge_meta[eid]["road"] == meta["road"] and self.edge_meta[eid]["dir_code"] == meta["dir_code"]]
            return label, edges

        if meta["axis"] == "H":
            lo = meta["col"]
            hi = meta["col"] + 1
            if scope == "range":
                lo = max(0, meta["col"] - self.rng.randint(0, 1))
                hi = min(len(COL_NAMES) - 1, meta["col"] + self.rng.randint(1, 2))
                if hi <= lo:
                    hi = min(len(COL_NAMES) - 1, lo + 1)
            label = f"{COL_NAMES[lo]}至{COL_NAMES[hi]}"
            edges = [f"R{meta['row']}C{idx}_{meta['dir_code']}" for idx in range(lo, hi)]
        else:
            lo = meta["row"]
            hi = meta["row"] + 1
            if scope == "range":
                lo = max(0, meta["row"] - self.rng.randint(0, 1))
                hi = min(len(ROW_NAMES) - 1, meta["row"] + self.rng.randint(1, 2))
                if hi <= lo:
                    hi = min(len(ROW_NAMES) - 1, lo + 1)
            label = f"{ROW_NAMES[lo]}至{ROW_NAMES[hi]}"
            edges = [f"C{meta['col']}R{idx}_{meta['dir_code']}" for idx in range(lo, hi)]

        return label, [eid for eid in edges if eid in self.valid_edges]

    def _make_event(
        self,
        meta: dict,
        *,
        level: str,
        scope: str = "segment",
        time_label: str = "当前",
        propagation: str = "无明显传播",
        conflict: str = "单约束",
        active: bool = True,
        custom_text: str | None = None,
        propagation_strength: float = 0.0,
    ) -> dict:
        range_label, anchor_edges = self._range_label(meta, scope)
        desc = custom_text
        if not desc:
            seg_desc = "全线" if range_label == "全线" else f"{range_label}段"
            desc = f"{meta['road']}{seg_desc}{meta['dir']}{self.rng.choice(LEVEL_DESCS[level])}"
        return {
            "road": meta["road"],
            "dir": meta["dir"],
            "range": range_label,
            "level": level,
            "weight": level_to_weight(level),
            "time": time_label,
            "propagation": propagation,
            "conflict": conflict,
            "text": desc,
            "anchor_edges": list(anchor_edges),
            "active": active,
            "propagation_strength": propagation_strength,
        }

    def _apply_event(self, weights: dict[str, float], record: dict) -> None:
        if not record.get("active", True):
            return
        base_weight = float(record["weight"])
        for eid in record.get("anchor_edges", []):
            prev = weights.get(eid, NORMAL_DEFAULT_WEIGHT)
            if abs(base_weight - NORMAL_DEFAULT_WEIGHT) >= abs(prev - NORMAL_DEFAULT_WEIGHT):
                weights[eid] = round(base_weight, 2)
            prop = float(record.get("propagation_strength", 0.0) or 0.0)
            if prop <= 0:
                continue
            for nb in self.neighbors.get(eid, ()):
                if nb in record.get("anchor_edges", []):
                    continue
                candidate = NORMAL_DEFAULT_WEIGHT + (base_weight - NORMAL_DEFAULT_WEIGHT) * prop
                prev_nb = weights.get(nb, NORMAL_DEFAULT_WEIGHT)
                if abs(candidate - NORMAL_DEFAULT_WEIGHT) > abs(prev_nb - NORMAL_DEFAULT_WEIGHT):
                    weights[nb] = round(candidate, 2)

    def _render_constraint(self, scene_type: str, records: list[dict], length_bucket: str) -> str:
        clauses = [rec["text"] for rec in records]
        if length_bucket == "short":
            return "；".join(clauses[:2])

        if length_bucket == "medium":
            prefix = f"{self.scene_prefix.get(scene_type, '交通场景')}："
            return prefix + "；".join(clauses)

        base = [
            f"当前需要根据{self.scene_prefix.get(scene_type, '交通场景')}进行路径规划。",
            "请优先按当前生效约束理解，不要被已解除或背景信息误导。",
            "；".join(clauses),
        ]
        if scene_type == "temporal_switch":
            base.append("若描述里同时出现历史状态和当前状态，应以当前时段为准。")
        elif scene_type == "propagation_range":
            base.append("若有排队外溢，应注意主事件与波及范围不是同一强度。")
        elif scene_type == "directional_asymmetry":
            base.append("若同一路存在双向差异，应严格区分方向。")

        if length_bucket == "long":
            base.append("其余未明确提及路段按常态理解，不要额外扩大影响范围。")
            return " ".join(base)

        noisy = list(base)
        noisy.append(self.rng.choice(self.length_noise))
        noisy.append(self.rng.choice(self.length_noise))
        noisy.append("部分播报会提到周边商圈或停车场信息，但主判断仍以道路方向、区间、时段和传播范围为核心。")
        return " ".join(noisy)

    def _simple_local(self, length_bucket: str) -> dict:
        if self.rng.random() < 0.38:
            eid, meta = self._pick_edge()
            normal_level = self.rng.choice(["正常", "畅通"])
            record = self._make_event(meta, level=normal_level, scope=self.rng.choice(["segment", "range"]), active=True)
            records = [record]
        else:
            eid, meta = self._pick_edge()
            records = [
                self._make_event(
                    meta,
                    level=self.rng.choice(["中度拥堵", "严重拥堵"]),
                    scope=self.rng.choice(["segment", "range"]),
                    propagation="无明显传播",
                    propagation_strength=0.18,
                )
            ]
            if self.rng.random() < 0.55:
                _, relief_meta = self._pick_edge(axis="V" if meta["axis"] == "H" else "H")
                records.append(
                    self._make_event(relief_meta, level=self.rng.choice(["正常", "畅通"]), scope="segment", active=True)
                )
        return {"scene_type": "simple_local", "records": records}

    def _directional_asymmetry(self, length_bucket: str) -> dict:
        axis = self.rng.choice(["H", "V"])
        if axis == "H":
            dir_a, dir_b = "E", "W"
        else:
            dir_a, dir_b = "N", "S"
        _, meta_a = self._pick_edge(axis=axis, dir_code=dir_a)
        meta_b = dict(meta_a)
        meta_b["dir_code"] = dir_b
        meta_b["dir"] = {"E": "向东", "W": "向西", "N": "向北", "S": "向南"}[dir_b]
        records = [
            self._make_event(
                meta_a,
                level=self.rng.choice(["严重拥堵", "封闭"]),
                scope=self.rng.choice(["segment", "range"]),
                propagation_strength=0.16,
            ),
            self._make_event(meta_b, level=self.rng.choice(["正常", "畅通"]), scope=self.rng.choice(["segment", "range"])),
        ]
        if self.rng.random() < 0.45:
            _, aux = self._pick_edge(axis="V" if axis == "H" else "H")
            records.append(self._make_event(aux, level="中度拥堵", scope="segment"))
        return {"scene_type": "directional_asymmetry", "records": records}

    def _core_blockage(self, length_bucket: str) -> dict:
        preferred = ["经六路", "花园路", "黄河路"]
        _, meta = self._pick_edge(preferred_roads=preferred)
        records = [
            self._make_event(
                meta,
                level="封闭",
                scope=self.rng.choice(["range", "full_line"]),
                propagation="存在传播/外溢",
                propagation_strength=0.32,
            )
        ]
        _, relief = self._pick_edge(axis="V" if meta["axis"] == "H" else "H")
        records.append(self._make_event(relief, level=self.rng.choice(["正常", "畅通"]), scope="range"))
        if self.rng.random() < 0.35:
            _, spill = self._pick_edge(axis="V" if meta["axis"] == "H" else "H")
            records.append(
                self._make_event(
                    spill,
                    level="中度拥堵",
                    scope="range",
                    propagation="受核心封锁波及",
                    conflict="主堵点外溢",
                    propagation_strength=0.18,
                )
            )
        return {"scene_type": "core_blockage", "records": records}

    def _compound_disaster(self, length_bucket: str) -> dict:
        records = []
        _, block = self._pick_edge(preferred_roads=["经六路", "花园路", "黄河路"])
        records.append(self._make_event(block, level="封闭", scope=self.rng.choice(["range", "full_line"]), propagation_strength=0.34))
        _, heavy = self._pick_edge()
        records.append(self._make_event(heavy, level="严重拥堵", scope="range", propagation_strength=0.22))
        _, medium = self._pick_edge()
        records.append(self._make_event(medium, level="中度拥堵", scope=self.rng.choice(["segment", "range"]), propagation_strength=0.14))
        if self.rng.random() < 0.55:
            _, aux = self._pick_edge()
            records.append(self._make_event(aux, level=self.rng.choice(["正常", "畅通"]), scope="segment"))
        return {"scene_type": "compound_disaster", "records": records}

    def _temporal_switch(self, length_bucket: str) -> dict:
        _, meta = self._pick_edge()
        current_time = self.rng.choice(["08:20", "08:45", "10:15", "10:40"])
        active_now = current_time in {"08:20", "08:45"}
        recovery_text = "09:30后恢复正常" if current_time.startswith("08") else f"当前时间{current_time}，已恢复正常"
        records = [
            self._make_event(
                meta,
                level="严重拥堵",
                scope=self.rng.choice(["segment", "range"]),
                time_label="07:00-09:00",
                conflict="历史状态/时段切换",
                active=active_now,
                custom_text=f"早高峰07:00-09:00{meta['road']}{meta['dir']}{self.rng.choice(LEVEL_DESCS['严重拥堵'])}",
                propagation_strength=0.20,
            ),
            self._make_event(
                meta,
                level="正常",
                scope=self.rng.choice(["segment", "range"]),
                time_label=f"当前时间{current_time}" if not active_now else "09:30后",
                conflict="当前时段生效",
                active=not active_now,
                custom_text=f"{meta['road']}{meta['dir']}{recovery_text}",
                propagation_strength=0.0,
            ),
        ]
        return {"scene_type": "temporal_switch", "records": records}

    def _propagation_range(self, length_bucket: str) -> dict:
        _, primary = self._pick_edge(preferred_roads=["经六路", "花园路", "黄河路"])
        _, spill = self._pick_edge(axis="V" if primary["axis"] == "H" else "H")
        records = [
            self._make_event(
                primary,
                level=self.rng.choice(["封闭", "严重拥堵"]),
                scope=self.rng.choice(["range", "full_line"]),
                propagation="主事件",
                conflict="单约束",
                propagation_strength=0.36,
            ),
            self._make_event(
                spill,
                level="中度拥堵",
                scope="range",
                propagation=f"{spill['road']}受排队外溢影响",
                conflict="传播范围",
                propagation_strength=0.20,
            ),
        ]
        if self.rng.random() < 0.4:
            _, relief = self._pick_edge()
            records.append(self._make_event(relief, level="正常", scope="segment"))
        return {"scene_type": "propagation_range", "records": records}

    def _build_scene(self, scene_type: str, length_bucket: str) -> dict:
        if scene_type == "simple_local":
            return self._simple_local(length_bucket)
        if scene_type == "directional_asymmetry":
            return self._directional_asymmetry(length_bucket)
        if scene_type == "core_blockage":
            return self._core_blockage(length_bucket)
        if scene_type == "compound_disaster":
            return self._compound_disaster(length_bucket)
        if scene_type == "temporal_switch":
            return self._temporal_switch(length_bucket)
        return self._propagation_range(length_bucket)

    def _make_sample(self, split_name: str, scene_type: str, length_bucket: str) -> dict | None:
        scene = self._build_scene(scene_type, length_bucket)
        records = scene["records"]
        narrative = self._render_constraint(scene["scene_type"], records, length_bucket)
        prompt = format_sparse_prompt(
            scene_type=scene["scene_type"],
            events=records,
            narrative=narrative,
        )
        response, anchor_count = build_anchor_response(records, scene_type=scene["scene_type"])
        ok, issues = validate_sparse_response(response)
        if not ok:
            return None

        parsed = parse_sparse_output_bundle(response, scene_type=scene["scene_type"])
        if parsed.get("parse_fail"):
            return None
        if parsed.get("anchor_count", 0) != anchor_count:
            return None
        if parsed.get("conflicts"):
            return None
        if parsed.get("parse_confidence", 0.0) < 0.72:
            return None

        weights = {eid: NORMAL_DEFAULT_WEIGHT for eid in self.edge_ids}
        for record in records:
            self._apply_event(weights, record)

        if anchor_count == 0 and any(abs(weight - NORMAL_DEFAULT_WEIGHT) > 2.0 for weight in weights.values()):
            return None

        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "scene_type": scene["scene_type"],
            "scene_bucket": scene_bucket(scene["scene_type"]),
            "length_bucket": length_bucket,
            "split": split_name,
            "constraint_text": narrative,
            "prompt_text": prompt,
            "response_text": response,
            "anchor_count": anchor_count,
            "ground_truth_json": json.dumps(weights, ensure_ascii=False),
            "event_records_json": json.dumps(records, ensure_ascii=False),
            "parse_confidence": parsed.get("parse_confidence", 0.0),
            "total_edges": len(self.edge_ids),
        }

    def _scene_for_length(self, policy_scene_weights: dict[str, float], length_bucket: str) -> str:
        weights = dict(policy_scene_weights)
        if length_bucket in {"long", "noisy_long"}:
            boost = {"compound_disaster", "temporal_switch", "propagation_range"}
            for key in list(weights):
                if key in boost:
                    weights[key] *= 1.45
                else:
                    weights[key] *= 0.82
        total = sum(weights.values())
        normalized = {key: value / total for key, value in weights.items() if value > 0}
        return self._choice(normalized)

    def generate_split(self, split_name: str, *, size: int, scene_weights: dict[str, float], length_weights: dict[str, float]) -> list[dict]:
        samples = []
        retries = 0
        while len(samples) < size:
            length_bucket = self._choice(length_weights)
            scene_type = self._scene_for_length(scene_weights, length_bucket)
            sample = self._make_sample(split_name, scene_type, length_bucket)
            if sample is None:
                retries += 1
                if retries > size * 25:
                    raise RuntimeError(f"{split_name} 生成失败过多，请检查生成逻辑")
                continue
            samples.append(sample)
        return samples


def save_parquet(samples: list[dict], out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = {
        "messages": [json.dumps(sample["messages"], ensure_ascii=False) for sample in samples],
        "scene_type": [sample["scene_type"] for sample in samples],
        "scene_bucket": [sample["scene_bucket"] for sample in samples],
        "length_bucket": [sample["length_bucket"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "constraint_text": [sample["constraint_text"] for sample in samples],
        "prompt_text": [sample["prompt_text"] for sample in samples],
        "response_text": [sample["response_text"] for sample in samples],
        "anchor_count": [sample["anchor_count"] for sample in samples],
        "ground_truth_json": [sample["ground_truth_json"] for sample in samples],
        "event_records_json": [sample["event_records_json"] for sample in samples],
        "parse_confidence": [sample["parse_confidence"] for sample in samples],
        "total_edges": [sample["total_edges"] for sample in samples],
    }
    table = pa.table(rows)
    pq.write_table(table, os.path.join(out_dir, "data.parquet"))


def summarize_samples(samples: list[dict]) -> dict:
    summary = {}
    summary["size"] = len(samples)
    summary["scene_type"] = dict(Counter(sample["scene_type"] for sample in samples))
    summary["scene_bucket"] = dict(Counter(sample["scene_bucket"] for sample in samples))
    summary["length_bucket"] = dict(Counter(sample["length_bucket"] for sample in samples))
    summary["avg_anchor_count"] = round(sum(sample["anchor_count"] for sample in samples) / max(len(samples), 1), 3)
    summary["avg_parse_confidence"] = round(sum(sample["parse_confidence"] for sample in samples) / max(len(samples), 1), 3)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=5600)
    parser.add_argument("--eval-size", type=int, default=840)
    args = parser.parse_args()

    builder = SparseDatasetBuilder(seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}

    generated["train"] = builder.generate_split(
        "train",
        size=args.train_size,
        scene_weights=GLOBAL_SCENE_WEIGHTS,
        length_weights=GLOBAL_LENGTH_WEIGHTS,
    )
    generated["eval"] = builder.generate_split(
        "eval",
        size=args.eval_size,
        scene_weights=GLOBAL_SCENE_WEIGHTS,
        length_weights={"short": 0.22, "medium": 0.38, "long": 0.20, "noisy_long": 0.20},
    )

    for stage_name, policy in STAGE_POLICIES.items():
        generated[f"{stage_name}_train"] = builder.generate_split(
            f"{stage_name}_train",
            size=policy["train_size"],
            scene_weights=policy["scene_weights"],
            length_weights=policy["length_weights"],
        )
        generated[f"{stage_name}_eval"] = builder.generate_split(
            f"{stage_name}_eval",
            size=policy["eval_size"],
            scene_weights=policy["scene_weights"],
            length_weights=policy["length_weights"],
        )

    for split_name, policy in VALIDATION_POLICIES.items():
        generated[split_name] = builder.generate_split(
            split_name,
            size=policy["size"],
            scene_weights=policy["scene_weights"],
            length_weights=policy["length_weights"],
        )

    save_parquet(generated["train"], str(out_dir / "train"))
    save_parquet(generated["eval"], str(out_dir / "eval"))
    for stage_name in STAGE_POLICIES:
        save_parquet(generated[f"{stage_name}_train"], str(out_dir / stage_name / "train"))
        save_parquet(generated[f"{stage_name}_eval"], str(out_dir / stage_name / "eval"))
    for split_name in VALIDATION_POLICIES:
        save_parquet(generated[split_name], str(out_dir / split_name))

    for name, items in generated.items():
        summary[name] = summarize_samples(items)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Sparse-LoRA-v2 数据集生成完成")
    print(f"输出目录: {out_dir}")
    print(f"主路网边数: {len(builder.edge_ids)}")
    print("=" * 72)
    for name in ["train", "eval", "stage1_train", "stage2_train", "stage3_train", "val_normal", "val_special", "val_anti_truncation"]:
        stats = summary[name]
        print(f"[{name}] size={stats['size']} avg_anchor={stats['avg_anchor_count']:.2f} avg_conf={stats['avg_parse_confidence']:.2f}")
        print(f"  scene_type={stats['scene_type']}")
        print(f"  length_bucket={stats['length_bucket']}")


if __name__ == "__main__":
    main()
