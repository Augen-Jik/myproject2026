"""Sparse anchor prompt, parsing, and mapping helpers."""

from __future__ import annotations

import re
from typing import Iterable

from roadnet_meta import (
    NORMAL_DEFAULT_WEIGHT,
    COL_MAP,
    COL_NAMES,
    DIR_EN,
    ROW_MAP,
    ROW_NAMES,
    expand_anchor_edges,
    normalize_dir_label,
)
from scenarios import classify_scene_type

LEVEL_TO_WEIGHT = {
    "封闭": 9.5,
    "极度拥堵": 8.8,
    "严重拥堵": 7.5,
    "中度拥堵": 5.0,
    "轻度拥堵": 3.2,
    "畅通": 1.2,
    "正常": 2.0,
    "待判断": 2.0,
}

LEVEL_HINTS = [
    ("封闭", ("全封", "封闭", "管制", "禁行", "禁止通行")),
    ("极度拥堵", ("极度拥堵", "极度", "堵塞")),
    ("严重拥堵", ("严重拥堵", "严重事故", "追尾事故", "追尾", "重度拥堵")),
    ("中度拥堵", ("中度拥堵", "中度", "缓行", "排队", "车多")),
    ("轻度拥堵", ("轻微拥堵", "轻微", "较慢", "轻度")),
    ("畅通", ("畅通", "通畅", "无阻", "顺畅")),
    ("正常", ("正常", "恢复正常", "正常通行")),
]

DIR_TOKENS = [
    ("向东", "向东"),
    ("东向", "向东"),
    ("向西", "向西"),
    ("西向", "向西"),
    ("向北", "向北"),
    ("北向", "向北"),
    ("向南", "向南"),
    ("南向", "向南"),
]

ROAD_PATTERN = "|".join(re.escape(name) for name in list(ROW_NAMES) + list(COL_NAMES))
ANCHOR_PATTERN = re.compile(
    r"ANCHOR\|ROAD=(?P<road>[^|\n]+)\|DIR=(?P<dir>[^|\n]+)\|RANGE=(?P<range>[^|\n]+)\|LEVEL=(?P<level>[^|\n]+)"
)


def strip_legacy_generation_suffix(constraint: str, total_edges: int | None = None) -> str:
    base = str(constraint or "")
    endings = [
        "请严格按格式输出各路段权重（路段ID:数值），用于自动驾驶路径规划。",
        "不要输出解释。",
        "只输出答案。",
    ]
    if total_edges:
        endings.extend(
            [
                f"请为全部{total_edges}条路段生成权重（0-10，越大越拥堵）。",
                f"请按「路段ID:权重值」格式输出所有{total_edges}条路段权重。",
                f"请为郑州市金水区路网的{total_edges}条路段分配拥堵权重，越大越拥堵。",
                f"基于上述路况，为全路网{total_edges}条路段输出权重（0-10），请勿遗漏任何路段。",
            ]
        )
    for ending in endings:
        base = base.replace(ending, "")
    return base.strip().rstrip("。；; ")


def level_to_weight(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
        return max(0.0, min(10.0, parsed))
    except Exception:
        pass
    for label, keywords in LEVEL_HINTS:
        if label == text or any(keyword in text for keyword in keywords):
            return LEVEL_TO_WEIGHT[label]
    return None


def weight_to_level(weight: float | int | None) -> str:
    if weight is None:
        return "待判断"
    value = float(weight)
    if value >= 9.2:
        return "封闭"
    if value >= 8.4:
        return "极度拥堵"
    if value >= 6.8:
        return "严重拥堵"
    if value >= 4.6:
        return "中度拥堵"
    if value >= 2.8:
        return "轻度拥堵"
    if value <= 1.3:
        return "畅通"
    return "正常"


def extract_level_label(clause: str) -> str:
    for label, keywords in LEVEL_HINTS:
        if any(keyword in clause for keyword in keywords):
            return label
    return "待判断"


def extract_range_label(clause: str) -> str:
    if any(token in clause for token in ("全线", "全段", "全程")):
        return "全线"
    match = re.search(rf"({ROAD_PATTERN})[至到]({ROAD_PATTERN})", clause)
    if match:
        return f"{match.group(1)}至{match.group(2)}"
    return "局部"


def extract_time_label(clause: str) -> str:
    patterns = [
        r"当前时间\s*([0-2]?\d[:：][0-5]\d)",
        r"([0-2]?\d[:：][0-5]\d\s*[-~到至]\s*[0-2]?\d[:：][0-5]\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clause)
        if match:
            return match.group(1).replace("：", ":").replace(" ", "")
    for token in ("早高峰", "晚高峰", "平峰", "午间", "夜间", "工作日", "周末", "节假日", "散场"):
        if token in clause:
            return token
    for token in ("09:30后", "恢复正常", "切换", "临时管制"):
        if token in clause:
            return token
    return "未指明"


def extract_propagation_label(clause: str) -> str:
    if any(token in clause for token in ("外溢", "波及", "扩散", "蔓延", "回溢", "传播", "排队至", "影响到")):
        match = re.search(rf"({ROAD_PATTERN})[至到]({ROAD_PATTERN})", clause)
        if match:
            return f"{match.group(1)}至{match.group(2)}"
        return "存在传播/外溢"
    return "无明显传播"


def extract_conflict_label(clause: str) -> str:
    if any(token in clause for token in ("升级", "但", "然而", "同时", "恢复", "切换", "优先", "冲突")):
        return "存在冲突/切换"
    return "单约束"


def infer_event_from_clause(clause: str) -> dict | None:
    clause = str(clause or "").strip()
    if not clause:
        return None
    dir_label = next((canonical for token, canonical in DIR_TOKENS if token in clause), None)
    if dir_label in ("向东", "向西"):
        road = next((name for name in ROW_MAP if name in clause), None)
    elif dir_label in ("向北", "向南"):
        road = next((name for name in COL_MAP if name in clause), None)
    else:
        road = next((name for name in ROW_MAP if name in clause), None)
        if road is None:
            road = next((name for name in COL_MAP if name in clause), None)

    level = extract_level_label(clause)
    if road is None and level == "待判断":
        return None

    return {
        "road": road or "未识别",
        "dir": normalize_dir_label(dir_label),
        "range": extract_range_label(clause),
        "level": level,
        "time": extract_time_label(clause),
        "propagation": extract_propagation_label(clause),
        "conflict": extract_conflict_label(clause),
        "text": clause,
    }


def format_sparse_prompt(
    *,
    scene_type: str,
    events: list[dict],
    narrative: str | None = None,
) -> str:
    lines = [
        "TASK=sparse_output",
        f"SCENE_TYPE={scene_type}",
        "OUTPUT_OBJECT=异常语义锚点",
        "OUTPUT_FORMAT=ANCHOR|ROAD=道路名|DIR=方向|RANGE=范围|LEVEL=等级",
        "OUTPUT_RULE=只输出异常语义锚点；不要补全全量边权；不要解释。",
        "DIR_RULE=ANCHOR中DIR字段须与EVENT中DIR完全一致；不得颠倒方向（向东≠向西，向北≠向南）；无明确方向时填写双向。",
        "SCHEMA=ROAD|DIR|RANGE|LEVEL|TIME|PROPAGATION|CONFLICT",
    ]
    if narrative:
        lines.append(f"NARRATIVE={narrative}")
    for idx, event in enumerate(events, 1):
        lines.append(
            "EVENT_{idx}: ROAD={road} | DIR={dir} | RANGE={range_} | LEVEL={level} | TIME={time} | PROPAGATION={propagation} | CONFLICT={conflict} | TEXT={text}".format(
                idx=idx,
                road=event["road"],
                dir=event["dir"],
                range_=event["range"],
                level=event["level"],
                time=event.get("time", "未指明"),
                propagation=event.get("propagation", "无明显传播"),
                conflict=event.get("conflict", "单约束"),
                text=event["text"],
            )
        )
    lines.append("仅输出锚点，不要解释。")
    return "\n".join(lines)


def build_sparse_task_prompt(constraint: str, *, total_edges: int | None = None, scene_type: str | None = None) -> str:
    base = strip_legacy_generation_suffix(constraint, total_edges=total_edges)
    clauses = [item.strip() for item in re.split(r"[；。;\n]+", base) if item.strip()]
    events = [event for event in (infer_event_from_clause(clause) for clause in clauses) if event]
    if not events:
        events = [
            {
                "road": "未识别",
                "dir": "未指明",
                "range": "全文",
                "level": "待判断",
                "time": "未指明",
                "propagation": "无明显传播",
                "conflict": "单约束",
                "text": base or "无交通描述",
            }
        ]

    scene = scene_type or classify_scene_type(text=base, event_count=len(events))
    return format_sparse_prompt(scene_type=scene, events=events, narrative=base)


def build_anchor_response(event_records: Iterable[dict], *, scene_type: str, include_think: bool = True) -> tuple[str, int]:
    anchors = []
    for record in event_records:
        if record.get("active", True) is False:
            continue
        weight = level_to_weight(record.get("weight", record.get("level")))
        if weight is None:
            continue
        if abs(float(weight) - NORMAL_DEFAULT_WEIGHT) <= 2.0:
            continue
        anchors.append(
            {
                "road": record["road"],
                "dir": normalize_dir_label(record.get("dir")),
                "range": record.get("range", "局部"),
                "level": record.get("level") or weight_to_level(weight),
                "weight": float(weight),
            }
        )
    lines = []
    if include_think:
        lines.extend(
            [
                "<think>",
                f"scene={scene_type}",
                f"anchor_count={len(anchors)}",
                "仅输出异常语义锚点，边级补全由本地映射与图对齐模块处理。",
                "</think>",
            ]
        )
    if anchors:
        for anchor in anchors:
            lines.append(
                "ANCHOR|ROAD={road}|DIR={dir}|RANGE={range_}|LEVEL={level}".format(
                    road=anchor["road"],
                    dir=anchor["dir"],
                    range_=anchor["range"],
                    level=anchor["level"],
                )
            )
    else:
        lines.append("NO_ANCHOR")
    return "\n".join(lines), len(anchors)


def empty_sparse_bundle(scene_type: str | None = None) -> dict:
    return {
        "scene_type": scene_type or "simple_local",
        "anchors": [],
        "anchor_count": 0,
        "mapped_weights": {},
        "edge_confidence": {},
        "parse_confidence": 0.0,
        "conflicts": [],
        "protected_edges": [],
        "parse_fail": False,
    }


def _register_anchor(bundle: dict, anchors_by_key: dict, anchor: dict, source: str) -> None:
    road = str(anchor.get("road") or "").strip()
    if not road or road == "未识别":
        return
    dir_label = normalize_dir_label(anchor.get("dir"))
    range_label = str(anchor.get("range") or "局部").strip() or "局部"
    level = str(anchor.get("level") or "待判断").strip()
    weight = level_to_weight(anchor.get("weight", level))
    if weight is None:
        return
    edges = expand_anchor_edges(road, dir_label, range_label)
    if not edges:
        return

    base_conf = {"structured": 0.94, "clause": 0.78, "fallback": 0.55}.get(source, 0.66)
    if dir_label not in {"未指明", "双向"}:
        base_conf += 0.02
    if range_label not in {"局部", "全文", "未指明"}:
        base_conf += 0.02
    if level not in {"待判断", "正常"}:
        base_conf += 0.02
    confidence = max(0.12, min(0.98, base_conf))

    key = f"{road}|{dir_label}|{range_label}"
    current = anchors_by_key.get(key)
    if current is not None:
        new_strength = abs(float(weight) - NORMAL_DEFAULT_WEIGHT)
        cur_strength = abs(float(current["weight"]) - NORMAL_DEFAULT_WEIGHT)
        if new_strength < cur_strength or (new_strength == cur_strength and confidence <= float(current["confidence"])):
            return
        if round(float(weight), 2) != round(float(current["weight"]), 2):
            bundle["conflicts"].append(
                {
                    "key": key,
                    "kept": round(float(weight), 2),
                    "discarded": round(float(current["weight"]), 2),
                }
            )

    anchors_by_key[key] = {
        "road": road,
        "dir": dir_label,
        "range": range_label,
        "level": level or weight_to_level(weight),
        "weight": round(float(weight), 2),
        "confidence": round(confidence, 3),
        "source": source,
        "edges": list(edges),
        "text": anchor.get("text", ""),
    }

    for eid in edges:
        old_weight = bundle["mapped_weights"].get(eid, NORMAL_DEFAULT_WEIGHT)
        if eid not in bundle["mapped_weights"] or abs(float(weight) - NORMAL_DEFAULT_WEIGHT) > abs(old_weight - NORMAL_DEFAULT_WEIGHT):
            bundle["mapped_weights"][eid] = round(float(weight), 2)
        bundle["edge_confidence"][eid] = max(bundle["edge_confidence"].get(eid, 0.0), confidence)
        if float(weight) >= 9.0:
            bundle["protected_edges"].append(eid)


def parse_sparse_output_bundle(raw: str, *, scene_type: str | None = None) -> dict:
    bundle = empty_sparse_bundle(scene_type=scene_type or classify_scene_type(text=raw or ""))
    if not raw:
        bundle["parse_fail"] = True
        return bundle

    anchors_by_key: dict[str, dict] = {}
    struct_text = raw.split("</think>")[-1].strip() if "</think>" in raw else str(raw).strip()

    for match in ANCHOR_PATTERN.finditer(struct_text):
        _register_anchor(
            bundle,
            anchors_by_key,
            {
                "road": match.group("road").strip(),
                "dir": match.group("dir").strip(),
                "range": match.group("range").strip(),
                "level": match.group("level").strip(),
                "text": match.group(0),
            },
            source="structured",
        )

    for clause in [piece.strip() for piece in re.split(r"[；。\n;]+", struct_text) if piece.strip()]:
        anchor = infer_event_from_clause(clause)
        if anchor:
            _register_anchor(bundle, anchors_by_key, anchor, source="clause")

    bundle["anchors"] = list(anchors_by_key.values())
    bundle["anchor_count"] = len(bundle["anchors"])
    bundle["protected_edges"] = sorted(set(bundle["protected_edges"]))

    if bundle["anchors"]:
        avg_conf = sum(item["confidence"] for item in bundle["anchors"]) / len(bundle["anchors"])
        penalty = min(0.08 * len(bundle["conflicts"]), 0.24)
        bundle["parse_confidence"] = round(max(0.05, min(0.98, avg_conf - penalty)), 3)
    elif "NO_ANCHOR" in struct_text:
        bundle["parse_confidence"] = 0.9
    else:
        bundle["parse_fail"] = True
    return bundle


def validate_sparse_response(text: str) -> tuple[bool, list[str]]:
    issues = []
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        issues.append("empty")
        return False, issues
    if any("R0C0_E:" in line or "C0R0_N:" in line for line in lines):
        issues.append("contains_full_output")
    has_anchor = any(line.startswith("ANCHOR|ROAD=") for line in lines) or any(line == "NO_ANCHOR" for line in lines)
    if not has_anchor:
        issues.append("missing_anchor")
    for line in lines:
        if line.startswith("ANCHOR|ROAD=") and not ANCHOR_PATTERN.fullmatch(line):
            issues.append(f"bad_anchor:{line}")
    return len(issues) == 0, issues
