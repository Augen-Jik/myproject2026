"""Shared scene taxonomy, presets, and lightweight classifiers."""

from __future__ import annotations

import re

from sim_scene_profiles import resolve_scene_profile_name

SCENE_TYPES = (
    "normal_baseline",
    "simple_local",
    "directional_asymmetry",
    "core_blockage",
    "compound_disaster",
    "temporal_switch",
    "propagation_range",
)

SCENE_BUCKETS = {
    "normal_baseline": "baseline",
    "simple_local": "simple_local",
    "directional_asymmetry": "directional_asymmetry",
    "core_blockage": "core_blockage",
    "compound_disaster": "complex_spatiotemporal",
    "temporal_switch": "complex_spatiotemporal",
    "propagation_range": "complex_spatiotemporal",
}

APP_SCENARIO_PRESETS = [
    {
        "id": "N",
        "name": "场景N normal_baseline 正常通行",
        "scene_type": "normal_baseline",
        "scene_profile": "normal_baseline",
        "scene_bucket": "baseline",
        "start": "R3C0_E",
        "end": "R3C4_E",
        "constraint": "当前路网整体通行正常，无明显拥堵或事故；各主干道保持基线交通状态。",
    },
    {
        "id": "A",
        "name": "场景A 黄河路单向拥堵",
        "scene_type": "simple_local",
        "scene_profile": "simple_local",
        "scene_bucket": "simple_local",
        "start": "R3C0_E",
        "end": "R3C4_E",
        "constraint": "黄河路经一路至经三路段向东因追尾事故严重拥堵；花园路向北畅通无阻。",
    },
    {
        "id": "B",
        "name": "场景B 花园路封闭绕行",
        "scene_type": "core_blockage",
        "scene_profile": "core_blockage",
        "scene_bucket": "core_blockage",
        "start": "C4R0_N",
        "end": "C4R3_N",
        "constraint": "花园路农业路至红专路段向北施工封闭；经六路向北畅通无阻。",
    },
    {
        "id": "C",
        "name": "场景C 多路段复合拥堵",
        "scene_type": "compound_disaster",
        "scene_profile": "compound_disaster",
        "scene_bucket": "complex_spatiotemporal",
        "start": "R0C0_E",
        "end": "R3C4_E",
        "constraint": "经六路全线封闭施工；花园路向北严重拥堵；黄河路向东中度拥堵。",
    },
    {
        "id": "D",
        "name": "场景D 中央核心节点封锁",
        "scene_type": "core_blockage",
        "scene_profile": "core_blockage",
        "scene_bucket": "core_blockage",
        "start": "R0C0_E",
        "end": "R4C3_E",
        "constraint": "经六路农业路至黄河路段全线封闭；花园路政七街至黄河路段完全封闭；经三路向南严重拥堵；经八路向北正常通行。",
    },
    {
        "id": "E",
        "name": "场景E 方向不对称拥堵",
        "scene_type": "directional_asymmetry",
        "scene_profile": "directional_asymmetry",
        "scene_bucket": "directional_asymmetry",
        "start": "R4C0_E",
        "end": "R4C4_E",
        "constraint": "纬五路向东方向因事故严重拥堵，向西方向畅通无阻；未来路向南中度拥堵。",
    },
    {
        "id": "F",
        "name": "场景F 时段切换管制",
        "scene_type": "temporal_switch",
        "scene_profile": "temporal_switch",
        "scene_bucket": "complex_spatiotemporal",
        "start": "R3C0_E",
        "end": "R3C4_E",
        "constraint": "早高峰07:00-09:00黄河路向东严重拥堵，09:30后恢复正常；当前时间08:30，请按当前时段规划。",
    },
    {
        "id": "G",
        "name": "场景G 传播范围外溢",
        "scene_type": "propagation_range",
        "scene_profile": "propagation_range",
        "scene_bucket": "complex_spatiotemporal",
        "start": "R0C0_E",
        "end": "C4R3_N",
        "constraint": "经六路农业路至黄河路段施工封闭，排队外溢至花园路红专路至黄河路段向北中度拥堵；其余方向保持正常。",
    },
    {
        "id": "H",
        "name": "场景H 活动管制+方向不对称",
        "scene_type": "directional_asymmetry",
        "scene_profile": "directional_asymmetry",
        "scene_bucket": "complex_spatiotemporal",
        "start": "R4C0_E",
        "end": "R4C4_E",
        "constraint": "演唱会散场临时管制：纬五路向东封闭，向西保持畅通；未来路向南因接驳车排队中度拥堵。",
    },
    {
        "id": "I",
        "name": "场景I 核心封锁+外溢",
        "scene_type": "core_blockage",
        "scene_profile": "compound_disaster",
        "scene_bucket": "complex_spatiotemporal",
        "start": "R0C0_E",
        "end": "R4C4_E",
        "constraint": "经六路红专路至黄河路段双向完全封闭；花园路政七街至黄河路段向北排队外溢形成中度拥堵；经三路向南严重拥堵。",
    },
]

SPATIOTEMPORAL_PROMPT_CASES = [
    {
        "id": "ST1",
        "name": "时空对照1 方向不对称",
        "focus": "directional_asymmetry",
        "scene_type": "directional_asymmetry",
        "start": "R4C0_E",
        "end": "R4C4_E",
        "constraint": "纬五路向东方向因事故严重拥堵，向西方向畅通无阻；未来路向南中度拥堵。",
        "ground_truth": {
            "R4C0_E": 8.5, "R4C1_E": 8.0, "R4C2_E": 7.5, "R4C3_E": 7.0,
            "R4C0_W": 1.2, "R4C1_W": 1.2, "R4C2_W": 1.2, "R4C3_W": 1.2,
            "C5R0_S": 5.5, "C5R1_S": 5.0, "C5R2_S": 4.5,
        },
        "focus_edges": ["R4C0_E", "R4C1_E", "R4C2_E", "R4C3_E"],
        "forbidden_edges": ["R4C0_W", "R4C1_W", "R4C2_W", "R4C3_W"],
    },
    {
        "id": "ST2",
        "name": "时空对照2 时段切换",
        "focus": "temporal_switch",
        "scene_type": "temporal_switch",
        "start": "R3C0_E",
        "end": "R3C4_E",
        "constraint": "早高峰07:00-09:00黄河路向东严重拥堵，09:30后恢复正常；当前时间08:30，请按当前时段规划。",
        "ground_truth": {
            "R3C0_E": 8.5, "R3C1_E": 8.0, "R3C2_E": 7.5, "R3C3_E": 7.0,
        },
        "focus_edges": ["R3C0_E", "R3C1_E", "R3C2_E", "R3C3_E"],
        "forbidden_edges": ["R3C0_W", "R3C1_W", "R3C2_W", "R3C3_W"],
        "expected_time_hint": "08:30",
    },
    {
        "id": "ST3",
        "name": "时空对照3 传播范围",
        "focus": "propagation_range",
        "scene_type": "propagation_range",
        "start": "R0C0_E",
        "end": "C4R3_N",
        "constraint": "经六路农业路至黄河路段施工封闭，排队外溢至花园路红专路至黄河路段向北中度拥堵；其余方向保持正常。",
        "ground_truth": {
            "C2R0_N": 9.5, "C2R1_N": 9.5, "C2R2_N": 9.5,
            "C4R1_N": 5.5, "C4R2_N": 5.0,
        },
        "focus_edges": ["C2R0_N", "C2R1_N", "C2R2_N", "C4R1_N", "C4R2_N"],
        "forbidden_edges": ["C4R1_S", "C4R2_S"],
    },
    {
        "id": "ST4",
        "name": "时空对照4 多约束冲突",
        "focus": "multi_constraint_conflict",
        "scene_type": "compound_disaster",
        "start": "R3C0_E",
        "end": "R3C4_E",
        "constraint": "黄河路向东中度拥堵；但经一路至经六路段因事故升级为严重拥堵；花园路向北正常通行。",
        "ground_truth": {
            "R3C0_E": 7.5, "R3C1_E": 7.5,
            "R3C2_E": 5.0, "R3C3_E": 5.0, "R3C4_E": 5.0,
            "C4R0_N": 2.0, "C4R1_N": 2.0,
        },
        "focus_edges": ["R3C0_E", "R3C1_E", "R3C2_E", "R3C3_E", "R3C4_E"],
        "forbidden_edges": ["R3C0_W", "R3C1_W", "R3C2_W", "R3C3_W"],
    },
]

SPECIAL_STRESS_SCENARIOS = [
    {
        "id": "S1",
        "name": "特殊场景S1 核心封锁+外溢排队",
        "scene_type": "core_blockage",
        "scene_bucket": "special",
        "stress_tags": ["core_blockage", "spillover"],
        "start": "R0C0_E",
        "end": "R4C4_E",
        "desc": "经六路红专路至黄河路段双向完全封闭；花园路政七街至黄河路段向北排队外溢形成中度拥堵；经三路向南严重拥堵。",
        "ground_truth": {
            "C2R1_N": 9.5, "C2R1_S": 9.5, "C2R2_N": 9.5, "C2R2_S": 9.5,
            "C4R2_N": 5.5, "C4R1_N": 5.0,
            "C1R1_S": 7.5, "C1R2_S": 7.0,
        },
    },
    {
        "id": "S2",
        "name": "特殊场景S2 活动管制+方向不对称",
        "scene_type": "directional_asymmetry",
        "scene_bucket": "special",
        "stress_tags": ["event_control", "directional_asymmetry"],
        "start": "R4C0_E",
        "end": "R4C4_E",
        "desc": "演唱会散场管制：纬五路向东临时封闭，向西保持畅通；未来路向南因接驳车排队中度拥堵。",
        "ground_truth": {
            "R4C0_E": 9.5, "R4C1_E": 9.5, "R4C2_E": 9.5, "R4C3_E": 9.5,
            "R4C0_W": 1.2, "R4C1_W": 1.2, "R4C2_W": 1.2, "R4C3_W": 1.2,
            "C5R0_S": 5.5, "C5R1_S": 5.0, "C5R2_S": 4.5,
        },
    },
    {
        "id": "S3",
        "name": "特殊场景S3 传播范围+拓扑回溢",
        "scene_type": "propagation_range",
        "scene_bucket": "special",
        "stress_tags": ["propagation_range", "spillover"],
        "start": "R0C0_E",
        "end": "R3C4_E",
        "desc": "黄河路经三路口向东事故点严重拥堵，仅局部明确；排队回溢至经三路向南中度拥堵，经六路向北保持畅通。",
        "ground_truth": {
            "R3C1_E": 8.5, "R3C2_E": 7.5,
            "C1R1_S": 5.5, "C1R2_S": 5.0,
            "C2R0_N": 1.2, "C2R1_N": 1.2,
        },
    },
]

GENERATOR_SCENE_SPECS = [
    {"scene_type": "simple_local", "bucket": "simple_local", "difficulty": "easy"},
    {"scene_type": "directional_asymmetry", "bucket": "directional_asymmetry", "difficulty": "medium"},
    {"scene_type": "core_blockage", "bucket": "core_blockage", "difficulty": "medium"},
    {"scene_type": "compound_disaster", "bucket": "complex_spatiotemporal", "difficulty": "hard"},
    {"scene_type": "temporal_switch", "bucket": "complex_spatiotemporal", "difficulty": "hard"},
    {"scene_type": "propagation_range", "bucket": "complex_spatiotemporal", "difficulty": "hard"},
]


def normalize_scene_type(scene_type: str | None) -> str:
    scene = str(scene_type or "").strip().lower()
    if scene in SCENE_TYPES:
        return scene
    return "simple_local"


def scene_bucket(scene_type: str | None) -> str:
    return SCENE_BUCKETS.get(normalize_scene_type(scene_type), "simple_local")


def classify_scene_type(name: str = "", text: str = "", event_count: int | None = None) -> str:
    blob = f"{name} {text}".strip()
    baseline_global_tokens = (
        "normal_baseline",
        "baseline",
        "基线场景",
        "基线通行",
        "正常基线",
        "整体通行正常",
        "路网整体通行正常",
        "当前路网整体通行正常",
        "全网通行正常",
        "路网整体正常",
        "整体通畅",
        "整体畅通",
    )
    baseline_neutral_tokens = (
        "无明显拥堵",
        "无拥堵",
        "无明显异常",
        "无事故",
        "无封闭",
        "无管制",
        "保持基线交通状态",
    )
    if any(token in blob for token in baseline_global_tokens):
        return "normal_baseline"
    if sum(1 for token in baseline_neutral_tokens if token in blob) >= 2:
        return "normal_baseline"
    if any(token in blob for token in ("当前时间", "恢复正常", "切换", "早高峰", "晚高峰", "09:30后")):
        return "temporal_switch"
    if any(token in blob for token in ("外溢", "传播", "排队至", "波及", "回溢", "扩散")):
        return "propagation_range"
    if event_count is not None and event_count >= 3:
        return "compound_disaster"
    if any(token in blob for token in ("大规模", "复合", "灾害", "多路段", "多处", "同时", "极限压力")):
        return "compound_disaster"

    if "不对称" in blob:
        return "directional_asymmetry"
    directional_pairs = [("向东", "向西"), ("东向", "西向"), ("向北", "向南"), ("北向", "南向")]
    road_names = ("农业路", "红专路", "政七街", "黄河路", "纬五路", "经一路", "经三路", "经六路", "经八路", "花园路", "未来路")
    for road in road_names:
        local_blob = re.sub(r"[；。;\n]", " ", blob)
        for left, right in directional_pairs:
            if road in local_blob and left in local_blob and right in local_blob:
                return "directional_asymmetry"

    if any(token in blob for token in ("中央", "核心", "封锁", "全线封闭", "完全封闭", "绕行", "管制")):
        return "core_blockage"
    return "simple_local"


def get_ui_scenarios() -> dict[str, tuple[str, str, str, str, str]]:
    return {
        item["name"]: (
            item["start"],
            item["end"],
            item["constraint"],
            normalize_scene_type(item.get("scene_type")),
            resolve_scene_profile_name(
                scene_name=item.get("scene_profile"),
                scene_type=item.get("scene_type"),
            ),
        )
        for item in APP_SCENARIO_PRESETS
    }


def get_spatiotemporal_prompt_cases() -> list[dict]:
    return [dict(item) for item in SPATIOTEMPORAL_PROMPT_CASES]


def get_special_stress_cases() -> list[dict]:
    return [dict(item) for item in SPECIAL_STRESS_SCENARIOS]


def get_generator_scene_specs() -> list[dict]:
    return [dict(item) for item in GENERATOR_SCENE_SPECS]


def get_scene_library() -> list[dict]:
    out = []
    for item in APP_SCENARIO_PRESETS:
        row = dict(item)
        row["scene_profile"] = resolve_scene_profile_name(
            scene_name=item.get("scene_profile"),
            scene_type=item.get("scene_type"),
        )
        out.append(row)
    return out
