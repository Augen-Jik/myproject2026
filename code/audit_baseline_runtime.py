from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import ConfigManager
from demo_scene_selector import select_showcase_scenarios
from roadnet_meta import live_edge_ids
from sim_scene_profiles import build_environment_weight_layers, get_scene_profile, list_scene_profiles


DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/results/runtime_audit"


def _ratio(path_edges: list[str], edge_set: set[str]) -> float:
    if not path_edges:
        return float("nan")
    return len(set(path_edges) & set(edge_set or set())) / max(len(path_edges), 1)


def _common_edge_ratio(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return float("nan")
    return len(set(left) & set(right)) / max(len(left), len(right), 1)


def _safe_scene_key(scene: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(scene or "scene")).strip("_") or "scene"


def _resolve_scene_inputs(scene: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    config = ConfigManager().config
    showcase_cases: dict[str, Any] = {}
    try:
        showcase_cases = select_showcase_scenarios(config.get("SUMO_NET_PATH"))
    except Exception as exc:
        warnings.append(f"warning: could not load showcase cases: {exc}")

    scene_key = str(scene or "").strip()
    if scene_key in showcase_cases:
        case = dict(showcase_cases[scene_key] or {})
        profile = get_scene_profile(str(case.get("scene_profile") or "normal_baseline"))
        return {
            "requested_scene": scene_key,
            "scene": scene_key,
            "scene_profile": profile,
            "start_edge": case.get("start_edge"),
            "end_edge": case.get("end_edge"),
            "baseline_path_edges": list(case.get("baseline_path") or []),
            "planned_path_edges": list(case.get("planned_path") or []),
            "scene_weight_layers": dict(case.get("scene_weight_layers") or {}),
            "source": "demo_scene_selector.showcase_case",
        }, warnings

    if scene_key in list_scene_profiles():
        profile = get_scene_profile(scene_key)
        warnings.append(
            "warning: scene is a profile name, but no OD/path result was found; "
            "path-dependent overlap metrics will be NaN."
        )
        return {
            "requested_scene": scene_key,
            "scene": scene_key,
            "scene_profile": profile,
            "start_edge": None,
            "end_edge": None,
            "baseline_path_edges": [],
            "planned_path_edges": [],
            "scene_weight_layers": {},
            "source": "scene_profile_only",
        }, warnings

    warnings.append(
        f"warning: no showcase case or scene profile named {scene_key!r} was found; "
        "writing an empty audit instead of aborting."
    )
    profile = get_scene_profile("normal_baseline")
    return {
        "requested_scene": scene_key,
        "scene": scene_key,
        "scene_profile": profile,
        "start_edge": None,
        "end_edge": None,
        "baseline_path_edges": [],
        "planned_path_edges": [],
        "scene_weight_layers": {},
        "source": "missing_scene_fallback",
    }, warnings


def build_audit_summary(
    *,
    scene: str,
    baseline_method: str,
    planned_method: str,
) -> tuple[dict[str, Any], list[str]]:
    resolved, warnings = _resolve_scene_inputs(scene)
    profile = resolved["scene_profile"]
    edge_ids = tuple(live_edge_ids())
    scene_weight_layers = dict(resolved.get("scene_weight_layers") or {})
    if not scene_weight_layers:
        enable_spillover = str(resolved.get("scene") or "").endswith("_showcase")
        scene_weight_layers = build_environment_weight_layers(edge_ids, profile, enable_spillover=enable_spillover)

    baseline_path = [str(edge_id) for edge_id in list(resolved.get("baseline_path_edges") or [])]
    planned_path = [str(edge_id) for edge_id in list(resolved.get("planned_path_edges") or [])]
    if not baseline_path:
        warnings.append("warning: baseline path result not found; baseline path metrics are NaN.")
    if not planned_path:
        warnings.append("warning: planned path result not found; planned path metrics are NaN.")

    direct_affected = (
        set(dict(profile.incident_edges or {}).keys())
        | set(tuple(profile.blocked_edges or ()))
        | set(dict(profile.lane_reduction_edges or {}).keys())
    )
    spillover_edges = set(dict(scene_weight_layers.get("spillover_weights") or {}).keys())
    all_affected = direct_affected | spillover_edges

    baseline_affected_ratio = _ratio(baseline_path, all_affected)
    planned_affected_ratio = _ratio(planned_path, all_affected)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scene": resolved.get("scene"),
        "requested_scene": resolved.get("requested_scene"),
        "scene_profile": profile.scene_name,
        "scene_type": profile.scene_type,
        "source": resolved.get("source"),
        "baseline_method": baseline_method,
        "planned_method": planned_method,
        "start_edge": resolved.get("start_edge"),
        "end_edge": resolved.get("end_edge"),
        "incident_edges": dict(profile.incident_edges or {}),
        "blocked_edges": list(profile.blocked_edges or ()),
        "lane_reduction_edges": dict(profile.lane_reduction_edges or {}),
        "spillover_edges": sorted(spillover_edges),
        "baseline_path_edges": baseline_path,
        "planned_path_edges": planned_path,
        "baseline_incident_overlap_ratio": _ratio(baseline_path, direct_affected),
        "baseline_spillover_overlap_ratio": _ratio(baseline_path, spillover_edges),
        "planned_incident_overlap_ratio": _ratio(planned_path, direct_affected),
        "planned_spillover_overlap_ratio": _ratio(planned_path, spillover_edges),
        "baseline_edge_count": len(baseline_path),
        "planned_edge_count": len(planned_path),
        "common_edge_ratio": _common_edge_ratio(baseline_path, planned_path),
        "baseline_affected_edge_overlap_ratio": baseline_affected_ratio,
        "planned_affected_edge_overlap_ratio": planned_affected_ratio,
        "affected_edge_overlap_delta": (
            baseline_affected_ratio - planned_affected_ratio
            if math.isfinite(baseline_affected_ratio) and math.isfinite(planned_affected_ratio)
            else float("nan")
        ),
        "warnings": warnings,
    }
    return summary, warnings


def save_summary(summary: dict[str, Any], output_dir: str) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit_summary.json"
    csv_path = out_dir / "audit_summary.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    scalar_row = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
        for key, value in summary.items()
    }
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_row.keys()))
        writer.writeheader()
        writer.writerow(scalar_row)
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit baseline/planned route overlap with runtime-affected SUMO edges.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--baseline-method", default="baseline")
    parser.add_argument("--planned-method", default="sparse_lora")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary, warnings = build_audit_summary(
        scene=args.scene,
        baseline_method=args.baseline_method,
        planned_method=args.planned_method,
    )
    json_path, csv_path = save_summary(summary, args.output_dir)
    for warning in warnings:
        print(warning, file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
