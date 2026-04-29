from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

# 统一结果根目录：项目内所有新结果均写入该根目录。
RESULTS_ROOT = "/root/autodl-tmp/results"
RUNS_SUBDIR = "runs"


@dataclass
class ResultBundle:
    run_id: str
    run_dir: str
    summary_json: str
    metrics_csv: str
    eval_report_txt: str
    config_snapshot_json: str


def now_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str, limit: int = 48) -> str:
    if value is None:
        value = "na"
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-._")
    return (s or "na")[:limit]


def create_result_bundle(
    *,
    mode: str,
    planner: str,
    model: str,
    results_root: str = RESULTS_ROOT,
    timestamp: str | None = None,
) -> ResultBundle:
    ts = timestamp or now_utc_timestamp()
    run_id = f"{ts}__{_slug(mode)}__{_slug(planner)}__{_slug(model)}"
    run_dir = os.path.join(results_root, RUNS_SUBDIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return ResultBundle(
        run_id=run_id,
        run_dir=run_dir,
        summary_json=os.path.join(run_dir, "summary.json"),
        metrics_csv=os.path.join(run_dir, "metrics.csv"),
        eval_report_txt=os.path.join(run_dir, "eval_report.txt"),
        config_snapshot_json=os.path.join(run_dir, "config_snapshot.json"),
    )


STANDARD_TIMING_KEYS = ("init_ms", "infer_ms", "smooth_ms", "total_ms", "cold_start")
PROTOCOL_SNAPSHOT_KEYS = (
    "simulation_seed",
    "warmup_seconds",
    "evaluation_start_time",
    "evaluation_end_time",
    "reroute_period",
    "scene_profile_version",
)
_INTEGER_PROTOCOL_KEYS = frozenset(PROTOCOL_SNAPSHOT_KEYS[:-1])

def normalize_timing_info(
    timing: Mapping[str, Any] | None,
    *,
    defaults: Mapping[str, Any] | None = None,
    precision: int = 2,
) -> dict[str, Any]:
    """归一化标准计时字段，统一毫秒口径。"""
    base = {
        "init_ms": 0.0,
        "infer_ms": 0.0,
        "smooth_ms": 0.0,
        "total_ms": 0.0,
        "cold_start": False,
    }
    if defaults:
        for k in base:
            if k in defaults:
                base[k] = defaults[k]

    src = dict(timing or {})
    out = {}
    for k in ("init_ms", "infer_ms", "smooth_ms", "total_ms"):
        try:
            out[k] = round(float(src.get(k, base[k])), precision)
        except Exception:
            out[k] = round(float(base[k]), precision)
    out["cold_start"] = bool(src.get("cold_start", base["cold_start"]))
    return out

def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_metrics_csv(path: str, rows: Iterable[Mapping[str, Any]], field_order: list[str] | None = None) -> list[str]:
    rows = list(rows)
    if field_order:
        fields = list(field_order)
    else:
        field_set: list[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    field_set.append(k)
        fields = field_set

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return fields


def _is_missing_protocol_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _coerce_protocol_scalar(field_name: str, value: Any) -> Any:
    if _is_missing_protocol_value(value):
        return None
    if field_name not in _INTEGER_PROTOCOL_KEYS:
        return str(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if "|" in text:
            return text
        try:
            return int(float(text))
        except Exception:
            return text
    return value


def _parse_protocol_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


def _protocol_row_id(row: Mapping[str, Any]) -> str | None:
    for key in ("scene_profile", "scenario", "scene", "case_name", "case_id", "scene_type", "subset"):
        value = row.get(key)
        if not _is_missing_protocol_value(value):
            return str(value)
    return None


def _collect_protocol_fields_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    collected: dict[str, list[tuple[str | None, Any]]] = {key: [] for key in PROTOCOL_SNAPSHOT_KEYS}
    nested_protocol_keys = ("protocol_snapshot", "sim_eval_protocol", "protocol", "scene_eval_protocol")

    for raw_row in rows:
        row = dict(raw_row)
        row_id = _protocol_row_id(row)
        payloads = [row]

        for nested_key in nested_protocol_keys:
            payload = _parse_protocol_payload(row.get(nested_key))
            if payload:
                payloads.append(payload)

        payload = _parse_protocol_payload(row.get("protocol_snapshot_json"))
        if payload:
            payloads.append(payload)

        for payload in payloads:
            for field_name in PROTOCOL_SNAPSHOT_KEYS:
                if field_name not in payload:
                    continue
                value = _coerce_protocol_scalar(field_name, payload.get(field_name))
                if _is_missing_protocol_value(value):
                    continue
                collected[field_name].append((row_id, value))

    derived: dict[str, Any] = {}
    for field_name, entries in collected.items():
        if not entries:
            continue

        unique_values: list[Any] = []
        row_map: dict[str, Any] = {}
        for row_id, value in entries:
            if all(value != existing for existing in unique_values):
                unique_values.append(value)
            if row_id is not None and row_id not in row_map:
                row_map[row_id] = value

        if len(unique_values) == 1:
            derived[field_name] = unique_values[0]
        elif row_map:
            derived[field_name] = dict(sorted(row_map.items()))
        else:
            derived[field_name] = unique_values

    return derived


def ensure_protocol_snapshot(
    config_snapshot: Mapping[str, Any],
    *,
    metrics_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    out = dict(config_snapshot)
    derived = _collect_protocol_fields_from_rows(metrics_rows or [])

    for field_name in PROTOCOL_SNAPSHOT_KEYS:
        current_value = out.get(field_name)
        if _is_missing_protocol_value(current_value):
            out[field_name] = derived.get(field_name)
        else:
            out[field_name] = current_value
        if field_name not in out:
            out[field_name] = None

    return out


def default_config_snapshot(
    *,
    timestamp: str,
    mode: str,
    seed: int,
    planner: str,
    model: str,
    scenarios: list[Any],
    fallback_policy: str,
    default_weight_policy: str,
    timing_policy: str,
    simulation_seed: Any | None = None,
    warmup_seconds: Any | None = None,
    evaluation_start_time: Any | None = None,
    evaluation_end_time: Any | None = None,
    reroute_period: Any | None = None,
    scene_profile_version: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "timestamp": timestamp,
        "mode": mode,
        "seed": seed,
        "planner": planner,
        "model": model,
        "scenario_list": scenarios,
        "fallback_policy": fallback_policy,
        "default_weight_policy": default_weight_policy,
        "timing_policy": timing_policy,
    }
    protocol_fields = {
        "simulation_seed": simulation_seed,
        "warmup_seconds": warmup_seconds,
        "evaluation_start_time": evaluation_start_time,
        "evaluation_end_time": evaluation_end_time,
        "reroute_period": reroute_period,
        "scene_profile_version": scene_profile_version,
    }
    out.update(protocol_fields)
    if extra:
        out.update(dict(extra))
    return ensure_protocol_snapshot(out)


def write_standard_artifacts(
    bundle: ResultBundle,
    *,
    summary: Mapping[str, Any],
    metrics_rows: Iterable[Mapping[str, Any]],
    report_text: str,
    config_snapshot: Mapping[str, Any],
    metrics_field_order: list[str] | None = None,
) -> None:
    metrics_rows = list(metrics_rows)
    config_snapshot = ensure_protocol_snapshot(config_snapshot, metrics_rows=metrics_rows)
    write_json(bundle.summary_json, dict(summary))
    write_metrics_csv(bundle.metrics_csv, metrics_rows, field_order=metrics_field_order)
    write_text(bundle.eval_report_txt, report_text)
    write_json(bundle.config_snapshot_json, dict(config_snapshot))
