from __future__ import annotations

"""统一的边级指标定义与兼容读取工具。"""

from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

DEFAULT_EDGE_WEIGHT = 2.0
DEFAULT_SIGNAL_TOLERANCE = 0.05
METHOD_DEFAULT_WEIGHT: Dict[str, float] = {
    # 等权基线以 1.0 作为中性默认值，避免被误记为 100% 非默认信号。
    "Uniform-Dijkstra": 1.0,
    "Dijkstra": 1.0,
}


def resolve_default_weight(method_name: str | None = None, default_weight: float | None = None) -> float:
    """统一解析某个方法的默认/中性权重。"""
    if default_weight is not None:
        return float(default_weight)
    if method_name and method_name in METHOD_DEFAULT_WEIGHT:
        return float(METHOD_DEFAULT_WEIGHT[method_name])
    return DEFAULT_EDGE_WEIGHT


def _edge_list(all_edges: Iterable[str]) -> list[str]:
    return list(all_edges)


def _explicit_edge_ids(
    explicit_weights: Mapping[str, float] | None = None,
    explicit_edge_ids: Iterable[str] | None = None,
    all_edges: Iterable[str] | None = None,
) -> list[str]:
    valid = set(all_edges) if all_edges is not None else None
    seen: list[str] = []
    for source in (explicit_weights.keys() if explicit_weights else [], explicit_edge_ids or []):
        for eid in source:
            if valid is not None and eid not in valid:
                continue
            if eid not in seen:
                seen.append(eid)
    return seen


def count_parsed_edges(
    all_edges: Iterable[str],
    explicit_weights: Mapping[str, float] | None = None,
    explicit_edge_ids: Iterable[str] | None = None,
) -> int:
    """统计真正由模型/规则显式给出的边数，不把默认补全计入其中。"""
    return len(_explicit_edge_ids(explicit_weights, explicit_edge_ids, all_edges))


def count_signal_edges(
    final_weights: Mapping[str, float],
    all_edges: Iterable[str],
    method_name: str | None = None,
    default_weight: float | None = None,
    tolerance: float = DEFAULT_SIGNAL_TOLERANCE,
) -> int:
    """统计最终权重中偏离默认权重的边数。"""
    default_w = resolve_default_weight(method_name, default_weight)
    signal_edges = 0
    for eid in _edge_list(all_edges):
        if abs(float(final_weights.get(eid, default_w)) - default_w) > tolerance:
            signal_edges += 1
    return signal_edges


def count_default_edges(
    final_weights: Mapping[str, float],
    all_edges: Iterable[str],
    method_name: str | None = None,
    default_weight: float | None = None,
    tolerance: float = DEFAULT_SIGNAL_TOLERANCE,
) -> int:
    """统计最终权重保持默认值的边数。"""
    total_edges = len(_edge_list(all_edges))
    return total_edges - count_signal_edges(final_weights, all_edges, method_name, default_weight, tolerance)


def compute_edge_metrics(
    final_weights: Mapping[str, float],
    all_edges: Iterable[str],
    explicit_weights: Mapping[str, float] | None = None,
    explicit_edge_ids: Iterable[str] | None = None,
    method_name: str | None = None,
    default_weight: float | None = None,
    tolerance: float = DEFAULT_SIGNAL_TOLERANCE,
) -> dict:
    """
    统一输出五个核心指标：
    - parsed_edges / parsed_edge_ratio
    - signal_edges / signal_ratio
    - default_edges
    """
    edge_ids = _edge_list(all_edges)
    total_edges = len(edge_ids)
    parsed_edges = count_parsed_edges(edge_ids, explicit_weights, explicit_edge_ids)
    signal_edges = count_signal_edges(final_weights, edge_ids, method_name, default_weight, tolerance)
    default_edges = total_edges - signal_edges
    parsed_edge_ratio = round(parsed_edges / total_edges * 100, 1) if total_edges else 0.0
    signal_ratio = round(signal_edges / total_edges * 100, 1) if total_edges else 0.0
    return {
        "total_edges": total_edges,
        "parsed_edges": int(parsed_edges),
        "parsed_edge_ratio": parsed_edge_ratio,
        "signal_edges": int(signal_edges),
        "signal_ratio": signal_ratio,
        "default_edges": int(default_edges),
    }


def add_legacy_metric_aliases(metrics: Mapping[str, float], *, include_parse_rate: bool = False, include_coverage: bool = False, include_parsed_ratio: bool = False) -> dict:
    """
    为旧版结果文件提供只读兼容字段。

    deprecated:
      - parse_rate   -> parsed_edge_ratio
      - coverage     -> signal_ratio
      - parsed_ratio -> signal_ratio
    """
    out = dict(metrics)
    if include_parse_rate:
        out["parse_rate"] = out.get("parsed_edge_ratio", 0.0)
    if include_coverage:
        out["coverage"] = out.get("signal_ratio", 0.0)
    if include_parsed_ratio:
        out["parsed_ratio"] = out.get("signal_ratio", 0.0)
    return out


def normalize_edge_metric_record(
    record: Mapping[str, float],
    *,
    total_edges: int | None = None,
    method_name: str | None = None,
    default_weight: float | None = None,
) -> dict:
    """
    读取新旧结果文件时统一成标准字段。

    优先级：
      1. 新字段 parsed_edge_ratio / signal_ratio / signal_edges / default_edges
      2. 兼容旧字段 parse_rate / coverage / parsed_ratio
      3. 用 total_edges 反推缺失计数
    """
    total = int(record.get("total_edges", total_edges or 0) or 0)

    parsed_edges = record.get("parsed_edges")
    parsed_edge_ratio = record.get("parsed_edge_ratio")
    signal_edges = record.get("signal_edges")
    signal_ratio = record.get("signal_ratio")
    default_edges = record.get("default_edges")

    if parsed_edge_ratio is None:
        parsed_edge_ratio = record.get("parse_rate")
    if signal_ratio is None:
        # deprecated: `coverage` / `parsed_ratio` 曾混用为“非默认占比”。
        signal_ratio = record.get("coverage")
    if signal_ratio is None:
        signal_ratio = record.get("parsed_ratio")

    if parsed_edges is None and parsed_edge_ratio is not None and total:
        parsed_edges = round(float(parsed_edge_ratio) / 100.0 * total)
    if signal_edges is None and signal_ratio is not None and total:
        signal_edges = round(float(signal_ratio) / 100.0 * total)
    if default_edges is None and signal_edges is not None and total:
        default_edges = total - int(signal_edges)

    parsed_edges = int(parsed_edges or 0)
    signal_edges = int(signal_edges or 0)
    default_edges = int(default_edges if default_edges is not None else max(total - signal_edges, 0))

    if parsed_edge_ratio is None:
        parsed_edge_ratio = round(parsed_edges / total * 100, 1) if total else 0.0
    if signal_ratio is None:
        signal_ratio = round(signal_edges / total * 100, 1) if total else 0.0

    return {
        "total_edges": total,
        "parsed_edges": parsed_edges,
        "parsed_edge_ratio": float(parsed_edge_ratio or 0.0),
        "signal_edges": signal_edges,
        "signal_ratio": float(signal_ratio or 0.0),
        "default_edges": default_edges,
        "default_weight": resolve_default_weight(method_name, default_weight),
    }


def _run_sanity_checks() -> None:
    edges = ["E1", "E2", "E3", "E4"]

    case1 = compute_edge_metrics(
        final_weights={eid: 2.0 for eid in edges},
        explicit_weights={},
        all_edges=edges,
    )
    assert case1["signal_ratio"] == 0.0 and case1["parsed_edge_ratio"] == 0.0

    case2 = compute_edge_metrics(
        final_weights={"E1": 7.5, "E2": 2.0, "E3": 2.0, "E4": 2.0},
        explicit_weights={"E1": 7.5, "E2": 2.0},
        all_edges=edges,
    )
    assert case2["parsed_edges"] == 2 and case2["signal_edges"] == 1 and case2["default_edges"] == 3

    case3 = compute_edge_metrics(
        final_weights={"E1": 7.5, "E2": 2.0, "E3": 2.0, "E4": 2.0},
        explicit_weights={"E1": 7.5},
        all_edges=edges,
    )
    assert case3["parsed_edges"] == 1 and case3["parsed_edge_ratio"] == 25.0
    assert case3["signal_edges"] == 1 and case3["signal_ratio"] == 25.0


if __name__ == "__main__":
    _run_sanity_checks()
    print("eval_metrics sanity checks passed")
