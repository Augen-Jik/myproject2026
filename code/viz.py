import io
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from enum import Enum
from functools import lru_cache
from math import cos, radians
from pathlib import Path

import folium
import networkx as nx
import pandas as pd
import streamlit as st
import sumolib
from branca.element import Element

from display_geometry import (
    build_real_road_corridors as display_build_real_road_corridors,
    logical_path_to_real_display_path,
    map_logical_grid_to_corridors,
)
from roadnet_meta import COL_NAMES, ROW_NAMES, load_roadnet_meta

# SUMO 路网缓存，避免重复读 net.xml。
_net_cache = {}

MAP_STYLE_LABELS = {
    "realistic_overlay": "Realistic Block Overlay Map",
    "clean_schematic": "Clean Schematic Map",
}

BASEMAP_LABELS = {
    "labeled_light": "Labeled Light Basemap",
    "nolabel_light": "No-Label Light Basemap",
    "minimal": "Minimal Basemap",
}

CONGESTION_STYLE = {
    "clear": {"label": "畅通", "color": "#22C55E", "width": 6, "opacity": 0.82},
    "light": {"label": "轻度拥堵", "color": "#FACC15", "width": 7, "opacity": 0.86},
    "medium": {"label": "中度拥堵", "color": "#F97316", "width": 8, "opacity": 0.9},
    "severe": {"label": "严重拥堵", "color": "#DC2626", "width": 9, "opacity": 0.94},
    "closed": {"label": "封闭", "color": "#7F1D1D", "width": 10, "opacity": 0.96},
}

ROAD_TYPE_STYLE = {
    "A": {"label": "主干道", "color": "#52606D", "weight": 4.5, "opacity": 0.78},
    "S": {"label": "次干道", "color": "#7F8EA3", "weight": 3.5, "opacity": 0.66},
    "M": {"label": "支路", "color": "#B7C4D1", "weight": 2.5, "opacity": 0.58},
}

ROAD_LABEL_STYLE = {
    "A": {"font_size": 14, "color": "#0F172A", "bg": "rgba(255,255,255,0.78)"},
    "S": {"font_size": 12, "color": "#334155", "bg": "rgba(255,255,255,0.72)"},
    "M": {"font_size": 11, "color": "#475569", "bg": "rgba(255,255,255,0.68)"},
}


def _geo_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Approximate lat/lon distance in meters."""
    lat1, lon1 = a
    lat2, lon2 = b
    x = (lon2 - lon1) * 111320.0 * cos(radians((lat1 + lat2) / 2.0))
    y = (lat2 - lat1) * 110540.0
    return (x * x + y * y) ** 0.5


def _polyline_length(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    return sum(_geo_distance_m(a, b) for a, b in zip(coords, coords[1:]))


def _mean_latlon(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(lat for lat, _ in points) / len(points),
        sum(lon for _, lon in points) / len(points),
    )


def _edge_bounds(edge_coords: dict[str, list[tuple[float, float]]]) -> tuple[float, float, float, float] | None:
    xs, ys = [], []
    for coords in edge_coords.values():
        for x, y in coords:
            xs.append(float(x))
            ys.append(float(y))
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _geo_bounds_from_lines(
    lines: list[list[tuple[float, float]]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    pts = [pt for line in lines for pt in line]
    if not pts:
        return None
    min_lat = min(lat for lat, _ in pts)
    min_lon = min(lon for _, lon in pts)
    max_lat = max(lat for lat, _ in pts)
    max_lon = max(lon for _, lon in pts)
    return (min_lat, min_lon), (max_lat, max_lon)


def _pad_geo_bounds(
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
    *,
    lat_ratio: float = 0.12,
    lon_ratio: float = 0.12,
    min_pad: float = 0.00035,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not bounds:
        return None
    (min_lat, min_lon), (max_lat, max_lon) = bounds
    lat_pad = max((max_lat - min_lat) * lat_ratio, min_pad)
    lon_pad = max((max_lon - min_lon) * lon_ratio, min_pad)
    return (
        (min_lat - lat_pad, min_lon - lon_pad),
        (max_lat + lat_pad, max_lon + lon_pad),
    )


def _warp_projected_polyline(
    coords: list[tuple[float, float]],
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    min_x, min_y, max_x, max_y = source_bounds
    min_lat, min_lon, max_lat, max_lon = target_bounds
    dx = max(max_x - min_x, 1e-9)
    dy = max(max_y - min_y, 1e-9)
    mapped = []
    for x, y in coords:
        lon = min_lon + ((float(x) - min_x) / dx) * (max_lon - min_lon)
        lat = min_lat + ((float(y) - min_y) / dy) * (max_lat - min_lat)
        mapped.append((lat, lon))
    return mapped


def _normalize_map_style(map_style: str | None) -> str:
    token = str(map_style or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {
        "realistic",
        "realistic_light",
        "light_map",
        "realistic_overlay",
        "realistic_overlay_map",
    }:
        return "realistic_overlay"
    if token in {"clean_grid", "clean_schematic", "clean_schematic_map"}:
        return "clean_schematic"
    return "realistic_overlay"


def _normalize_basemap_style(basemap_style: str | None) -> str:
    token = str(basemap_style or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {
        "labeled",
        "labeled_light",
        "labeled_light_basemap",
        "label_light",
        "light_with_labels",
    }:
        return "labeled_light"
    if token in {
        "nolabel",
        "no_label",
        "no_label_light",
        "nolabel_light",
        "no_label_light_basemap",
        "light_nolabels",
    }:
        return "nolabel_light"
    if token in {"minimal", "minimal_basemap", "none"}:
        return "minimal"
    return "nolabel_light"


def _classify_congestion(weight: float | None) -> str:
    value = float(weight) if weight is not None else 2.0
    if value >= 9.0:
        return "closed"
    if value >= 7.0:
        return "severe"
    if value >= 4.5:
        return "medium"
    if value >= 2.5:
        return "light"
    return "clear"


def _congestion_style(weight: float | None) -> dict:
    return CONGESTION_STYLE[_classify_congestion(weight)]


def _route_time_metrics(
    baseline_time_s: float | None,
    planned_time_s: float | None,
) -> dict[str, float | None]:
    if baseline_time_s is None or planned_time_s is None:
        return {
            "baseline_time_s": baseline_time_s,
            "planned_time_s": planned_time_s,
            "delta_time_s": None,
            "delta_ratio_pct": None,
            "is_positive_gain": None,
        }
    delta = float(baseline_time_s) - float(planned_time_s)
    neutral_threshold = max(3.0, abs(float(baseline_time_s)) * 0.003)
    if abs(delta) <= neutral_threshold:
        trend = "equal"
        is_positive = False
    elif delta > 0.0:
        trend = "better"
        is_positive = True
    else:
        trend = "worse"
        is_positive = False
    ratio = (abs(delta) / float(baseline_time_s) * 100.0) if float(baseline_time_s) > 1e-9 else 0.0
    return {
        "baseline_time_s": float(baseline_time_s),
        "planned_time_s": float(planned_time_s),
        "delta_signed_s": float(delta),
        "delta_time_s": abs(delta),
        "delta_ratio_pct": ratio,
        "is_positive_gain": is_positive,
        "trend": trend,
    }


def _resolve_display_bounds(bundle) -> tuple[float, float, float, float]:
    bounds = bundle.get("reference_bounds")
    if bounds is None and bundle.get("fit_bounds"):
        (min_lat, min_lon), (max_lat, max_lon) = bundle["fit_bounds"]
        bounds = (min_lat, min_lon, max_lat, max_lon)
    if bounds is None:
        bounds = (34.7697100, 113.6489300, 34.7945600, 113.6969900)
    min_lat, min_lon, max_lat, max_lon = bounds
    lat_pad = max((max_lat - min_lat) * 0.08, 1e-4)
    lon_pad = max((max_lon - min_lon) * 0.08, 1e-4)
    return (
        min_lat + lat_pad,
        min_lon + lon_pad,
        max_lat - lat_pad,
        max_lon - lon_pad,
    )


def _fit_bounds_from_box(bounds: tuple[float, float, float, float]):
    min_lat, min_lon, max_lat, max_lon = bounds
    return (min_lat, min_lon), (max_lat, max_lon)


def _offset_latlon_line(
    coords: list[tuple[float, float]],
    *,
    axis: str,
    lat_step: float,
    lon_step: float,
    factor: float,
) -> list[tuple[float, float]]:
    if not coords:
        return []
    lat_shift = lat_step * factor if axis == "H" else 0.0
    lon_shift = lon_step * factor if axis == "V" else 0.0
    return [(lat + lat_shift, lon + lon_shift) for lat, lon in coords]


def _canonicalize_axis_polyline(
    coords: list[tuple[float, float]],
    axis: str,
) -> list[tuple[float, float]]:
    if not coords:
        return []
    line = list(coords)
    if axis == "H":
        return line if line[0][1] <= line[-1][1] else list(reversed(line))
    return line if line[0][0] >= line[-1][0] else list(reversed(line))


def _blend_points(
    a: tuple[float, float],
    b: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    return (
        float(a[0]) * (1.0 - alpha) + float(b[0]) * alpha,
        float(a[1]) * (1.0 - alpha) + float(b[1]) * alpha,
    )


def _polyline_sample_at(
    coords: list[tuple[float, float]],
    t: float,
) -> tuple[float, float]:
    if not coords:
        return (0.0, 0.0)
    if len(coords) == 1:
        return coords[0]
    t = min(max(float(t), 0.0), 1.0)
    total = _polyline_length(coords)
    if total <= 1e-9:
        return coords[0]
    target = total * t
    acc = 0.0
    for src, dst in zip(coords, coords[1:]):
        seg = _geo_distance_m(src, dst)
        if acc + seg >= target:
            ratio = 0.0 if seg <= 1e-9 else (target - acc) / seg
            return _blend_points(src, dst, ratio)
        acc += seg
    return coords[-1]


def _resample_polyline(
    coords: list[tuple[float, float]],
    num_points: int = 5,
) -> list[tuple[float, float]]:
    if len(coords) <= 2:
        return list(coords)
    count = max(int(num_points), 2)
    return [_polyline_sample_at(coords, idx / (count - 1)) for idx in range(count)]


def _offset_polyline_perpendicular(
    coords: list[tuple[float, float]],
    offset_m: float,
) -> list[tuple[float, float]]:
    if len(coords) < 2 or abs(offset_m) < 1e-6:
        return list(coords)
    start = coords[0]
    end = coords[-1]
    mean_lat = (start[0] + end[0]) / 2.0
    dx = (end[1] - start[1]) * 111320.0 * cos(radians(mean_lat))
    dy = (end[0] - start[0]) * 110540.0
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    nx = -dy / length
    ny = dx / length
    dlon = (offset_m * nx) / max(111320.0 * cos(radians(mean_lat)), 1e-6)
    dlat = (offset_m * ny) / 110540.0
    return [(lat + dlat, lon + dlon) for lat, lon in coords]


def _smooth_display_path(
    coords: list[tuple[float, float]],
    strength: float = 0.2,
) -> list[tuple[float, float]]:
    if len(coords) < 3:
        return list(coords)
    strength = min(max(float(strength), 0.0), 0.45)
    smoothed = [coords[0]]
    for idx in range(1, len(coords) - 1):
        prev_pt = coords[idx - 1]
        cur_pt = coords[idx]
        next_pt = coords[idx + 1]
        avg_pt = (
            (prev_pt[0] + cur_pt[0] + next_pt[0]) / 3.0,
            (prev_pt[1] + cur_pt[1] + next_pt[1]) / 3.0,
        )
        smoothed.append(_blend_points(cur_pt, avg_pt, strength))
    smoothed.append(coords[-1])
    return smoothed


def _merge_path_polylines(
    edge_ids: list[str] | tuple[str, ...],
    edge_geo_coords: dict[str, list[tuple[float, float]]],
    *,
    offset_m: float = 0.0,
    smooth_strength: float = 0.0,
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for edge_id in list(edge_ids or []):
        coords = list(edge_geo_coords.get(edge_id) or [])
        if not coords:
            continue
        if abs(offset_m) > 1e-6:
            coords = _offset_polyline_perpendicular(coords, offset_m)
        if merged and _geo_distance_m(merged[-1], coords[0]) < 0.8:
            merged.extend(coords[1:])
        else:
            merged.extend(coords)
    if smooth_strength > 0.0:
        merged = _smooth_display_path(merged, strength=smooth_strength)
    return merged


def load_net_cached(net_path: str):
    """读取 SUMO 路网，并缓存路段坐标。"""
    global _net_cache

    if net_path in _net_cache:
        return _net_cache[net_path]

    net = sumolib.net.readNet(net_path)
    edge_coords = {}
    for edge in net.getEdges():
        shape = edge.getShape()
        if shape:
            edge_coords[edge.getID()] = [(float(x), float(y)) for x, y in shape]

    result = (net, edge_coords)
    _net_cache[net_path] = result
    return result


def safe_table_markdown(df: pd.DataFrame) -> str:
    """把 DataFrame 转成 markdown 表格，顺手处理表格里的特殊字符。"""
    try:
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == "object":
                df_copy[col] = df_copy[col].apply(
                    lambda x: str(x).replace("|", "\\|").replace("\n", " ") if pd.notna(x) else x
                )

        buffer = io.StringIO()
        df_copy.to_markdown(buffer, tablefmt="github", index=False)
        markdown_str = buffer.getvalue()
        return markdown_str.replace("||", "| |")
    except Exception as e:
        return f"`Error generating table: {str(e)}`"


def make_json_safe(obj):
    """把组件参数整理成 JSON 能直接序列化的值。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if callable(obj):
        return f"<callable:{getattr(obj, '__name__', obj.__class__.__name__)}>"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        safe_dict = {}
        for key, value in obj.items():
            safe_value = make_json_safe(value)
            if isinstance(safe_value, str) and safe_value.startswith("<callable:"):
                continue
            safe_dict[str(key)] = safe_value
        return safe_dict
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(item) for item in obj]
    if hasattr(obj, "tolist"):
        try:
            return make_json_safe(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return make_json_safe(obj.item())
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return make_json_safe(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def sanitize_component_args(obj):
    """保留这个名字，现有组件调用还在用。"""
    return make_json_safe(obj)


def render_folium_html(map_obj) -> str:
    """把 Folium 地图渲染成 Streamlit 可嵌入的 HTML。"""
    return map_obj.get_root().render()


def clear_viz_cache():
    """清掉可视化相关缓存。"""
    global _net_cache
    _net_cache.clear()
    load_roadnet_visual_meta.cache_clear()
    load_osm_reference.cache_clear()
    load_geo_edge_bundle.cache_clear()
    load_grid_geo_bundle.cache_clear()
    build_display_anchor_grid.cache_clear()
    build_real_map_anchor_corridors.cache_clear()
    build_block_mapping.cache_clear()
    remap_grid_to_realistic_overlay.cache_clear()
    load_display_geo_bundle.cache_clear()
    display_build_real_road_corridors.cache_clear()
    map_logical_grid_to_corridors.cache_clear()


@lru_cache(maxsize=2)
def load_osm_reference(osm_path: str):
    """
    Parse local OSM XML once and keep only the named roads used by the SUMO grid.
    """
    if not osm_path or not os.path.exists(osm_path):
        return {
            "osm_path": osm_path,
            "bounds": None,
            "node_latlon": {},
            "road_way_lines": {},
            "road_nodes": {},
            "road_points": {},
            "road_graphs": {},
        }

    root = ET.parse(osm_path).getroot()
    node_latlon = {}
    for node in root.findall("node"):
        try:
            node_latlon[node.attrib["id"]] = (
                float(node.attrib["lat"]),
                float(node.attrib["lon"]),
            )
        except Exception:
            continue

    bounds = None
    bounds_tag = root.find("bounds")
    if bounds_tag is not None:
        try:
            bounds = (
                float(bounds_tag.attrib["minlat"]),
                float(bounds_tag.attrib["minlon"]),
                float(bounds_tag.attrib["maxlat"]),
                float(bounds_tag.attrib["maxlon"]),
            )
        except Exception:
            bounds = None

    target_roads = set(ROW_NAMES + COL_NAMES)
    road_way_lines = defaultdict(list)
    road_nodes = defaultdict(set)
    road_points = defaultdict(list)
    road_graphs = defaultdict(nx.Graph)

    for way in root.findall("way"):
        tags = {tag.attrib.get("k"): tag.attrib.get("v") for tag in way.findall("tag")}
        road_name = tags.get("name")
        if road_name not in target_roads or not tags.get("highway"):
            continue

        refs = [nd.attrib.get("ref") for nd in way.findall("nd")]
        refs = [ref for ref in refs if ref in node_latlon]
        if len(refs) < 2:
            continue

        coords = [node_latlon[ref] for ref in refs]
        road_way_lines[road_name].append(coords)
        road_nodes[road_name].update(refs)
        road_points[road_name].extend(coords)

        graph = road_graphs[road_name]
        for ref in refs:
            graph.add_node(ref)
        for src, dst in zip(refs, refs[1:]):
            graph.add_edge(src, dst, weight=_geo_distance_m(node_latlon[src], node_latlon[dst]))

    return {
        "osm_path": osm_path,
        "bounds": bounds,
        "node_latlon": node_latlon,
        "road_way_lines": {name: tuple(lines) for name, lines in road_way_lines.items()},
        "road_nodes": {name: frozenset(nodes) for name, nodes in road_nodes.items()},
        "road_points": {name: tuple(points) for name, points in road_points.items()},
        "road_graphs": dict(road_graphs),
    }


def _cross_candidates(reference, road_name: str, cross_name: str, limit: int = 6):
    road_nodes = reference["road_nodes"].get(road_name, frozenset())
    cross_nodes = reference["road_nodes"].get(cross_name, frozenset())
    shared = sorted(road_nodes & cross_nodes)
    if shared:
        return shared[:limit], "shared"

    graph = reference["road_graphs"].get(road_name)
    cross_points = list(reference["road_points"].get(cross_name, ()))
    if graph is None or graph.number_of_nodes() == 0 or not cross_points:
        return [], "missing"

    target = _mean_latlon(cross_points)
    if target is None:
        return [], "missing"

    node_latlon = reference["node_latlon"]
    candidates = sorted(
        graph.nodes,
        key=lambda nid: _geo_distance_m(node_latlon[nid], target),
    )
    return list(candidates[:limit]), "nearest"


def _extract_osm_segment(reference, road_name: str, start_cross: str, end_cross: str):
    graph = reference["road_graphs"].get(road_name)
    if graph is None or graph.number_of_nodes() == 0:
        return None, None

    start_nodes, start_mode = _cross_candidates(reference, road_name, start_cross)
    end_nodes, end_mode = _cross_candidates(reference, road_name, end_cross)
    if not start_nodes or not end_nodes:
        return None, None

    node_latlon = reference["node_latlon"]
    best_path = None
    best_cost = float("inf")
    for src in start_nodes:
        for dst in end_nodes:
            try:
                path_nodes = nx.shortest_path(graph, src, dst, weight="weight")
                cost = nx.path_weight(graph, path_nodes, weight="weight")
            except Exception:
                continue
            if cost < best_cost:
                best_path = path_nodes
                best_cost = cost

    if not best_path:
        return None, None

    coords = [node_latlon[nid] for nid in best_path if nid in node_latlon]
    if len(coords) == 1:
        coords = [coords[0], coords[0]]
    if len(coords) < 2:
        return None, None

    source = "osm_exact" if start_mode == "shared" and end_mode == "shared" else "osm_nearest"
    return coords, source


def _net_has_geo_projection(net, edge_coords: dict[str, list[tuple[float, float]]]) -> bool:
    for coords in edge_coords.values():
        if not coords:
            continue
        x, y = coords[0]
        try:
            net.convertXY2LonLat(x, y)
            return True
        except Exception:
            return False
    return False


def _convert_net_to_geo(net, edge_coords: dict[str, list[tuple[float, float]]]):
    geo_coords = {}
    for edge_id, coords in edge_coords.items():
        geo_line = []
        for x, y in coords:
            lon, lat = net.convertXY2LonLat(x, y)
            geo_line.append((float(lat), float(lon)))
        geo_coords[edge_id] = geo_line
    return geo_coords


@lru_cache(maxsize=4)
def load_geo_edge_bundle(net_path: str, osm_path: str = ""):
    """
    Return geospatial polylines for every SUMO edge.

    Priority:
    1. Direct SUMO geo projection.
    2. Real road geometry from local map.osm matched by road name/intersections.
    3. Linear fallback that warps synthetic XY coordinates into OSM bounds.
    """
    net, edge_coords = load_net_cached(net_path)
    road_meta = load_roadnet_meta(net_path)
    reference = load_osm_reference(osm_path)
    projected_bounds = _edge_bounds(edge_coords)

    if _net_has_geo_projection(net, edge_coords):
        geo_edge_coords = _convert_net_to_geo(net, edge_coords)
        source_map = {edge_id: "sumo_geo" for edge_id in geo_edge_coords}
        geo_lines = [coords for coords in geo_edge_coords.values() if len(coords) >= 2]
        bounds_pair = _geo_bounds_from_lines(geo_lines)
        center = (
            ((bounds_pair[0][0] + bounds_pair[1][0]) / 2.0, (bounds_pair[0][1] + bounds_pair[1][1]) / 2.0)
            if bounds_pair
            else (34.78, 113.67)
        )
        return {
            "mode": "sumo_geo",
            "edge_geo_coords": geo_edge_coords,
            "edge_source": source_map,
            "source_counts": dict(Counter(source_map.values())),
            "road_way_lines": reference.get("road_way_lines", {}),
            "center": center,
            "fit_bounds": bounds_pair,
            "reference_bounds": reference.get("bounds"),
        }

    geo_edge_coords = {}
    edge_source = {}
    target_bounds = reference.get("bounds")
    if target_bounds is None:
        target_bounds = (34.7697100, 113.6489300, 34.7945600, 113.6969900)

    for edge_id, meta in road_meta.edge_meta.items():
        coords = None
        source = None
        segment = meta.get("segment", "")
        if reference["road_graphs"] and "至" in segment:
            start_cross, end_cross = [part.strip() for part in segment.split("至", 1)]
            coords, source = _extract_osm_segment(reference, meta["road"], start_cross, end_cross)
            if coords and meta.get("dir_code") in {"W", "S"}:
                coords = list(reversed(coords))

        if not coords:
            projected = edge_coords.get(edge_id) or road_meta.edge_shapes.get(edge_id)
            if projected and projected_bounds:
                coords = _warp_projected_polyline(projected, projected_bounds, target_bounds)
                source = "synthetic_warp"

        if coords:
            geo_edge_coords[edge_id] = coords
            edge_source[edge_id] = source or "synthetic_warp"

    lines = [coords for coords in geo_edge_coords.values() if len(coords) >= 2]
    fit_bounds = _geo_bounds_from_lines(lines)
    if reference.get("bounds"):
        ref_min_lat, ref_min_lon, ref_max_lat, ref_max_lon = reference["bounds"]
        center = ((ref_min_lat + ref_max_lat) / 2.0, (ref_min_lon + ref_max_lon) / 2.0)
    elif fit_bounds:
        center = (
            (fit_bounds[0][0] + fit_bounds[1][0]) / 2.0,
            (fit_bounds[0][1] + fit_bounds[1][1]) / 2.0,
        )
    else:
        center = (34.78, 113.67)

    source_counts = Counter(edge_source.values())
    mode = "osm_reference" if source_counts.get("osm_exact", 0) or source_counts.get("osm_nearest", 0) else "synthetic_warp"
    return {
        "mode": mode,
        "edge_geo_coords": geo_edge_coords,
        "edge_source": edge_source,
        "source_counts": dict(source_counts),
        "road_way_lines": reference.get("road_way_lines", {}),
        "center": center,
        "fit_bounds": fit_bounds,
        "reference_bounds": reference.get("bounds"),
    }


@lru_cache(maxsize=4)
def load_grid_geo_bundle(net_path: str, osm_path: str = ""):
    """Build a paper-friendly schematic grid aligned to the road meta topology."""
    geo_bundle = load_geo_edge_bundle(net_path, osm_path)
    road_meta = load_roadnet_meta(net_path)
    min_lat, min_lon, max_lat, max_lon = _resolve_display_bounds(geo_bundle)
    row_count = len(ROW_NAMES)
    col_count = len(COL_NAMES)
    lat_values = [
        max_lat - idx * ((max_lat - min_lat) / max(row_count - 1, 1))
        for idx in range(row_count)
    ]
    lon_values = [
        min_lon + idx * ((max_lon - min_lon) / max(col_count - 1, 1))
        for idx in range(col_count)
    ]
    node_geo = {
        (row_idx, col_idx): (lat_values[row_idx], lon_values[col_idx])
        for row_idx in range(row_count)
        for col_idx in range(col_count)
    }

    edge_geo_coords = {}
    road_midpoints = defaultdict(list)
    for edge_id, meta in road_meta.edge_meta.items():
        axis = meta.get("axis")
        row_idx = int(meta.get("row", 0))
        col_idx = int(meta.get("col", 0))
        if axis == "H":
            west = node_geo[(row_idx, col_idx)]
            east = node_geo[(row_idx, col_idx + 1)]
            coords = [west, east] if meta.get("dir_code") == "E" else [east, west]
        else:
            north = node_geo[(row_idx, col_idx)]
            south = node_geo[(row_idx + 1, col_idx)]
            coords = [north, south] if meta.get("dir_code") == "S" else [south, north]
        edge_geo_coords[edge_id] = coords
        road_midpoints[meta.get("road", edge_id)].append(_mean_latlon(coords))

    block_polygons = []
    for row_idx in range(row_count - 1):
        for col_idx in range(col_count - 1):
            north_west = node_geo[(row_idx, col_idx)]
            north_east = node_geo[(row_idx, col_idx + 1)]
            south_west = node_geo[(row_idx + 1, col_idx)]
            south_east = node_geo[(row_idx + 1, col_idx + 1)]
            lat_gap = abs(north_west[0] - south_west[0])
            lon_gap = abs(north_east[1] - north_west[1])
            lat_margin = lat_gap * 0.22
            lon_margin = lon_gap * 0.22
            block_polygons.append(
                [
                    (north_west[0] - lat_margin, north_west[1] + lon_margin),
                    (north_east[0] - lat_margin, north_east[1] - lon_margin),
                    (south_east[0] + lat_margin, south_east[1] - lon_margin),
                    (south_west[0] + lat_margin, south_west[1] + lon_margin),
                ]
            )

    road_midpoint_map = {}
    for road_name, pts in road_midpoints.items():
        mean_pt = _mean_latlon([pt for pt in pts if pt is not None])
        if mean_pt is not None:
            road_midpoint_map[road_name] = mean_pt

    return {
        "center": (
            (min_lat + max_lat) / 2.0,
            (min_lon + max_lon) / 2.0,
        ),
        "fit_bounds": _fit_bounds_from_box((min_lat, min_lon, max_lat, max_lon)),
        "edge_geo_coords": edge_geo_coords,
        "node_geo": node_geo,
        "road_midpoints": road_midpoint_map,
        "block_polygons": block_polygons,
        "lat_step": abs(lat_values[0] - lat_values[1]) if len(lat_values) > 1 else 0.001,
        "lon_step": abs(lon_values[1] - lon_values[0]) if len(lon_values) > 1 else 0.001,
        "reference_bundle": geo_bundle,
    }


def _estimate_node_grid_from_geo(
    road_meta,
    geo_edge_coords: dict[str, list[tuple[float, float]]],
) -> dict[tuple[int, int], tuple[float, float]]:
    node_candidates: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for edge_id, meta in road_meta.edge_meta.items():
        coords = list(geo_edge_coords.get(edge_id) or [])
        if len(coords) < 2:
            continue
        axis = meta.get("axis")
        canonical = _canonicalize_axis_polyline(coords, axis)
        row_idx = int(meta.get("row", 0))
        col_idx = int(meta.get("col", 0))
        if axis == "H":
            node_candidates[(row_idx, col_idx)].append(canonical[0])
            node_candidates[(row_idx, col_idx + 1)].append(canonical[-1])
        else:
            node_candidates[(row_idx, col_idx)].append(canonical[0])
            node_candidates[(row_idx + 1, col_idx)].append(canonical[-1])

    row_count = len(ROW_NAMES)
    col_count = len(COL_NAMES)
    row_lat = {}
    col_lon = {}
    for row_idx in range(row_count):
        pts = [pt for (r, _), items in node_candidates.items() if r == row_idx for pt in items]
        if pts:
            row_lat[row_idx] = sum(pt[0] for pt in pts) / len(pts)
    for col_idx in range(col_count):
        pts = [pt for (_, c), items in node_candidates.items() if c == col_idx for pt in items]
        if pts:
            col_lon[col_idx] = sum(pt[1] for pt in pts) / len(pts)

    actual_nodes = {}
    for row_idx in range(row_count):
        for col_idx in range(col_count):
            candidates = node_candidates.get((row_idx, col_idx), [])
            if candidates:
                actual = _mean_latlon(candidates)
            else:
                actual = (
                    row_lat.get(row_idx, sum(row_lat.values()) / max(len(row_lat), 1)),
                    col_lon.get(col_idx, sum(col_lon.values()) / max(len(col_lon), 1)),
                )
            base = (row_lat.get(row_idx, actual[0]), col_lon.get(col_idx, actual[1]))
            actual_nodes[(row_idx, col_idx)] = (
                base[0] * 0.72 + actual[0] * 0.28,
                base[1] * 0.72 + actual[1] * 0.28,
            )
    return actual_nodes


@lru_cache(maxsize=4)
def build_display_anchor_grid(net_path: str, osm_path: str = ""):
    """Estimate non-uniform display anchors from real-map edge corridors."""
    geo_bundle = load_geo_edge_bundle(net_path, osm_path)
    road_meta = load_roadnet_meta(net_path)
    actual_nodes = _estimate_node_grid_from_geo(road_meta, geo_bundle["edge_geo_coords"])
    row_count = len(ROW_NAMES)
    col_count = len(COL_NAMES)

    row_lat = {
        row_idx: sum(actual_nodes[(row_idx, col_idx)][0] for col_idx in range(col_count)) / col_count
        for row_idx in range(row_count)
    }
    col_lon = {
        col_idx: sum(actual_nodes[(row_idx, col_idx)][1] for row_idx in range(row_count)) / row_count
        for col_idx in range(col_count)
    }

    display_nodes = {}
    for row_idx in range(row_count):
        for col_idx in range(col_count):
            actual = actual_nodes[(row_idx, col_idx)]
            display_nodes[(row_idx, col_idx)] = (
                row_lat[row_idx] * 0.84 + actual[0] * 0.16,
                col_lon[col_idx] * 0.84 + actual[1] * 0.16,
            )

    return {
        "reference_bundle": geo_bundle,
        "actual_nodes": actual_nodes,
        "display_nodes": display_nodes,
        "row_lat": row_lat,
        "col_lon": col_lon,
    }


@lru_cache(maxsize=4)
def build_real_map_anchor_corridors(net_path: str, osm_path: str = ""):
    """Build horizontal/vertical anchor corridors over the real-map street skeleton."""
    anchor_grid = build_display_anchor_grid(net_path, osm_path)
    row_corridors = {}
    col_corridors = {}
    for row_idx in range(len(ROW_NAMES)):
        row_corridors[row_idx] = [
            anchor_grid["display_nodes"][(row_idx, col_idx)]
            for col_idx in range(len(COL_NAMES))
        ]
    for col_idx in range(len(COL_NAMES)):
        col_corridors[col_idx] = [
            anchor_grid["display_nodes"][(row_idx, col_idx)]
            for row_idx in range(len(ROW_NAMES))
        ]
    return {
        "horizontal_corridors": row_corridors,
        "vertical_corridors": col_corridors,
        "display_nodes": dict(anchor_grid["display_nodes"]),
        "actual_nodes": dict(anchor_grid["actual_nodes"]),
        "reference_bundle": anchor_grid["reference_bundle"],
    }


@lru_cache(maxsize=4)
def build_block_mapping(net_path: str, osm_path: str = ""):
    """Map logical rows/cols/blocks onto the real-map corridor skeleton."""
    corridor_bundle = build_real_map_anchor_corridors(net_path, osm_path)
    blocks = {}
    for row_idx in range(len(ROW_NAMES) - 1):
        for col_idx in range(len(COL_NAMES) - 1):
            nw = corridor_bundle["display_nodes"][(row_idx, col_idx)]
            ne = corridor_bundle["display_nodes"][(row_idx, col_idx + 1)]
            sw = corridor_bundle["display_nodes"][(row_idx + 1, col_idx)]
            se = corridor_bundle["display_nodes"][(row_idx + 1, col_idx + 1)]
            blocks[(row_idx, col_idx)] = [nw, ne, se, sw]
    return {
        "row_mapping": {row_idx: row_idx for row_idx in range(len(ROW_NAMES))},
        "col_mapping": {col_idx: col_idx for col_idx in range(len(COL_NAMES))},
        "blocks": blocks,
        "corridors": corridor_bundle,
    }


def node_to_display_coord(
    node_key,
    *,
    display_nodes: dict[tuple[int, int], tuple[float, float]] | None = None,
    block_mapping: dict | None = None,
):
    """Return the stable display coordinate for one logical node."""
    if isinstance(node_key, (list, tuple)) and len(node_key) >= 2:
        row_idx, col_idx = int(node_key[0]), int(node_key[1])
    else:
        raise ValueError(f"Unsupported node key: {node_key}")
    nodes = display_nodes or ((block_mapping or {}).get("corridors") or {}).get("display_nodes") or {}
    return nodes.get((row_idx, col_idx))


def edge_to_display_polyline(
    edge_id: str,
    *,
    geo_edge_coords: dict[str, list[tuple[float, float]]],
    display_nodes: dict[tuple[int, int], tuple[float, float]],
    road_meta,
    style: str,
) -> list[tuple[float, float]]:
    meta = road_meta.edge_meta.get(edge_id)
    if meta is None:
        return list(geo_edge_coords.get(edge_id) or [])
    axis = meta.get("axis")
    row_idx = int(meta.get("row", 0))
    col_idx = int(meta.get("col", 0))
    if axis == "H":
        start_key = (row_idx, col_idx)
        end_key = (row_idx, col_idx + 1)
    else:
        start_key = (row_idx, col_idx)
        end_key = (row_idx + 1, col_idx)

    start_node = display_nodes.get(start_key)
    end_node = display_nodes.get(end_key)
    original = list(geo_edge_coords.get(edge_id) or [])
    if style == "clean_schematic":
        if start_node is None or end_node is None:
            return original
        schematic = [start_node, end_node]
        if meta.get("dir_code") in {"W", "N"}:
            schematic.reverse()
        return schematic

    canonical_original = _canonicalize_axis_polyline(original, axis)
    if not canonical_original:
        if start_node is None or end_node is None:
            return []
        canonical_original = [start_node, end_node]
    if start_node is None:
        start_node = canonical_original[0]
    if end_node is None:
        end_node = canonical_original[-1]

    sampled = _resample_polyline(canonical_original, num_points=max(4, len(canonical_original)))
    remapped = []
    for idx, point in enumerate(sampled):
        t = idx / max(len(sampled) - 1, 1)
        anchor_point = _blend_points(start_node, end_node, t)
        remapped.append(_blend_points(anchor_point, point, 0.62))
    remapped[0] = start_node
    remapped[-1] = end_node
    if meta.get("dir_code") in {"W", "N"}:
        remapped.reverse()
    return remapped


def remap_logical_path_to_display_path(
    path: list[str] | tuple[str, ...],
    *,
    edge_geo_coords: dict[str, list[tuple[float, float]]],
    offset_m: float = 0.0,
    smooth_strength: float = 0.0,
) -> list[tuple[float, float]]:
    """Merge logical path edges into one display polyline on the real-road skeleton."""
    merged = logical_path_to_real_display_path(path, edge_geo_coords=edge_geo_coords)
    if abs(offset_m) > 1e-6:
        merged = _offset_polyline_perpendicular(merged, offset_m)
    if smooth_strength > 0.0:
        merged = _smooth_display_path(merged, strength=smooth_strength)
    return merged


@lru_cache(maxsize=4)
def remap_grid_to_realistic_overlay(net_path: str, osm_path: str = ""):
    """Create display geometry by snapping the logical grid onto extracted real-road corridors."""
    geo_bundle = load_geo_edge_bundle(net_path, osm_path)
    mapped = map_logical_grid_to_corridors(net_path, osm_path)
    corridor_bundle = mapped["corridor_bundle"]
    return {
        "center": mapped["center"],
        "fit_bounds": mapped["fit_bounds"],
        "edge_geo_coords": mapped["edge_geo_coords"],
        "node_geo": dict(mapped["node_geo"]),
        "road_midpoints": dict(mapped["road_midpoints"]),
        "block_polygons": list(mapped["block_polygons"]),
        "reference_bundle": geo_bundle,
        "actual_nodes": dict(mapped["node_geo"]),
        "anchor_corridors": corridor_bundle,
        "block_mapping": {
            "row_mapping": {idx: idx for idx in range(len(ROW_NAMES))},
            "col_mapping": {idx: idx for idx in range(len(COL_NAMES))},
            "blocks": {
                (row_idx, col_idx): list(mapped["block_polygons"][row_idx * (len(COL_NAMES) - 1) + col_idx])
                for row_idx in range(len(ROW_NAMES) - 1)
                for col_idx in range(len(COL_NAMES) - 1)
            },
            "corridors": corridor_bundle,
        },
        "debug_layers": {
            "reference_roads": geo_bundle.get("road_way_lines", {}),
            "row_corridors": {
                idx: list(item["coords"])
                for idx, item in corridor_bundle.get("row_corridors", {}).items()
            },
            "col_corridors": {
                idx: list(item["coords"])
                for idx, item in corridor_bundle.get("col_corridors", {}).items()
            },
            "mapped_nodes": dict(mapped["node_geo"]),
            "mapped_edges": dict(mapped["edge_geo_coords"]),
        },
    }


@lru_cache(maxsize=8)
def load_display_geo_bundle(net_path: str, osm_path: str = "", map_style: str = "realistic_overlay"):
    style = _normalize_map_style(map_style)
    if style == "clean_schematic":
        return load_grid_geo_bundle(net_path, osm_path)
    return remap_grid_to_realistic_overlay(net_path, osm_path)


@lru_cache(maxsize=4)
def load_roadnet_visual_meta(net_path: str):
    """Return road label positions and simple building polygons for display layers."""
    meta = load_roadnet_meta(net_path)
    return {
        "road_midpoints": dict(meta.road_midpoints),
        "building_polygons": [list(poly) for poly in meta.building_polygons],
    }


def _add_reference_roads(map_obj, road_way_lines, *, show: bool = True, color: str = "#CBD5E1", opacity: float = 0.18, weight: float = 2.0):
    if not road_way_lines:
        return
    fg = folium.FeatureGroup(name="真实底图参考", show=show)
    for _road_name, lines in road_way_lines.items():
        for coords in lines:
            if len(coords) < 2:
                continue
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=weight,
                opacity=opacity,
                smooth_factor=1.0,
            ).add_to(fg)
    fg.add_to(map_obj)


def _add_grid_blocks(map_obj, block_polygons, *, fill_color: str, fill_opacity: float, line_opacity: float):
    if not block_polygons:
        return
    fg = folium.FeatureGroup(name="城市街区", show=True)
    for poly in block_polygons:
        folium.Polygon(
            locations=poly,
            color="#CBD5E1",
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            weight=1,
            opacity=line_opacity,
        ).add_to(fg)
    fg.add_to(map_obj)


def _add_map_overlays(
    map_obj,
    *,
    map_style: str,
    basemap_style: str = "nolabel_light",
    bundle,
    grid_bundle,
    layer_visibility: dict[str, bool] | None = None,
):
    style = _normalize_map_style(map_style)
    basemap = _normalize_basemap_style(basemap_style)
    ref_bundle = bundle
    layer_visibility = _normalize_layer_visibility(layer_visibility)
    if style == "realistic_overlay":
        if basemap == "labeled_light":
            folium.TileLayer(
                tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                name=BASEMAP_LABELS[basemap],
                attr="CartoDB",
                opacity=0.98,
                show=True,
            ).add_to(map_obj)
        elif basemap == "nolabel_light":
            folium.TileLayer(
                tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
                name=BASEMAP_LABELS[basemap],
                attr="CartoDB",
                opacity=0.96,
                show=True,
            ).add_to(map_obj)
        _add_reference_roads(
            map_obj,
            ref_bundle.get("road_way_lines", {}),
            show=layer_visibility.get("reference_map", True),
            color="#B8C4D4" if basemap == "labeled_light" else "#CBD5E1",
            opacity=0.10 if basemap == "labeled_light" else 0.14,
            weight=1.8,
        )
        _add_grid_blocks(
            map_obj,
            grid_bundle.get("block_polygons", []),
            fill_color="#F8FAFC",
            fill_opacity=0.10 if basemap == "labeled_light" else 0.22 if basemap == "nolabel_light" else 0.06,
            line_opacity=0.05 if basemap == "labeled_light" else 0.08 if basemap == "nolabel_light" else 0.04,
        )
        return

    if basemap == "labeled_light":
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            name=BASEMAP_LABELS[basemap],
            attr="CartoDB",
            opacity=0.40,
            show=True,
        ).add_to(map_obj)
    elif basemap == "nolabel_light":
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
            name=BASEMAP_LABELS[basemap],
            attr="CartoDB",
            opacity=0.28,
            show=True,
        ).add_to(map_obj)
    _add_grid_blocks(
        map_obj,
        grid_bundle.get("block_polygons", []),
        fill_color="#F5F7FA",
        fill_opacity=0.86 if basemap != "minimal" else 0.92,
        line_opacity=0.18 if basemap != "minimal" else 0.12,
    )


def _add_fixed_box(map_obj, html: str):
    map_obj.get_root().html.add_child(Element(html))


def _add_map_legend(
    map_obj,
    *,
    map_style: str,
    basemap_style: str,
    time_metrics: dict[str, float | None],
):
    road_rows = "".join(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
            <span style="display:inline-block;width:26px;height:0;border-top:{cfg['weight']:.1f}px solid {cfg['color']};border-radius:999px;"></span>
            <span style="font-size:12px;color:#334155;">{cfg['label']}</span>
        </div>
        """
        for cfg in (ROAD_TYPE_STYLE["A"], ROAD_TYPE_STYLE["S"], ROAD_TYPE_STYLE["M"])
    )
    congestion_rows = "".join(
        f"""
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
            <span style="display:inline-block;width:26px;height:0;border-top:{cfg['width']-2}px solid {cfg['color']};border-radius:999px;"></span>
            <span style="font-size:12px;color:#334155;">{cfg['label']}</span>
        </div>
        """
        for cfg in (
            CONGESTION_STYLE["clear"],
            CONGESTION_STYLE["light"],
            CONGESTION_STYLE["medium"],
            CONGESTION_STYLE["severe"],
            CONGESTION_STYLE["closed"],
        )
    )
    route_rows = """
        <div style="display:flex;align-items:center;gap:8px;margin:6px 0 2px;">
            <span style="display:inline-block;width:26px;height:0;border-top:4px dashed #DC2626;"></span>
            <span style="font-size:12px;color:#334155;">原始路线</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin:4px 0 0;">
            <span style="display:inline-block;width:26px;height:0;border-top:5px solid #2563EB;"></span>
            <span style="font-size:12px;color:#334155;">规划路线</span>
        </div>
    """
    _add_fixed_box(
        map_obj,
        f"""
        <div style="
            position: fixed;
            bottom: 18px;
            right: 14px;
            z-index: 9999;
            width: 184px;
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 6px 18px rgba(15,23,42,0.10);
            border-radius: 8px;
            padding: 8px 10px;
            backdrop-filter: blur(3px);
        ">
            <div style="font-weight:700;font-size:13px;color:#0F172A;margin-bottom:4px;">地图图例</div>
            <div style="font-size:11px;color:#64748B;margin-bottom:2px;">{MAP_STYLE_LABELS[_normalize_map_style(map_style)]}</div>
            <div style="font-size:11px;color:#64748B;margin-bottom:8px;">{BASEMAP_LABELS[_normalize_basemap_style(basemap_style)]}</div>
            {road_rows}
            {congestion_rows}
            {route_rows}
        </div>
        """,
    )


def _edge_midpoint_latlon(coords: list[tuple[float, float]]):
    if not coords:
        return None
    return coords[len(coords) // 2]


def _render_geo_status(bundle) -> str:
    counts = bundle.get("source_counts", {})
    if bundle.get("mode") == "sumo_geo":
        return "地理模式：直接使用 SUMO net.xml 的投影转换为经纬度。"
    if counts.get("osm_exact") or counts.get("osm_nearest"):
        exact = counts.get("osm_exact", 0)
        near = counts.get("osm_nearest", 0)
        warp = counts.get("synthetic_warp", 0)
        return (
            f"地理模式：本地 map.osm 参考映射。真实道路段 {exact} 条，"
            f"最近道路近似 {near} 条，合成补位 {warp} 条。"
        )
    warp = counts.get("synthetic_warp", 0)
    return f"地理模式：缺少 geo-projection，当前仅使用边界线性映射补位（{warp} 条）。"


def _resolve_focus_edge_ids(
    road_meta,
    *,
    weight_dict: dict[str, float] | None,
    path: list[str],
    baseline_path: list[str],
    focus_path: bool,
) -> set[str]:
    if not focus_path:
        return set(road_meta.edge_meta)
    route_union = set(path) | set(baseline_path)
    if not route_union:
        return set(road_meta.edge_meta)
    focus_edges = set(route_union)
    route_roads = {
        road_meta.edge_meta.get(edge_id, {}).get("road")
        for edge_id in route_union
        if edge_id in road_meta.edge_meta
    }
    for edge_id in list(route_union):
        focus_edges.update(road_meta.line_graph_neighbors.get(edge_id, ()))
    for edge_id, weight in dict(weight_dict or {}).items():
        meta = road_meta.edge_meta.get(edge_id, {})
        if _classify_congestion(weight) != "clear" and meta.get("road") in route_roads:
            focus_edges.add(edge_id)
    return focus_edges


def _normalize_layer_visibility(layer_visibility: dict | None) -> dict[str, bool]:
    defaults = {
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
    for key, value in dict(layer_visibility or {}).items():
        if key in defaults:
            defaults[key] = bool(value)
    return defaults


def _add_mapped_node_layer(map_obj, node_geo: dict[tuple[int, int], tuple[float, float]], *, show: bool):
    fg = folium.FeatureGroup(name="规则图映射节点", show=show)
    for (row_idx, col_idx), point in sorted(node_geo.items()):
        folium.CircleMarker(
            location=point,
            radius=4,
            color="#0F172A",
            weight=1.5,
            fill=True,
            fill_color="#38BDF8",
            fill_opacity=0.92,
            tooltip=f"node ({row_idx}, {col_idx})",
        ).add_to(fg)
    fg.add_to(map_obj)


def _add_mapped_edge_layer(map_obj, road_meta, edge_geo_coords, *, show: bool, focus_edge_ids: set[str]):
    fg = folium.FeatureGroup(name="规则图映射边", show=show)
    for edge_id, coords in edge_geo_coords.items():
        if edge_id not in focus_edge_ids or len(coords) < 2:
            continue
        meta = road_meta.edge_meta.get(edge_id, {})
        axis = meta.get("axis", "?")
        folium.PolyLine(
            locations=coords,
            color="#6366F1" if axis == "H" else "#10B981",
            weight=3,
            opacity=0.52,
            smooth_factor=1.0,
            tooltip=f"{edge_id} · logical {axis}",
        ).add_to(fg)
    fg.add_to(map_obj)


def _add_skeleton_layer(
    map_obj,
    row_corridors: dict[int, list[tuple[float, float]]],
    col_corridors: dict[int, list[tuple[float, float]]],
    *,
    show: bool,
):
    fg = folium.FeatureGroup(name="骨架走廊参考", show=show)
    for _idx, coords in sorted(row_corridors.items()):
        if len(coords) < 2:
            continue
        folium.PolyLine(
            locations=coords,
            color="#1D4ED8",
            weight=3,
            opacity=0.42,
            smooth_factor=1.0,
        ).add_to(fg)
    for _idx, coords in sorted(col_corridors.items()):
        if len(coords) < 2:
            continue
        folium.PolyLine(
            locations=coords,
            color="#059669",
            weight=3,
            opacity=0.42,
            smooth_factor=1.0,
        ).add_to(fg)
    fg.add_to(map_obj)


def _road_type_by_name(road_meta) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for meta in road_meta.edge_meta.values():
        road_name = str(meta.get("road") or "").strip()
        if not road_name:
            continue
        road_type = str(meta.get("rtype") or "M").strip()
        previous = mapping.get(road_name)
        if previous == "A":
            continue
        if previous == "S" and road_type == "M":
            continue
        mapping[road_name] = road_type
    return mapping


def _road_label_candidates(
    road_meta,
    road_midpoints: dict[str, tuple[float, float]],
    *,
    focus_edge_ids: set[str],
    focus_path: bool,
) -> list[tuple[str, tuple[float, float], str]]:
    road_type_map = _road_type_by_name(road_meta)
    focus_roads = {
        str(road_meta.edge_meta.get(edge_id, {}).get("road") or "").strip()
        for edge_id in focus_edge_ids
    }
    focus_roads.discard("")
    labels = []
    for road_name, point in sorted(dict(road_midpoints or {}).items()):
        road_type = road_type_map.get(road_name, "M")
        if focus_path:
            if road_type != "A" and road_name not in focus_roads:
                continue
        else:
            if road_type == "M":
                continue
        labels.append((road_name, point, road_type))
    return labels


def _add_road_label_layer(
    map_obj,
    road_meta,
    road_midpoints: dict[str, tuple[float, float]],
    *,
    focus_edge_ids: set[str],
    focus_path: bool,
    show: bool,
):
    label_candidates = _road_label_candidates(
        road_meta,
        road_midpoints,
        focus_edge_ids=focus_edge_ids,
        focus_path=focus_path,
    )
    if not label_candidates:
        return
    fg = folium.FeatureGroup(name="道路名称", show=show)
    for road_name, point, road_type in label_candidates:
        style = ROAD_LABEL_STYLE.get(road_type, ROAD_LABEL_STYLE["M"])
        folium.map.Marker(
            point,
            icon=folium.DivIcon(
                html=(
                    "<div style=\""
                    f"font-size:{style['font_size']}px;"
                    f"color:{style['color']};"
                    "font-weight:700;"
                    "letter-spacing:0.3px;"
                    "white-space:nowrap;"
                    f"background:{style['bg']};"
                    "padding:1px 6px;"
                    "border-radius:10px;"
                    "box-shadow:0 1px 4px rgba(15,23,42,0.08);"
                    "border:1px solid rgba(148,163,184,0.20);"
                    "\">"
                    f"{road_name}"
                    "</div>"
                )
            ),
        ).add_to(fg)
    fg.add_to(map_obj)


def _add_weight_overlay_layer(
    map_obj,
    edge_geo_coords,
    overlay_weights: dict[str, float],
    *,
    name: str,
    color_resolver,
    tooltip_prefix: str,
    show: bool,
    focus_edge_ids: set[str] | None = None,
    hop_by_edge: dict[str, int] | None = None,
    dashed: bool = False,
):
    fg = folium.FeatureGroup(name=name, show=show)
    for edge_id, weight in sorted(dict(overlay_weights or {}).items()):
        coords = list(edge_geo_coords.get(edge_id) or [])
        if len(coords) < 2:
            continue
        if focus_edge_ids is not None and focus_edge_ids and edge_id not in focus_edge_ids:
            continue
        style = color_resolver(edge_id, float(weight))
        hop_label = ""
        if hop_by_edge and edge_id in hop_by_edge:
            hop_label = f" · hop={int(hop_by_edge[edge_id])}"
        folium.PolyLine(
            locations=coords,
            color=style["color"],
            weight=style["weight"],
            opacity=style["opacity"],
            dash_array="10,6" if dashed else None,
            smooth_factor=1.2,
            tooltip=f"{tooltip_prefix} · {edge_id} · score={float(weight):.2f}{hop_label}",
        ).add_to(fg)
    fg.add_to(map_obj)


def build_folium_map(
    net_path: str,
    *,
    osm_path: str = "",
    weight_dict: dict[str, float] | None = None,
    raw_incident_weights: dict[str, float] | None = None,
    spillover_weights: dict[str, float] | None = None,
    spillover_hops: dict[str, int] | None = None,
    gat_smoothed_weights: dict[str, float] | None = None,
    overlay_stats: dict | None = None,
    path: list[str] | tuple[str, ...] | None = None,
    baseline_path: list[str] | tuple[str, ...] | None = None,
    planned_time_s: float | None = None,
    baseline_time_s: float | None = None,
    map_style: str = "realistic_overlay",
    basemap_style: str = "nolabel_light",
    focus_path: bool = False,
    layer_visibility: dict | None = None,
):
    map_style = _normalize_map_style(map_style)
    basemap_style = _normalize_basemap_style(basemap_style)
    geo_bundle = load_geo_edge_bundle(net_path, osm_path)
    display_bundle = load_display_geo_bundle(net_path, osm_path, map_style)
    debug_layers = dict(display_bundle.get("debug_layers") or {})
    road_meta = load_roadnet_meta(net_path)
    geo_edge_coords = display_bundle["edge_geo_coords"]
    map_center = display_bundle["center"]
    layer_visibility = _normalize_layer_visibility(layer_visibility)
    path = list(path or [])
    baseline_path = list(baseline_path or [])
    raw_incident_weights = dict(raw_incident_weights or {})
    spillover_weights = dict(spillover_weights or {})
    spillover_hops = dict(spillover_hops or {})
    gat_smoothed_weights = dict(gat_smoothed_weights or {})
    path_set = set(path)
    baseline_set = set(baseline_path)
    route_union = path_set | baseline_set
    time_metrics = _route_time_metrics(baseline_time_s, planned_time_s)
    focus_edge_ids = _resolve_focus_edge_ids(
        road_meta,
        weight_dict=weight_dict,
        path=path,
        baseline_path=baseline_path,
        focus_path=focus_path,
    )
    full_network_edge_ids = {
        edge_id
        for edge_id, coords in geo_edge_coords.items()
        if len(coords) >= 2
    }

    folium_map = folium.Map(
        location=map_center,
        zoom_start=14.4,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    _add_map_overlays(
        folium_map,
        map_style=map_style,
        basemap_style=basemap_style,
        bundle=geo_bundle,
        grid_bundle=display_bundle,
        layer_visibility=layer_visibility,
    )

    reference_layer = folium.FeatureGroup(
        name="提取的道路骨架" if map_style == "realistic_overlay" else "示意路网",
        show=layer_visibility.get("road_skeleton", True),
    )
    congestion_layer = folium.FeatureGroup(name="拥堵路段", show=layer_visibility.get("congestion", True))
    baseline_layer = folium.FeatureGroup(name="baseline route", show=bool(baseline_path) and layer_visibility.get("baseline_route", True))
    route_layer = folium.FeatureGroup(name="planned route", show=layer_visibility.get("planned_route", True))

    for edge_id, coords in geo_edge_coords.items():
        if edge_id not in full_network_edge_ids:
            continue
        meta = road_meta.edge_meta.get(edge_id, {})
        road_style = ROAD_TYPE_STYLE.get(meta.get("rtype"), ROAD_TYPE_STYLE["M"])
        folium.PolyLine(
            locations=coords,
            color=road_style["color"],
            weight=road_style["weight"] - (0.2 if map_style == "realistic_overlay" else 0.0),
            opacity=min(road_style["opacity"], 0.70) if map_style == "realistic_overlay" else road_style["opacity"],
            smooth_factor=1.2,
            tooltip=f"{road_style['label']} · {edge_id}",
        ).add_to(reference_layer)

        weight = float(weight_dict.get(edge_id, 2.0)) if weight_dict else 2.0
        level = _classify_congestion(weight)
        if edge_id not in route_union and level == "clear":
            continue
        level_style = _congestion_style(weight)
        folium.PolyLine(
            locations=coords,
            color=level_style["color"],
            weight=level_style["width"] - (1.2 if map_style == "realistic_overlay" else 0.0),
            opacity=0.92 if edge_id in route_union else min(level_style["opacity"], 0.72),
            smooth_factor=1.2,
            tooltip=f"{edge_id} · {level_style['label']} · score={weight:.1f}",
        ).add_to(congestion_layer)

    row_corridors = debug_layers.get("row_corridors", {})
    col_corridors = debug_layers.get("col_corridors", {})
    if (row_corridors or col_corridors) and (layer_visibility.get("mapped_edges", False) or layer_visibility.get("mapped_nodes", False)):
        _add_skeleton_layer(
            folium_map,
            row_corridors=row_corridors,
            col_corridors=col_corridors,
            show=False,
        )

    _add_weight_overlay_layer(
        folium_map,
        geo_edge_coords,
        spillover_weights,
        name="Spillover",
        color_resolver=lambda _edge_id, _weight: {
            "color": "#F59E0B" if int(spillover_hops.get(_edge_id, 2)) == 1 else "#FCD34D",
            "weight": 6 if int(spillover_hops.get(_edge_id, 2)) == 1 else 4,
            "opacity": 0.74 if int(spillover_hops.get(_edge_id, 2)) == 1 else 0.58,
        },
        tooltip_prefix="Spillover",
        show=layer_visibility.get("spillover_layer", True),
        focus_edge_ids=None,
        hop_by_edge=spillover_hops,
    )
    _add_weight_overlay_layer(
        folium_map,
        geo_edge_coords,
        raw_incident_weights,
        name="Raw incident",
        color_resolver=lambda _edge_id, _weight: {
            "color": "#7F1D1D" if float(_weight) >= 8.5 else "#DC2626",
            "weight": 8 if float(_weight) >= 8.5 else 7,
            "opacity": 0.90,
        },
        tooltip_prefix="Raw incident",
        show=layer_visibility.get("raw_incident_layer", True),
        focus_edge_ids=None,
    )
    _add_weight_overlay_layer(
        folium_map,
        geo_edge_coords,
        gat_smoothed_weights,
        name="GAT 平滑层",
        color_resolver=lambda _edge_id, _weight: {
            "color": _congestion_style(_weight)["color"],
            "weight": max(_congestion_style(_weight)["width"] - 1, 4),
            "opacity": 0.62,
        },
        tooltip_prefix="gat smoothed",
        show=layer_visibility.get("gat_smoothed_layer", False),
        focus_edge_ids=None,
        dashed=True,
    )

    for edge_id in baseline_path:
        coords = geo_edge_coords.get(edge_id)
        if not coords or len(coords) < 2:
            continue
        weight = float(weight_dict.get(edge_id, 2.0)) if weight_dict else 2.0
        level = _classify_congestion(weight)
        shifted = _offset_polyline_perpendicular(coords, 4.5 if map_style == "realistic_overlay" else 3.2)
        if level in {"severe", "closed"}:
            folium.PolyLine(
                locations=shifted,
                color="#7F1D1D",
                weight=7 if map_style == "realistic_overlay" else 9,
                opacity=0.16,
                smooth_factor=1.2,
            ).add_to(baseline_layer)

    baseline_line = remap_logical_path_to_display_path(
        baseline_path,
        edge_geo_coords=geo_edge_coords,
        offset_m=4.5 if map_style == "realistic_overlay" else 3.2,
        smooth_strength=0.10 if map_style == "realistic_overlay" else 0.0,
    )
    if len(baseline_line) >= 2:
        folium.PolyLine(
            locations=baseline_line,
            color="#DC2626",
            weight=4.5,
            opacity=0.74,
            dash_array="10,8",
            smooth_factor=1.2,
            tooltip="baseline route",
        ).add_to(baseline_layer)

    planned_line = remap_logical_path_to_display_path(
        path,
        edge_geo_coords=geo_edge_coords,
        offset_m=0.0,
        smooth_strength=0.12 if map_style == "realistic_overlay" else 0.0,
    )
    if len(planned_line) >= 2:
        folium.PolyLine(
            locations=planned_line,
            color="#FFFFFF",
            weight=8 if map_style == "realistic_overlay" else 10,
            opacity=0.74,
            smooth_factor=1.2,
        ).add_to(route_layer)
        folium.PolyLine(
            locations=planned_line,
            color="#2563EB",
            weight=5.2 if map_style == "realistic_overlay" else 6,
            opacity=0.96,
            smooth_factor=1.2,
            tooltip="planned route",
        ).add_to(route_layer)

    if planned_line:
        start_point = planned_line[0]
        end_point = planned_line[-1]
        if start_point:
            folium.CircleMarker(
                location=start_point,
                radius=5,
                color="#FFFFFF",
                weight=2.0,
                fill=True,
                fill_color="#16A34A",
                fill_opacity=0.92,
                tooltip="起点",
            ).add_to(route_layer)
        if end_point:
            folium.CircleMarker(
                location=end_point,
                radius=5,
                color="#FFFFFF",
                weight=2.0,
                fill=True,
                fill_color="#F59E0B",
                fill_opacity=0.92,
                tooltip="终点",
            ).add_to(route_layer)

    if layer_visibility.get("mapped_edges", False):
        _add_mapped_edge_layer(
            folium_map,
            road_meta,
            geo_edge_coords,
            show=True,
            focus_edge_ids=focus_edge_ids,
        )
    if layer_visibility.get("mapped_nodes", False):
        _add_mapped_node_layer(
            folium_map,
            display_bundle.get("node_geo", {}),
            show=True,
        )

    reference_layer.add_to(folium_map)
    congestion_layer.add_to(folium_map)
    if baseline_path:
        baseline_layer.add_to(folium_map)
    route_layer.add_to(folium_map)
    _add_map_legend(
        folium_map,
        map_style=map_style,
        basemap_style=basemap_style,
        time_metrics=time_metrics,
    )
    folium.LayerControl(collapsed=True).add_to(folium_map)

    reference_bounds = display_bundle.get("fit_bounds")
    fit_bounds = display_bundle.get("fit_bounds")
    if focus_path and path:
        path_lines = [line for line in (baseline_line, planned_line) if len(line) >= 2]
        path_bounds = _pad_geo_bounds(_geo_bounds_from_lines(path_lines), lat_ratio=0.16, lon_ratio=0.14, min_pad=0.00045)
        if path_bounds:
            folium_map.fit_bounds([path_bounds[0], path_bounds[1]])
        elif fit_bounds:
            padded = _pad_geo_bounds(fit_bounds, lat_ratio=0.08, lon_ratio=0.08, min_pad=0.00035) or fit_bounds
            folium_map.fit_bounds([padded[0], padded[1]])
    elif fit_bounds:
        padded = _pad_geo_bounds(fit_bounds, lat_ratio=0.06, lon_ratio=0.06, min_pad=0.00030) or fit_bounds
        folium_map.fit_bounds([padded[0], padded[1]])
    elif reference_bounds:
        padded = _pad_geo_bounds(reference_bounds, lat_ratio=0.06, lon_ratio=0.06, min_pad=0.00030) or reference_bounds
        folium_map.fit_bounds([padded[0], padded[1]])

    return folium_map, sanitize_component_args(
        {
            "status_text": (
                f"展示模式：{MAP_STYLE_LABELS[map_style]}。"
                f" Basemap：{BASEMAP_LABELS[basemap_style]}。"
                f" 逻辑规则图已映射到真实道路骨架；baseline=free-flow shortest path，planned=scene-aware route。"
                f" {_render_geo_status(geo_bundle)}"
            ),
            "mode": map_style,
            "basemap_style": basemap_style,
            "source_counts": geo_bundle.get("source_counts", {}),
            "time_metrics": time_metrics,
            "layer_visibility": layer_visibility,
        }
    )


def export_folium_map(map_obj, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    map_obj.save(output_path)
    return output_path
