from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache

import networkx as nx

from roadnet_meta import ROW_NAMES, COL_NAMES, load_roadnet_meta


def _geo_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    x = (lon2 - lon1) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    y = (lat2 - lat1) * 110540.0
    return (x * x + y * y) ** 0.5


def _polyline_length(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    return sum(_geo_distance_m(a, b) for a, b in zip(coords, coords[1:]))


def _polyline_bounds(coords: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not coords:
        return None
    lats = [pt[0] for pt in coords]
    lons = [pt[1] for pt in coords]
    return min(lats), min(lons), max(lats), max(lons)


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(lat for lat, _ in points) / len(points),
        sum(lon for _, lon in points) / len(points),
    )


def _point_to_xy(pt: tuple[float, float]) -> tuple[float, float]:
    return pt[1], pt[0]


def _segment_intersection(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> tuple[float, float] | None:
    x1, y1 = _point_to_xy(a)
    x2, y2 = _point_to_xy(b)
    x3, y3 = _point_to_xy(c)
    x4, y4 = _point_to_xy(d)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

    def _within(value: float, left: float, right: float) -> bool:
        return min(left, right) - 1e-9 <= value <= max(left, right) + 1e-9

    if (
        _within(px, x1, x2)
        and _within(py, y1, y2)
        and _within(px, x3, x4)
        and _within(py, y3, y4)
    ):
        return (py, px)
    return None


def _project_point_to_segment(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    lat0, lon0 = point
    lat1, lon1 = a
    lat2, lon2 = b
    dx = lon2 - lon1
    dy = lat2 - lat1
    denom = (dx * dx) + (dy * dy)
    if denom <= 1e-15:
        return a, 0.0
    t = ((lon0 - lon1) * dx + (lat0 - lat1) * dy) / denom
    t = min(max(t, 0.0), 1.0)
    return (lat1 + dy * t, lon1 + dx * t), t


def _nearest_point_between_polylines(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    best = None
    for src in left:
        for a, b in zip(right, right[1:]):
            proj, _ = _project_point_to_segment(src, a, b)
            dist = _geo_distance_m(src, proj)
            if best is None or dist < best[2]:
                best = (src, proj, dist)
    for src in right:
        for a, b in zip(left, left[1:]):
            proj, _ = _project_point_to_segment(src, a, b)
            dist = _geo_distance_m(src, proj)
            if best is None or dist < best[2]:
                best = (proj, src, dist)
    return best


def _polyline_position(
    coords: list[tuple[float, float]],
    point: tuple[float, float],
) -> tuple[int, float]:
    if len(coords) < 2:
        return 0, 0.0
    best = (0, 0.0, float("inf"))
    for idx, (a, b) in enumerate(zip(coords, coords[1:])):
        proj, t = _project_point_to_segment(point, a, b)
        dist = _geo_distance_m(point, proj)
        if dist < best[2]:
            best = (idx, t, dist)
    return best[0], best[1]


def _extract_polyline_between_positions(
    coords: list[tuple[float, float]],
    start_pos: tuple[int, float],
    end_pos: tuple[int, float],
) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return list(coords)

    def _interp(idx: int, t: float) -> tuple[float, float]:
        a = coords[idx]
        b = coords[idx + 1]
        return (
            a[0] * (1.0 - t) + b[0] * t,
            a[1] * (1.0 - t) + b[1] * t,
        )

    start_idx, start_t = start_pos
    end_idx, end_t = end_pos
    reverse = (start_idx, start_t) > (end_idx, end_t)
    if reverse:
        start_idx, start_t, end_idx, end_t = end_idx, end_t, start_idx, start_t

    line = [_interp(start_idx, start_t)]
    for idx in range(start_idx + 1, end_idx + 1):
        line.append(coords[idx])
    line.append(_interp(end_idx, end_t))

    deduped: list[tuple[float, float]] = []
    for pt in line:
        if deduped and _geo_distance_m(deduped[-1], pt) < 0.25:
            deduped[-1] = pt
        else:
            deduped.append(pt)
    if reverse:
        deduped.reverse()
    return deduped


def _merge_path_polylines(
    path: list[str] | tuple[str, ...],
    edge_geo_coords: dict[str, list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for edge_id in list(path or []):
        coords = list(edge_geo_coords.get(edge_id) or [])
        if not coords:
            continue
        if merged:
            start_gap = _geo_distance_m(merged[-1], coords[0])
            end_gap = _geo_distance_m(merged[-1], coords[-1])
            if end_gap + 1e-6 < start_gap:
                coords.reverse()
                start_gap = end_gap

            if start_gap <= 22.0:
                anchor = (
                    (merged[-1][0] + coords[0][0]) / 2.0,
                    (merged[-1][1] + coords[0][1]) / 2.0,
                )
                merged[-1] = anchor
                coords[0] = anchor
            if merged and _geo_distance_m(merged[-1], coords[0]) < 0.5:
                merged.extend(coords[1:])
            else:
                merged.extend(coords)
        else:
            merged.extend(coords)
    deduped: list[tuple[float, float]] = []
    for pt in merged:
        if deduped and _geo_distance_m(deduped[-1], pt) < 0.25:
            deduped[-1] = pt
        else:
            deduped.append(pt)
    return deduped


@lru_cache(maxsize=4)
def _load_named_osm_roads(osm_path: str):
    if not osm_path or not os.path.exists(osm_path):
        return {
            "bounds": None,
            "node_latlon": {},
            "road_way_lines": {},
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

    road_way_lines = defaultdict(list)
    road_graphs = defaultdict(nx.Graph)

    for way in root.findall("way"):
        tags = {tag.attrib.get("k"): tag.attrib.get("v") for tag in way.findall("tag")}
        road_name = str(tags.get("name") or "").strip()
        if not road_name or not tags.get("highway"):
            continue

        refs = [nd.attrib.get("ref") for nd in way.findall("nd")]
        refs = [ref for ref in refs if ref in node_latlon]
        if len(refs) < 2:
            continue

        coords = [node_latlon[ref] for ref in refs]
        road_way_lines[road_name].append(coords)

        graph = road_graphs[road_name]
        for ref in refs:
            graph.add_node(ref)
        for src, dst in zip(refs, refs[1:]):
            graph.add_edge(src, dst, weight=_geo_distance_m(node_latlon[src], node_latlon[dst]))

    return {
        "bounds": bounds,
        "node_latlon": node_latlon,
        "road_way_lines": {name: tuple(lines) for name, lines in road_way_lines.items()},
        "road_graphs": dict(road_graphs),
    }


def _representative_road_polyline(
    road_name: str,
    graph,
    node_latlon: dict[str, tuple[float, float]],
):
    best = None
    for component_nodes in nx.connected_components(graph):
        comp_nodes = list(component_nodes)
        points = [node_latlon[nid] for nid in comp_nodes if nid in node_latlon]
        if len(points) < 2:
            continue
        min_lat = min(pt[0] for pt in points)
        max_lat = max(pt[0] for pt in points)
        min_lon = min(pt[1] for pt in points)
        max_lon = max(pt[1] for pt in points)
        lat_span = max_lat - min_lat
        lon_span = max_lon - min_lon
        axis = "H" if lon_span >= lat_span else "V"

        if axis == "H":
            start_node = min(comp_nodes, key=lambda nid: node_latlon[nid][1])
            end_node = max(comp_nodes, key=lambda nid: node_latlon[nid][1])
        else:
            start_node = min(comp_nodes, key=lambda nid: node_latlon[nid][0])
            end_node = max(comp_nodes, key=lambda nid: node_latlon[nid][0])

        sub_graph = graph.subgraph(comp_nodes)
        try:
            path_nodes = nx.shortest_path(sub_graph, start_node, end_node, weight="weight")
        except Exception:
            continue
        coords = [node_latlon[nid] for nid in path_nodes if nid in node_latlon]
        if len(coords) < 2:
            continue

        score = max(lon_span, lat_span)
        candidate = {
            "name": road_name,
            "axis": axis,
            "coords": coords,
            "center_lat": (min_lat + max_lat) / 2.0,
            "center_lon": (min_lon + max_lon) / 2.0,
            "lat_span": lat_span,
            "lon_span": lon_span,
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


@lru_cache(maxsize=4)
def _build_representative_roads(osm_path: str):
    osm_data = _load_named_osm_roads(osm_path)
    node_latlon = osm_data["node_latlon"]
    candidates = []
    for road_name, graph in osm_data["road_graphs"].items():
        item = _representative_road_polyline(road_name, graph, node_latlon)
        if item is not None:
            candidates.append(item)
    return {
        "bounds": osm_data["bounds"],
        "road_way_lines": osm_data["road_way_lines"],
        "candidates": tuple(candidates),
    }


def _target_focus_window(candidates: list[dict], names: list[str], axis: str):
    key = "center_lat" if axis == "H" else "center_lon"
    matched = [item[key] for item in candidates if item["name"] in names and item["axis"] == axis]
    if matched:
        lo = min(matched)
        hi = max(matched)
        pad = 0.003 if axis == "H" else 0.005
        return lo - pad, hi + pad
    values = [item[key] for item in candidates if item["axis"] == axis]
    if not values:
        return (0.0, 0.0)
    return min(values), max(values)


def _dedupe_candidates(candidates: list[dict], axis: str) -> list[dict]:
    key = "center_lat" if axis == "H" else "center_lon"
    span_key = "lon_span" if axis == "H" else "lat_span"
    tolerance = 0.0012 if axis == "H" else 0.0018
    ordered = sorted(candidates, key=lambda item: item[key], reverse=(axis == "H"))
    deduped: list[dict] = []
    for item in ordered:
        if not deduped:
            deduped.append(item)
            continue
        if abs(item[key] - deduped[-1][key]) <= tolerance:
            if item[span_key] > deduped[-1][span_key]:
                deduped[-1] = item
            continue
        deduped.append(item)
    return deduped


def _reduce_to_count(candidates: list[dict], count: int, axis: str) -> list[dict]:
    key = "center_lat" if axis == "H" else "center_lon"
    span_key = "lon_span" if axis == "H" else "lat_span"
    reduced = list(candidates)
    while len(reduced) > count:
        diffs = [abs(reduced[idx][key] - reduced[idx + 1][key]) for idx in range(len(reduced) - 1)]
        pair_idx = min(range(len(diffs)), key=diffs.__getitem__)
        left = reduced[pair_idx]
        right = reduced[pair_idx + 1]
        drop_idx = pair_idx if left[span_key] < right[span_key] else pair_idx + 1
        reduced.pop(drop_idx)
    return reduced


def _choose_axis_corridors(
    all_candidates: list[dict],
    *,
    axis: str,
    count: int,
    focus_names: list[str],
) -> list[dict]:
    focus_min, focus_max = _target_focus_window(all_candidates, focus_names, axis)
    axis_candidates = []
    for item in all_candidates:
        if item["axis"] != axis:
            continue
        center = item["center_lat"] if axis == "H" else item["center_lon"]
        span = item["lon_span"] if axis == "H" else item["lat_span"]
        if axis == "H" and span < 0.020:
            continue
        if axis == "V" and span < 0.024:
            continue
        if center < focus_min or center > focus_max:
            continue
        axis_candidates.append(item)

    axis_candidates = _dedupe_candidates(axis_candidates, axis)
    if len(axis_candidates) < count:
        broader = [item for item in all_candidates if item["axis"] == axis]
        axis_candidates = _dedupe_candidates(broader, axis)
    axis_candidates = _reduce_to_count(axis_candidates, count, axis)
    return sorted(axis_candidates, key=lambda item: item["center_lat"], reverse=True) if axis == "H" else sorted(axis_candidates, key=lambda item: item["center_lon"])


def _find_intersection_point(
    row_coords: list[tuple[float, float]],
    col_coords: list[tuple[float, float]],
) -> tuple[float, float]:
    for a, b in zip(row_coords, row_coords[1:]):
        for c, d in zip(col_coords, col_coords[1:]):
            inter = _segment_intersection(a, b, c, d)
            if inter is not None:
                return inter
    nearest = _nearest_point_between_polylines(row_coords, col_coords)
    if nearest is None:
        row_mid = _mean_point(row_coords) or (0.0, 0.0)
        col_mid = _mean_point(col_coords) or row_mid
        return (
            (row_mid[0] + col_mid[0]) / 2.0,
            (row_mid[1] + col_mid[1]) / 2.0,
        )
    row_pt, col_pt, _dist = nearest
    return (
        (row_pt[0] + col_pt[0]) / 2.0,
        (row_pt[1] + col_pt[1]) / 2.0,
    )


def _optimize_corridor_compatibility(
    selected_rows: list[dict],
    selected_cols: list[dict],
    row_pool: list[dict],
    col_pool: list[dict],
) -> tuple[list[dict], list[dict]]:
    rows = list(selected_rows)
    cols = list(selected_cols)

    for _ in range(2):
        for row_idx, current in enumerate(list(rows)):
            current_hits = sum(
                1
                for col in cols
                if _find_intersection_point(current["coords"], col["coords"]) is not None
            )
            best = current
            best_score = current_hits * 100.0 + current["lon_span"]
            for candidate in row_pool:
                if candidate in rows and candidate is not current:
                    continue
                prev_lat = rows[row_idx - 1]["center_lat"] if row_idx > 0 else None
                next_lat = rows[row_idx + 1]["center_lat"] if row_idx + 1 < len(rows) else None
                if prev_lat is not None and candidate["center_lat"] > prev_lat:
                    continue
                if next_lat is not None and candidate["center_lat"] < next_lat:
                    continue
                hit_count = sum(
                    1
                    for col in cols
                    if _find_intersection_point(candidate["coords"], col["coords"]) is not None
                )
                closeness = abs(candidate["center_lat"] - current["center_lat"])
                score = hit_count * 100.0 + candidate["lon_span"] - (closeness * 1000.0)
                if score > best_score:
                    best = candidate
                    best_score = score
            rows[row_idx] = best

        for col_idx, current in enumerate(list(cols)):
            current_hits = sum(
                1
                for row in rows
                if _find_intersection_point(row["coords"], current["coords"]) is not None
            )
            best = current
            best_score = current_hits * 100.0 + current["lat_span"]
            for candidate in col_pool:
                if candidate in cols and candidate is not current:
                    continue
                prev_lon = cols[col_idx - 1]["center_lon"] if col_idx > 0 else None
                next_lon = cols[col_idx + 1]["center_lon"] if col_idx + 1 < len(cols) else None
                if prev_lon is not None and candidate["center_lon"] < prev_lon:
                    continue
                if next_lon is not None and candidate["center_lon"] > next_lon:
                    continue
                hit_count = sum(
                    1
                    for row in rows
                    if _find_intersection_point(row["coords"], candidate["coords"]) is not None
                )
                closeness = abs(candidate["center_lon"] - current["center_lon"])
                score = hit_count * 100.0 + candidate["lat_span"] - (closeness * 1000.0)
                if score > best_score:
                    best = candidate
                    best_score = score
            cols[col_idx] = best

    return rows, cols


@lru_cache(maxsize=4)
def build_real_road_corridors(net_path: str, osm_path: str):
    road_meta = load_roadnet_meta(net_path)
    row_count = len(ROW_NAMES)
    col_count = len(COL_NAMES)
    rep = _build_representative_roads(osm_path)
    candidates = list(rep["candidates"])

    if not candidates:
        raise FileNotFoundError(f"No usable named OSM road candidates found: {osm_path}")

    row_pool = _choose_axis_corridors(
        candidates,
        axis="H",
        count=max(row_count, 5),
        focus_names=["农业路", "红专路", "黄河路", "纬五路"],
    )
    col_pool = _choose_axis_corridors(
        candidates,
        axis="V",
        count=max(col_count, 6),
        focus_names=["经八路", "经六路", "花园路", "经三路", "经一路"],
    )

    rows = _reduce_to_count(row_pool, row_count, "H")
    cols = _reduce_to_count(col_pool, col_count, "V")
    rows, cols = _optimize_corridor_compatibility(rows, cols, row_pool, col_pool)

    selected_rows = {idx: item for idx, item in enumerate(rows)}
    selected_cols = {idx: item for idx, item in enumerate(cols)}

    intersections = {}
    node_coords = {}
    row_positions = {}
    col_positions = {}
    for row_idx, row_item in selected_rows.items():
        row_positions[row_idx] = {}
        for col_idx, col_item in selected_cols.items():
            point = _find_intersection_point(row_item["coords"], col_item["coords"])
            intersections[(row_idx, col_idx)] = {
                "point": point,
                "row_pos": _polyline_position(row_item["coords"], point),
                "col_pos": _polyline_position(col_item["coords"], point),
            }
            node_coords[(row_idx, col_idx)] = point

    block_polygons = []
    for row_idx in range(row_count - 1):
        for col_idx in range(col_count - 1):
            nw = node_coords[(row_idx, col_idx)]
            ne = node_coords[(row_idx, col_idx + 1)]
            sw = node_coords[(row_idx + 1, col_idx)]
            se = node_coords[(row_idx + 1, col_idx + 1)]
            block_polygons.append([nw, ne, se, sw])

    all_points = list(node_coords.values())
    center = _mean_point(all_points) or (34.78, 113.67)
    if all_points:
        min_lat = min(pt[0] for pt in all_points)
        max_lat = max(pt[0] for pt in all_points)
        min_lon = min(pt[1] for pt in all_points)
        max_lon = max(pt[1] for pt in all_points)
        fit_bounds = ((min_lat, min_lon), (max_lat, max_lon))
    else:
        fit_bounds = None

    return {
        "center": center,
        "fit_bounds": fit_bounds,
        "row_corridors": selected_rows,
        "col_corridors": selected_cols,
        "node_coords": node_coords,
        "intersections": intersections,
        "block_polygons": block_polygons,
        "reference_bounds": rep["bounds"],
        "reference_way_lines": rep["road_way_lines"],
    }


@lru_cache(maxsize=4)
def map_logical_grid_to_corridors(net_path: str, osm_path: str):
    corridor_bundle = build_real_road_corridors(net_path, osm_path)
    road_meta = load_roadnet_meta(net_path)
    edge_geo_coords = {}
    for edge_id in road_meta.edge_ids:
        polyline = edge_to_real_display_polyline(
            edge_id,
            corridor_bundle=corridor_bundle,
            road_meta=road_meta,
        )
        if polyline:
            edge_geo_coords[edge_id] = polyline

    road_midpoints = {}
    grouped = defaultdict(list)
    for edge_id, coords in edge_geo_coords.items():
        meta = road_meta.edge_meta.get(edge_id, {})
        if not coords:
            continue
        mid = coords[len(coords) // 2]
        grouped[meta.get("road", edge_id)].append(mid)
    for road_name, points in grouped.items():
        mean_pt = _mean_point(points)
        if mean_pt is not None:
            road_midpoints[road_name] = mean_pt

    return {
        "center": corridor_bundle["center"],
        "fit_bounds": corridor_bundle["fit_bounds"],
        "edge_geo_coords": edge_geo_coords,
        "node_geo": corridor_bundle["node_coords"],
        "road_midpoints": road_midpoints,
        "block_polygons": corridor_bundle["block_polygons"],
        "corridor_bundle": corridor_bundle,
    }


def node_to_real_display_coord(
    node_key,
    *,
    corridor_bundle: dict,
):
    if not isinstance(node_key, (list, tuple)) or len(node_key) < 2:
        raise ValueError(f"Unsupported node key: {node_key}")
    return corridor_bundle["node_coords"].get((int(node_key[0]), int(node_key[1])))


def edge_to_real_display_polyline(
    edge_id: str,
    *,
    corridor_bundle: dict,
    road_meta,
) -> list[tuple[float, float]]:
    meta = road_meta.edge_meta.get(edge_id)
    if meta is None:
        return []

    axis = meta.get("axis")
    row_idx = int(meta.get("row", 0))
    col_idx = int(meta.get("col", 0))
    if axis == "H":
        start_key = (row_idx, col_idx)
        end_key = (row_idx, col_idx + 1)
        corridor = corridor_bundle["row_corridors"][row_idx]["coords"]
        start_pos = corridor_bundle["intersections"][start_key]["row_pos"]
        end_pos = corridor_bundle["intersections"][end_key]["row_pos"]
    else:
        start_key = (row_idx, col_idx)
        end_key = (row_idx + 1, col_idx)
        corridor = corridor_bundle["col_corridors"][col_idx]["coords"]
        start_pos = corridor_bundle["intersections"][start_key]["col_pos"]
        end_pos = corridor_bundle["intersections"][end_key]["col_pos"]

    coords = _extract_polyline_between_positions(corridor, start_pos, end_pos)
    if meta.get("dir_code") in {"W", "S"}:
        coords.reverse()
    return coords


def logical_path_to_real_display_path(
    path: list[str] | tuple[str, ...],
    *,
    edge_geo_coords: dict[str, list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    return _merge_path_polylines(path, edge_geo_coords)
