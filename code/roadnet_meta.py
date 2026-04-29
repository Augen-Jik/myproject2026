"""Shared 96-edge SUMO road network metadata and helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from functools import lru_cache

try:
    import sumolib
except Exception:  # pragma: no cover
    sumolib = None

SUMO_NET_PATH = "/root/autodl-tmp/SUMO/net/my_net.net.xml"
NORMAL_DEFAULT_WEIGHT = 2.0

ROW_NAMES = ["农业路", "红专路", "政七街", "黄河路", "纬五路"]
COL_NAMES = ["经一路", "经三路", "经六路", "经八路", "花园路", "未来路"]
ROW_MAP = {name: f"R{idx}" for idx, name in enumerate(ROW_NAMES)}
COL_MAP = {name: f"C{idx}" for idx, name in enumerate(COL_NAMES)}
ROW_TYPE = ["A", "S", "M", "A", "M"]
COL_TYPE = ["M", "S", "A", "S", "A", "M"]
TYPE2ID = {"A": 0, "S": 1, "M": 2}
DIR_CN = {"E": "向东", "W": "向西", "N": "向北", "S": "向南"}
DIR_EN = {"向东": "E", "东向": "E", "向西": "W", "西向": "W", "向北": "N", "北向": "N", "向南": "S", "南向": "S"}
DIR_OPP = {"E": "W", "W": "E", "N": "S", "S": "N"}
RANGE_FALLBACKS = {"", "局部", "全文", "未指明", "附近", "周边"}


@dataclass(frozen=True)
class RoadNetMeta:
    net_path: str
    edge_ids: tuple[str, ...]
    edge_index: dict[str, int]
    edge_meta: dict[str, dict]
    node_coords: dict[str, tuple[float, float]]
    edge_shapes: dict[str, list[tuple[float, float]]]
    road_midpoints: dict[str, tuple[float, float]]
    building_polygons: tuple[list[tuple[float, float]], ...]
    line_graph_neighbors: dict[str, tuple[str, ...]]


def _build_static_meta() -> dict[str, dict]:
    edge_meta: dict[str, dict] = {}
    for row_idx in range(len(ROW_NAMES)):
        for col_idx in range(len(COL_NAMES) - 1):
            for d in ("E", "W"):
                eid = f"R{row_idx}C{col_idx}_{d}"
                edge_meta[eid] = {
                    "axis": "H",
                    "row": row_idx,
                    "col": col_idx,
                    "road": ROW_NAMES[row_idx],
                    "segment": f"{COL_NAMES[col_idx]}至{COL_NAMES[col_idx + 1]}",
                    "dir_code": d,
                    "dir": DIR_CN[d],
                    "rtype": ROW_TYPE[row_idx],
                    "start_idx": col_idx,
                    "end_idx": col_idx + 1,
                    "opp": f"R{row_idx}C{col_idx}_{DIR_OPP[d]}",
                    "road_kind": "row",
                }

    for col_idx in range(len(COL_NAMES)):
        for row_idx in range(len(ROW_NAMES) - 1):
            for d in ("N", "S"):
                eid = f"C{col_idx}R{row_idx}_{d}"
                edge_meta[eid] = {
                    "axis": "V",
                    "row": row_idx,
                    "col": col_idx,
                    "road": COL_NAMES[col_idx],
                    "segment": f"{ROW_NAMES[row_idx]}至{ROW_NAMES[row_idx + 1]}",
                    "dir_code": d,
                    "dir": DIR_CN[d],
                    "rtype": COL_TYPE[col_idx],
                    "start_idx": row_idx,
                    "end_idx": row_idx + 1,
                    "opp": f"C{col_idx}R{row_idx}_{DIR_OPP[d]}",
                    "road_kind": "col",
                }
    return edge_meta


STATIC_EDGE_META = _build_static_meta()


def _extract_node_coords(net) -> dict[str, tuple[float, float]]:
    coords = {}
    for node in net.getNodes():
        try:
            coords[node.getID()] = tuple(float(v) for v in node.getCoord())
        except Exception:
            continue
    return coords


def _extract_edge_shapes(net) -> dict[str, list[tuple[float, float]]]:
    shapes = {}
    for edge in net.getEdges():
        try:
            shape = edge.getShape()
            if shape:
                shapes[edge.getID()] = [(float(x), float(y)) for x, y in shape]
        except Exception:
            continue
    return shapes


def _midpoint(coords: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> tuple[float, float]:
    if not coords:
        return (0.0, 0.0)
    pt = coords[len(coords) // 2]
    return float(pt[0]), float(pt[1])


def _build_road_midpoints(edge_shapes: dict[str, list[tuple[float, float]]], edge_meta: dict[str, dict]) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for eid, meta in edge_meta.items():
        if eid not in edge_shapes:
            continue
        grouped.setdefault(meta["road"], []).append(_midpoint(edge_shapes[eid]))
    out = {}
    for road, pts in grouped.items():
        if not pts:
            continue
        out[road] = (
            sum(pt[0] for pt in pts) / len(pts),
            sum(pt[1] for pt in pts) / len(pts),
        )
    return out


def _build_building_polygons(node_coords: dict[str, tuple[float, float]]) -> tuple[list[tuple[float, float]], ...]:
    xs = sorted({round(x, 3) for x, _ in node_coords.values()})
    ys = sorted({round(y, 3) for _, y in node_coords.values()})
    polygons: list[list[tuple[float, float]]] = []
    if len(xs) < 2 or len(ys) < 2:
        return tuple(polygons)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        dx = x1 - x0
        for y0, y1 in zip(ys[:-1], ys[1:]):
            dy = y1 - y0
            margin_x = max(dx * 0.18, 28.0)
            margin_y = max(dy * 0.18, 28.0)
            polygons.append([
                (x0 + margin_x, y0 + margin_y),
                (x1 - margin_x, y0 + margin_y),
                (x1 - margin_x, y1 - margin_y),
                (x0 + margin_x, y1 - margin_y),
            ])
    return tuple(polygons)


def _build_line_graph_neighbors(edge_ids: list[str], edge_shapes: dict[str, list[tuple[float, float]]]) -> dict[str, tuple[str, ...]]:
    node_to_edges: dict[tuple[float, float], list[str]] = {}
    for eid in edge_ids:
        coords = edge_shapes.get(eid)
        if not coords:
            continue
        for pt in (coords[0], coords[-1]):
            key = (round(pt[0], 3), round(pt[1], 3))
            node_to_edges.setdefault(key, []).append(eid)

    neighbors: dict[str, set[str]] = {eid: set() for eid in edge_ids}
    for linked in node_to_edges.values():
        for src in linked:
            for dst in linked:
                if src != dst:
                    neighbors[src].add(dst)
    return {eid: tuple(sorted(items)) for eid, items in neighbors.items()}


@lru_cache(maxsize=4)
def load_roadnet_meta(net_path: str = SUMO_NET_PATH) -> RoadNetMeta:
    edge_meta = dict(STATIC_EDGE_META)
    edge_ids = sorted(edge_meta)
    node_coords: dict[str, tuple[float, float]] = {}
    edge_shapes: dict[str, list[tuple[float, float]]] = {}

    if sumolib is not None and os.path.exists(net_path):
        net = sumolib.net.readNet(net_path)
        live_edge_ids = [edge.getID() for edge in net.getEdges() if edge is not None]
        live_set = set(live_edge_ids)
        edge_ids = [eid for eid in edge_ids if eid in live_set]
        edge_meta = {eid: edge_meta[eid] for eid in edge_ids}
        node_coords = _extract_node_coords(net)
        edge_shapes = _extract_edge_shapes(net)

    road_midpoints = _build_road_midpoints(edge_shapes, edge_meta)
    building_polygons = _build_building_polygons(node_coords)
    neighbors = _build_line_graph_neighbors(edge_ids, edge_shapes)
    return RoadNetMeta(
        net_path=net_path,
        edge_ids=tuple(edge_ids),
        edge_index={eid: idx for idx, eid in enumerate(edge_ids)},
        edge_meta=edge_meta,
        node_coords=node_coords,
        edge_shapes=edge_shapes,
        road_midpoints=road_midpoints,
        building_polygons=building_polygons,
        line_graph_neighbors=neighbors,
    )


def live_edge_ids(net_path: str = SUMO_NET_PATH) -> list[str]:
    return list(load_roadnet_meta(net_path).edge_ids)


def normalize_dir_label(token: str | None) -> str:
    if not token:
        return "未指明"
    value = str(token).strip()
    if value in DIR_EN:
        return DIR_CN[DIR_EN[value]]
    if value in DIR_CN.values():
        return value
    if value in {"双向", "全向"}:
        return "双向"
    return value or "未指明"


def edge_road_name(edge_id: str) -> str:
    meta = STATIC_EDGE_META.get(edge_id)
    return meta["road"] if meta else edge_id


def expand_anchor_edges(
    road: str,
    dir_label: str,
    range_label: str,
    *,
    net_path: str = SUMO_NET_PATH,
) -> list[str]:
    meta = load_roadnet_meta(net_path)
    edge_set = set(meta.edge_ids)
    dir_label = normalize_dir_label(dir_label)
    range_label = str(range_label or "局部").strip()

    if road in ROW_MAP:
        row_key = ROW_MAP[road]
        if dir_label == "向东":
            dirs = ["E"]
        elif dir_label == "向西":
            dirs = ["W"]
        else:
            dirs = ["E", "W"]
        cols = range(len(COL_NAMES) - 1)
        if range_label not in RANGE_FALLBACKS and range_label != "全线":
            match = re.search(r"(经一路|经三路|经六路|经八路|花园路|未来路)至(经一路|经三路|经六路|经八路|花园路|未来路)", range_label)
            if match:
                c0 = int(COL_MAP[match.group(1)][1:])
                c1 = int(COL_MAP[match.group(2)][1:])
                lo, hi = min(c0, c1), max(c0, c1)
                cols = range(lo, hi)
        return [eid for eid in (f"{row_key}C{col_idx}_{d}" for col_idx in cols for d in dirs) if eid in edge_set]

    if road in COL_MAP:
        col_key = COL_MAP[road]
        if dir_label == "向北":
            dirs = ["N"]
        elif dir_label == "向南":
            dirs = ["S"]
        else:
            dirs = ["N", "S"]
        rows = range(len(ROW_NAMES) - 1)
        if range_label not in RANGE_FALLBACKS and range_label != "全线":
            match = re.search(r"(农业路|红专路|政七街|黄河路|纬五路)至(农业路|红专路|政七街|黄河路|纬五路)", range_label)
            if match:
                r0 = int(ROW_MAP[match.group(1)][1:])
                r1 = int(ROW_MAP[match.group(2)][1:])
                lo, hi = min(r0, r1), max(r0, r1)
                rows = range(lo, hi)
        return [eid for eid in (f"{col_key}R{row_idx}_{d}" for row_idx in rows for d in dirs) if eid in edge_set]

    return []
