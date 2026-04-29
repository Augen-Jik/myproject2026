"""Edge-level routing algorithms for the Streamlit path planning UI."""

from __future__ import annotations

import heapq
import logging
import math
from typing import Callable


DEFAULT_EDGE_WEIGHT = 2.0


class RouteResult(dict):
    """Dict-like route result with tuple-unpacking compatibility."""

    def __init__(
        self,
        path_edges=None,
        total_cost: float = math.inf,
        planner_name: str = "",
        diagnostics: dict | None = None,
    ):
        path_edges = list(path_edges or [])
        diagnostics = diagnostics or {}
        total_cost = float(total_cost)
        super().__init__(
            path_edges=path_edges,
            total_cost=total_cost,
            planner_name=planner_name,
            diagnostics=diagnostics,
            path=path_edges,
            success=bool(path_edges) and math.isfinite(total_cost),
        )

    def __iter__(self):
        yield self["path_edges"]
        yield self["total_cost"]


def _route_result(path_edges, total_cost, planner_name, diagnostics=None):
    return RouteResult(
        path_edges=path_edges,
        total_cost=total_cost,
        planner_name=planner_name,
        diagnostics=diagnostics or {},
    )


def _callable_name(func: Callable | None, default: str) -> str:
    if func is None:
        return default
    return getattr(func, "__name__", default)


def resolve_route_query(net, start_eid, end_eid, error_handler=None):
    """Resolve the SUMO start/end edge ids into the node query used by planners."""
    try:
        start_edge = net.getEdge(start_eid)
        end_edge = net.getEdge(end_eid)
        if start_edge is None or end_edge is None:
            return None
        start_node = start_edge.getToNode()
        end_node = end_edge.getFromNode()
        if start_node is None or end_node is None:
            return None
        return start_edge, end_edge, start_node, end_node
    except Exception as exc:
        if error_handler is not None:
            error_handler(f"❌ 路段ID错误：{exc}")
        return None


def iter_outgoing_edges(node):
    """Iterate outgoing edges for either SUMO node API variant."""
    getter = node.getOutgoing if hasattr(node, "getOutgoing") else node.getOutgoingEdges
    return getter()


def weight_to_speed_ms(weight: float) -> float:
    """Map a non-negative congestion weight to an expected travel speed in m/s."""
    return max(1.4, (60.0 - (weight - 1.0) * 6.5) / 3.6)


def edge_travel_time(edge_obj, weight: float) -> float:
    """Convert one edge and one weight into the planner cost in seconds."""
    try:
        length = edge_obj.getLength()
    except Exception:
        length = 300.0
    return length / weight_to_speed_ms(weight)


def edge_freeflow_time(edge_obj) -> float:
    """Estimate one edge's uncongested travel time using the net speed limit."""
    try:
        length = float(edge_obj.getLength())
    except Exception:
        length = 300.0

    speed_ms = None
    try:
        speed_ms = float(edge_obj.getSpeed())
    except Exception:
        speed_ms = None

    if speed_ms is None or speed_ms <= 0:
        lane_speeds = []
        try:
            lane_speeds = [float(lane.getSpeed()) for lane in edge_obj.getLanes() if float(lane.getSpeed()) > 0]
        except Exception:
            lane_speeds = []
        speed_ms = max(lane_speeds) if lane_speeds else (50.0 / 3.6)

    return length / max(speed_ms, 1.0)


def heuristic_time(node, target_node) -> float:
    """Estimate the remaining travel time using straight-line distance."""
    try:
        max_speed_ms = 60.0 / 3.6
        tx, ty = target_node.getCoord()
        nx, ny = node.getCoord()
        return math.sqrt((tx - nx) ** 2 + (ty - ny) ** 2) / max_speed_ms
    except Exception:
        return 0.0


def _dijkstra_like_route(graph, source, target, edge_cost_fn, planner_name, cost_model, error_handler=None):
    query = resolve_route_query(graph, source, target, error_handler=error_handler)
    if query is None:
        return _route_result([], math.inf, planner_name)

    start_edge, _end_edge, start_node, end_node = query
    start_cost = float(edge_cost_fn(start_edge))

    best_cost = {start_node: start_cost}
    seq = 0
    open_heap = [(start_cost, start_cost, seq, start_node, [source])]
    expanded_nodes = 0
    relaxed_edges = 0

    logging.info("Routing start planner=%s source=%s target=%s", planner_name, source, target)

    while open_heap:
        _priority, g_cost, _seq, cur_node, path_edges = heapq.heappop(open_heap)
        if g_cost > best_cost.get(cur_node, math.inf):
            continue

        expanded_nodes += 1
        if cur_node == end_node:
            diagnostics = {
                "expanded_nodes": expanded_nodes,
                "relaxed_edges": relaxed_edges,
                "heuristic": None,
                "cost_model": cost_model,
            }
            logging.info(
                "Routing done planner=%s cost=%.3f hops=%d expanded=%d",
                planner_name,
                g_cost,
                len(path_edges),
                expanded_nodes,
            )
            return _route_result(path_edges, g_cost, planner_name, diagnostics)

        for edge in iter_outgoing_edges(cur_node):
            relaxed_edges += 1
            neighbor = edge.getToNode()
            next_cost = g_cost + float(edge_cost_fn(edge))

            if next_cost >= best_cost.get(neighbor, math.inf):
                continue

            best_cost[neighbor] = next_cost
            seq += 1
            heapq.heappush(
                open_heap,
                (next_cost, next_cost, seq, neighbor, path_edges + [edge.getID()]),
            )

    logging.info("Routing failed planner=%s source=%s target=%s", planner_name, source, target)
    return _route_result(
        [],
        math.inf,
        planner_name,
        {
            "expanded_nodes": expanded_nodes,
            "relaxed_edges": relaxed_edges,
            "heuristic": None,
            "cost_model": cost_model,
        },
    )


def astar_route(graph, source, target, weights, heuristic=None, error_handler=None):
    """Run a real A* search over the SUMO edge graph using travel time as cost."""
    planner_name = "A*"
    query = resolve_route_query(graph, source, target, error_handler=error_handler)
    if query is None:
        return _route_result([], math.inf, planner_name)

    start_edge, _end_edge, start_node, end_node = query
    heuristic_fn = heuristic or heuristic_time
    heuristic_name = _callable_name(heuristic_fn, "heuristic_time")

    start_weight = weights.get(source, DEFAULT_EDGE_WEIGHT)
    start_cost = edge_travel_time(start_edge, start_weight)

    best_cost = {start_node: start_cost}
    seq = 0
    open_heap = [
        (
            start_cost + heuristic_fn(start_node, end_node),
            start_cost,
            seq,
            start_node,
            [source],
        )
    ]
    expanded_nodes = 0
    relaxed_edges = 0

    logging.info(
        "Routing start planner=%s source=%s target=%s heuristic=%s",
        planner_name,
        source,
        target,
        heuristic_name,
    )

    while open_heap:
        _priority, g_cost, _seq, cur_node, path_edges = heapq.heappop(open_heap)
        if g_cost > best_cost.get(cur_node, math.inf):
            continue

        expanded_nodes += 1
        if cur_node == end_node:
            diagnostics = {
                "expanded_nodes": expanded_nodes,
                "relaxed_edges": relaxed_edges,
                "heuristic": heuristic_name,
                "cost_model": "edge travel time (seconds)",
                "ui_weight_assumption": "missing edges default to 2.0",
            }
            logging.info(
                "Routing done planner=%s cost=%.3f hops=%d expanded=%d",
                planner_name,
                g_cost,
                len(path_edges),
                expanded_nodes,
            )
            return _route_result(path_edges, g_cost, planner_name, diagnostics)

        for edge in iter_outgoing_edges(cur_node):
            relaxed_edges += 1
            eid = edge.getID()
            neighbor = edge.getToNode()
            edge_weight = weights.get(eid, DEFAULT_EDGE_WEIGHT)
            next_cost = g_cost + edge_travel_time(edge, edge_weight)

            if next_cost >= best_cost.get(neighbor, math.inf):
                continue

            best_cost[neighbor] = next_cost
            seq += 1
            next_priority = next_cost + heuristic_fn(neighbor, end_node)
            heapq.heappush(
                open_heap,
                (next_priority, next_cost, seq, neighbor, path_edges + [eid]),
            )

    logging.info("Routing failed planner=%s source=%s target=%s", planner_name, source, target)
    return _route_result(
        [],
        math.inf,
        planner_name,
        {
            "expanded_nodes": expanded_nodes,
            "relaxed_edges": relaxed_edges,
            "heuristic": heuristic_name,
            "cost_model": "edge travel time (seconds)",
            "ui_weight_assumption": "missing edges default to 2.0",
        },
    )


def dijkstra_route(graph, source, target, weights, error_handler=None):
    """Run a real Dijkstra search over the SUMO edge graph using travel time as cost."""
    planner_name = "Dijkstra"
    query = resolve_route_query(graph, source, target, error_handler=error_handler)
    if query is None:
        return _route_result([], math.inf, planner_name)

    start_edge, _end_edge, start_node, end_node = query
    start_weight = weights.get(source, DEFAULT_EDGE_WEIGHT)
    start_cost = edge_travel_time(start_edge, start_weight)

    best_cost = {start_node: start_cost}
    seq = 0
    open_heap = [(start_cost, start_cost, seq, start_node, [source])]
    expanded_nodes = 0
    relaxed_edges = 0

    logging.info("Routing start planner=%s source=%s target=%s", planner_name, source, target)

    while open_heap:
        _priority, g_cost, _seq, cur_node, path_edges = heapq.heappop(open_heap)
        if g_cost > best_cost.get(cur_node, math.inf):
            continue

        expanded_nodes += 1
        if cur_node == end_node:
            diagnostics = {
                "expanded_nodes": expanded_nodes,
                "relaxed_edges": relaxed_edges,
                "heuristic": None,
                "cost_model": "edge travel time (seconds)",
                "ui_weight_assumption": "missing edges default to 2.0",
            }
            logging.info(
                "Routing done planner=%s cost=%.3f hops=%d expanded=%d",
                planner_name,
                g_cost,
                len(path_edges),
                expanded_nodes,
            )
            return _route_result(path_edges, g_cost, planner_name, diagnostics)

        for edge in iter_outgoing_edges(cur_node):
            relaxed_edges += 1
            eid = edge.getID()
            neighbor = edge.getToNode()
            edge_weight = weights.get(eid, DEFAULT_EDGE_WEIGHT)
            next_cost = g_cost + edge_travel_time(edge, edge_weight)

            if next_cost >= best_cost.get(neighbor, math.inf):
                continue

            best_cost[neighbor] = next_cost
            seq += 1
            heapq.heappush(
                open_heap,
                (next_cost, next_cost, seq, neighbor, path_edges + [eid]),
            )

    logging.info("Routing failed planner=%s source=%s target=%s", planner_name, source, target)
    return _route_result(
        [],
        math.inf,
        planner_name,
        {
            "expanded_nodes": expanded_nodes,
            "relaxed_edges": relaxed_edges,
            "heuristic": None,
            "cost_model": "edge travel time (seconds)",
            "ui_weight_assumption": "missing edges default to 2.0",
        },
    )


def free_flow_route(graph, source, target, error_handler=None):
    """Run a baseline shortest path using edge free-flow travel time only."""
    return _dijkstra_like_route(
        graph,
        source,
        target,
        edge_cost_fn=edge_freeflow_time,
        planner_name="Free-Flow Baseline",
        cost_model="edge free-flow travel time (seconds)",
        error_handler=error_handler,
    )


def bellman_ford_route(graph, source, target, weights, error_handler=None):
    """Run a real Bellman-Ford shortest path on the SUMO edge graph."""
    planner_name = "Bellman-Ford"
    query = resolve_route_query(graph, source, target, error_handler=error_handler)
    if query is None:
        return _route_result([], math.inf, planner_name)

    start_edge, _end_edge, start_node, end_node = query
    nodes = list(graph.getNodes())
    all_edges = []
    for node in nodes:
        for edge in iter_outgoing_edges(node):
            all_edges.append((node, edge.getToNode(), edge))

    dist = {node: math.inf for node in nodes}
    prev = {}
    start_weight = weights.get(source, DEFAULT_EDGE_WEIGHT)
    dist[start_node] = edge_travel_time(start_edge, start_weight)

    relaxations = 0
    passes_run = 0

    logging.info("Routing start planner=%s source=%s target=%s", planner_name, source, target)

    # Current UI cost is always non-negative travel time, so Bellman-Ford stays here
    # for classical shortest-path comparison rather than because it is efficiency-optimal.
    for pass_idx in range(len(nodes) - 1):
        updated = False
        passes_run = pass_idx + 1
        for src_node, dst_node, edge in all_edges:
            src_dist = dist.get(src_node, math.inf)
            if src_dist == math.inf:
                continue

            relaxations += 1
            eid = edge.getID()
            edge_weight = weights.get(eid, DEFAULT_EDGE_WEIGHT)
            candidate = src_dist + edge_travel_time(edge, edge_weight)
            if candidate >= dist.get(dst_node, math.inf):
                continue

            dist[dst_node] = candidate
            prev[dst_node] = (src_node, eid)
            updated = True

        if not updated:
            break

    negative_cycle_detected = False
    for src_node, dst_node, edge in all_edges:
        src_dist = dist.get(src_node, math.inf)
        if src_dist == math.inf:
            continue
        eid = edge.getID()
        edge_weight = weights.get(eid, DEFAULT_EDGE_WEIGHT)
        if src_dist + edge_travel_time(edge, edge_weight) < dist.get(dst_node, math.inf):
            negative_cycle_detected = True
            break

    if dist.get(end_node, math.inf) == math.inf:
        logging.info("Routing failed planner=%s source=%s target=%s", planner_name, source, target)
        return _route_result(
            [],
            math.inf,
            planner_name,
            {
                "passes_run": passes_run,
                "relaxations": relaxations,
                "negative_cycle_detected": negative_cycle_detected,
                "cost_model": "edge travel time (seconds)",
                "experimental_note": "All UI edge costs are non-negative; Bellman-Ford is kept for algorithm comparison, not efficiency.",
                "ui_weight_assumption": "missing edges default to 2.0",
            },
        )

    rev_path = []
    cur = end_node
    while cur != start_node:
        prev_state = prev.get(cur)
        if prev_state is None:
            return _route_result(
                [],
                math.inf,
                planner_name,
                {
                    "passes_run": passes_run,
                    "relaxations": relaxations,
                    "negative_cycle_detected": negative_cycle_detected,
                    "cost_model": "edge travel time (seconds)",
                    "experimental_note": "All UI edge costs are non-negative; Bellman-Ford is kept for algorithm comparison, not efficiency.",
                    "ui_weight_assumption": "missing edges default to 2.0",
                },
            )
        cur, via_eid = prev_state
        rev_path.append(via_eid)

    rev_path.reverse()
    path_edges = [source] + rev_path
    total_cost = dist[end_node]
    diagnostics = {
        "passes_run": passes_run,
        "relaxations": relaxations,
        "negative_cycle_detected": negative_cycle_detected,
        "cost_model": "edge travel time (seconds)",
        "experimental_note": "All UI edge costs are non-negative; Bellman-Ford is kept for algorithm comparison, not efficiency.",
        "ui_weight_assumption": "missing edges default to 2.0",
    }
    logging.info(
        "Routing done planner=%s cost=%.3f hops=%d passes=%d",
        planner_name,
        total_cost,
        len(path_edges),
        passes_run,
    )
    return _route_result(path_edges, total_cost, planner_name, diagnostics)


def best_first_route(graph, source, target, weights, use_heuristic, error_handler=None, heuristic=None):
    """Compatibility helper that dispatches to A* or Dijkstra explicitly."""
    if use_heuristic:
        return astar_route(
            graph,
            source,
            target,
            weights,
            heuristic=heuristic,
            error_handler=error_handler,
        )
    return dijkstra_route(graph, source, target, weights, error_handler=error_handler)
