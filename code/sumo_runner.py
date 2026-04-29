import heapq
import logging
import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sumolib
import traci


LOGGER = logging.getLogger(__name__)

DEFAULT_PLANNER_VTYPE = "car"
DEFAULT_PLANNER_VCLASS = "passenger"
SPECIAL_EDGE_FUNCTIONS = {"internal", "crossing", "walkingarea", "connector"}


def _shape_midpoint(edge_obj) -> tuple[float, float]:
    shape = list(edge_obj.getShape() or [])
    if shape:
        point = shape[len(shape) // 2]
        return float(point[0]), float(point[1])
    try:
        return tuple(float(v) for v in edge_obj.getFromNode().getCoord())
    except Exception:
        return (0.0, 0.0)


def _lane_runtime_allows_vclass(lane_id: str, vclass: str) -> bool | None:
    try:
        allowed = tuple(traci.lane.getAllowed(lane_id))
        disallowed = tuple(traci.lane.getDisallowed(lane_id))
        max_speed = float(traci.lane.getMaxSpeed(lane_id))
    except Exception:
        return None

    if max_speed <= 0:
        return False
    if allowed:
        return vclass in allowed
    if disallowed:
        return vclass not in disallowed
    return True


def _lane_static_allows_vclass(lane_obj, vclass: str) -> bool:
    try:
        return bool(lane_obj.allows(vclass))
    except Exception:
        return True


def _edge_is_normal(edge_obj) -> bool:
    edge_id = str(edge_obj.getID())
    if not edge_id or edge_id.startswith(":"):
        return False
    try:
        if edge_obj.isSpecial():
            return False
    except Exception:
        pass
    try:
        function_name = str(edge_obj.getFunction() or "").strip().lower()
    except Exception:
        function_name = ""
    return function_name not in SPECIAL_EDGE_FUNCTIONS


def _edge_is_drivable(
    edge_obj,
    *,
    vclass: str,
    require_outgoing: bool = False,
    require_incoming: bool = False,
) -> tuple[bool, str]:
    if edge_obj is None:
        return False, "missing_edge"
    if not _edge_is_normal(edge_obj):
        return False, "special_or_internal"

    try:
        if require_outgoing and len(edge_obj.getOutgoing()) == 0:
            return False, "dead_end_depart"
        if require_incoming and len(edge_obj.getIncoming()) == 0:
            return False, "isolated_arrival"
    except Exception:
        pass

    lane_ids = []
    lane_allowed = False
    runtime_checked = False
    for lane_obj in edge_obj.getLanes():
        lane_id = lane_obj.getID()
        lane_ids.append(lane_id)
        runtime_allowed = _lane_runtime_allows_vclass(lane_id, vclass)
        if runtime_allowed is not None:
            runtime_checked = True
            if runtime_allowed:
                lane_allowed = True
                break
            continue
        if _lane_static_allows_vclass(lane_obj, vclass):
            lane_allowed = True
            break

    if not lane_ids:
        return False, "no_lanes"
    if not lane_allowed:
        return False, "vclass_not_allowed_runtime" if runtime_checked else "vclass_not_allowed"
    return True, "ok"


def _nearest_candidate_edges(
    net,
    edge_id: str,
    *,
    vclass: str,
    require_outgoing: bool,
    require_incoming: bool,
    limit: int = 12,
    exclude_edge_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    exclude = set(exclude_edge_ids or ())
    try:
        reference_edge = net.getEdge(edge_id)
    except Exception:
        reference_edge = None

    ref_x, ref_y = _shape_midpoint(reference_edge) if reference_edge is not None else (0.0, 0.0)
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    if reference_edge is not None:
        candidates.append(
            {
                "edge_id": reference_edge.getID(),
                "distance": 0.0,
                "valid": False,
                "reason": "unchecked",
            }
        )
        seen_ids.add(reference_edge.getID())

    for edge_obj in net.getEdges():
        candidate_id = edge_obj.getID()
        if candidate_id in seen_ids or candidate_id in exclude:
            continue
        cx, cy = _shape_midpoint(edge_obj)
        candidates.append(
            {
                "edge_id": candidate_id,
                "distance": math.hypot(cx - ref_x, cy - ref_y),
                "valid": False,
                "reason": "unchecked",
            }
        )

    candidates.sort(key=lambda item: (item["distance"], item["edge_id"]))
    valid_candidates: list[dict[str, Any]] = []
    for item in candidates:
        candidate_obj = net.getEdge(item["edge_id"])
        valid, reason = _edge_is_drivable(
            candidate_obj,
            vclass=vclass,
            require_outgoing=require_outgoing,
            require_incoming=require_incoming,
        )
        item["valid"] = valid
        item["reason"] = reason
        if valid:
            valid_candidates.append(item)
        if len(valid_candidates) >= limit:
            break
    return valid_candidates


def sanitize_edge_for_depart(
    net,
    original_edge_id: str,
    *,
    vclass: str = DEFAULT_PLANNER_VCLASS,
    limit: int = 12,
    exclude_edge_ids: Sequence[str] = (),
) -> dict[str, Any]:
    candidates = _nearest_candidate_edges(
        net,
        original_edge_id,
        vclass=vclass,
        require_outgoing=True,
        require_incoming=False,
        limit=limit,
        exclude_edge_ids=exclude_edge_ids,
    )
    chosen = candidates[0]["edge_id"] if candidates else None
    reason = "kept_original" if chosen == original_edge_id else "replaced_with_nearest_legal_depart_edge"
    return {
        "ok": bool(chosen),
        "original_edge": original_edge_id,
        "final_edge": chosen,
        "reason": reason if chosen else "no_legal_depart_edge_found",
        "candidates": candidates,
    }


def sanitize_edge_for_arrival(
    net,
    original_edge_id: str,
    *,
    vclass: str = DEFAULT_PLANNER_VCLASS,
    limit: int = 12,
    exclude_edge_ids: Sequence[str] = (),
) -> dict[str, Any]:
    candidates = _nearest_candidate_edges(
        net,
        original_edge_id,
        vclass=vclass,
        require_outgoing=False,
        require_incoming=True,
        limit=limit,
        exclude_edge_ids=exclude_edge_ids,
    )
    chosen = candidates[0]["edge_id"] if candidates else None
    reason = "kept_original" if chosen == original_edge_id else "replaced_with_nearest_legal_arrival_edge"
    return {
        "ok": bool(chosen),
        "original_edge": original_edge_id,
        "final_edge": chosen,
        "reason": reason if chosen else "no_legal_arrival_edge_found",
        "candidates": candidates,
    }


def validate_route_edges(
    net,
    route_edges: Sequence[str],
    *,
    vclass: str = DEFAULT_PLANNER_VCLASS,
) -> dict[str, Any]:
    edges = [str(edge_id) for edge_id in route_edges or [] if str(edge_id).strip()]
    result = {
        "ok": False,
        "reason": "unknown",
        "route_edges": edges,
        "edge_count": len(edges),
        "invalid_edge": None,
        "invalid_pair": None,
        "checks": [],
    }
    if len(edges) < 2:
        result["reason"] = "route_too_short"
        return result

    for index, edge_id in enumerate(edges):
        try:
            edge_obj = net.getEdge(edge_id)
        except Exception:
            result["reason"] = "edge_missing_in_net"
            result["invalid_edge"] = edge_id
            return result
        valid, reason = _edge_is_drivable(
            edge_obj,
            vclass=vclass,
            require_outgoing=index < len(edges) - 1,
            require_incoming=index > 0,
        )
        result["checks"].append({"edge_id": edge_id, "ok": valid, "reason": reason})
        if not valid:
            result["reason"] = reason
            result["invalid_edge"] = edge_id
            return result

    for left, right in zip(edges, edges[1:]):
        left_obj = net.getEdge(left)
        outgoing = {edge_obj.getID() for edge_obj in left_obj.getOutgoing().keys()}
        if right not in outgoing:
            result["reason"] = "disconnected_adjacent_edges"
            result["invalid_pair"] = [left, right]
            return result

    result["ok"] = True
    result["reason"] = "ok"
    return result


def _build_runtime_fallback_route(
    net,
    start_edge: str,
    end_edge: str,
    *,
    vclass: str,
) -> list[str]:
    if not start_edge or not end_edge:
        return []
    if start_edge == end_edge:
        return [start_edge]

    try:
        start_obj = net.getEdge(start_edge)
        end_obj = net.getEdge(end_edge)
    except Exception:
        return []

    frontier = [(float(start_obj.getLength()), start_edge, [start_edge])]
    best_cost = {start_edge: float(start_obj.getLength())}

    while frontier:
        cost, edge_id, path = heapq.heappop(frontier)
        if cost > best_cost.get(edge_id, math.inf):
            continue
        if edge_id == end_edge:
            return path
        edge_obj = net.getEdge(edge_id)
        for next_edge_obj in edge_obj.getOutgoing().keys():
            next_edge_id = next_edge_obj.getID()
            valid, _reason = _edge_is_drivable(
                next_edge_obj,
                vclass=vclass,
                require_outgoing=next_edge_id != end_edge,
                require_incoming=True,
            )
            if not valid:
                continue
            next_cost = cost + float(next_edge_obj.getLength())
            if next_cost >= best_cost.get(next_edge_id, math.inf):
                continue
            best_cost[next_edge_id] = next_cost
            heapq.heappush(frontier, (next_cost, next_edge_id, path + [next_edge_id]))

    return []


def build_safe_vehicle_route(
    net,
    *,
    original_start_edge: str,
    original_end_edge: str,
    original_route_edges: Sequence[str],
    vtype_id: str = DEFAULT_PLANNER_VTYPE,
    vclass: str = DEFAULT_PLANNER_VCLASS,
    candidate_limit: int = 12,
) -> dict[str, Any]:
    original_route = [str(edge_id) for edge_id in original_route_edges or [] if str(edge_id).strip()]
    depart_fix = sanitize_edge_for_depart(
        net,
        original_start_edge or (original_route[0] if original_route else ""),
        vclass=vclass,
        limit=candidate_limit,
    )
    arrival_fix = sanitize_edge_for_arrival(
        net,
        original_end_edge or (original_route[-1] if original_route else ""),
        vclass=vclass,
        limit=candidate_limit,
        exclude_edge_ids=[depart_fix.get("final_edge")] if depart_fix.get("final_edge") else (),
    )
    if not depart_fix["ok"] or not arrival_fix["ok"]:
        reason = depart_fix["reason"] if not depart_fix["ok"] else arrival_fix["reason"]
        return {
            "ok": False,
            "stage": "sumo_route_build",
            "reason": reason,
            "original_start_edge": original_start_edge,
            "final_start_edge": depart_fix.get("final_edge"),
            "original_end_edge": original_end_edge,
            "final_end_edge": arrival_fix.get("final_edge"),
            "original_route_edges": original_route,
            "final_route_edges": [],
            "route_validation": {"ok": False, "reason": reason},
            "repair_actions": [depart_fix["reason"], arrival_fix["reason"]],
            "depart_candidates": depart_fix.get("candidates", []),
            "arrival_candidates": arrival_fix.get("candidates", []),
            "used_sumo_fallback": False,
        }

    final_start = str(depart_fix["final_edge"])
    final_end = str(arrival_fix["final_edge"])
    candidate_route = list(original_route)
    repair_actions = []
    used_sumo_fallback = False

    if final_start != original_start_edge:
        repair_actions.append(
            f"original start edge {original_start_edge} replaced by legal depart edge {final_start}"
        )
    if final_end != original_end_edge:
        repair_actions.append(
            f"original end edge {original_end_edge} replaced by legal arrival edge {final_end}"
        )

    if candidate_route and candidate_route[0] == original_start_edge and candidate_route[-1] == original_end_edge:
        candidate_route[0] = final_start
        candidate_route[-1] = final_end
    else:
        candidate_route = []

    validation = validate_route_edges(net, candidate_route, vclass=vclass)
    if not validation["ok"]:
        used_sumo_fallback = True
        repair_actions.append(f"custom route invalid: {validation['reason']}")
        candidate_route = _build_runtime_fallback_route(
            net,
            final_start,
            final_end,
            vclass=vclass,
        )
        validation = validate_route_edges(net, candidate_route, vclass=vclass)

    if not validation["ok"]:
        start_candidates = [item["edge_id"] for item in depart_fix.get("candidates", [])[:candidate_limit]]
        end_candidates = [item["edge_id"] for item in arrival_fix.get("candidates", [])[:candidate_limit]]
        for start_candidate in start_candidates:
            for end_candidate in end_candidates:
                if start_candidate == end_candidate:
                    continue
                candidate_route = _build_runtime_fallback_route(
                    net,
                    start_candidate,
                    end_candidate,
                    vclass=vclass,
                )
                validation = validate_route_edges(net, candidate_route, vclass=vclass)
                if validation["ok"]:
                    final_start = start_candidate
                    final_end = end_candidate
                    used_sumo_fallback = True
                    repair_actions.append(
                        f"sumo fallback route built from {start_candidate} to {end_candidate}"
                    )
                    break
            if validation["ok"]:
                break

    if not validation["ok"]:
        return {
            "ok": False,
            "stage": "sumo_route_build",
            "reason": validation["reason"],
            "original_start_edge": original_start_edge,
            "final_start_edge": final_start,
            "original_end_edge": original_end_edge,
            "final_end_edge": final_end,
            "original_route_edges": original_route,
            "final_route_edges": list(candidate_route),
            "route_validation": validation,
            "repair_actions": repair_actions,
            "depart_candidates": depart_fix.get("candidates", []),
            "arrival_candidates": arrival_fix.get("candidates", []),
            "used_sumo_fallback": used_sumo_fallback,
        }

    return {
        "ok": True,
        "stage": "sumo_route_build",
        "reason": "ok",
        "original_start_edge": original_start_edge,
        "final_start_edge": final_start,
        "original_end_edge": original_end_edge,
        "final_end_edge": final_end,
        "original_route_edges": original_route,
        "final_route_edges": list(candidate_route),
        "route_validation": validation,
        "repair_actions": repair_actions or ["kept_original_route"],
        "depart_candidates": depart_fix.get("candidates", []),
        "arrival_candidates": arrival_fix.get("candidates", []),
        "used_sumo_fallback": used_sumo_fallback,
    }


class SUMORunner:
    """SUMO simulation runner."""

    def __init__(self, config: Dict):
        self.config = config
        self.sumo_binary = config.get("SUMO_BINARY", "sumo")
        self.sumo_cmd = None
        self.is_running = False
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start_simulation(
        self,
        net_file: str,
        route_file: str = None,
        additional_params: List[str] = None,
    ):
        try:
            cmd = [self.sumo_binary, "-n", net_file]
            if route_file:
                cmd.extend(["-r", route_file])
            if additional_params:
                cmd.extend(additional_params)
            cmd.extend(["--start", "--quit-on-end"])
            self.sumo_cmd = cmd
            self.logger.info("Starting SUMO with command: %s", " ".join(cmd))
            traci.start(cmd)
            self.is_running = True
            self.logger.info("SUMO simulation started successfully")
        except Exception as exc:
            self.logger.error("Failed to start SUMO simulation: %s", exc)
            raise

    def stop_simulation(self):
        try:
            if self.is_running:
                traci.close()
                self.is_running = False
                self.logger.info("SUMO simulation stopped successfully")
        except Exception as exc:
            self.logger.error("Error stopping SUMO simulation: %s", exc)

    def get_edge_info(self, edge_id: str) -> Dict:
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running")
        try:
            return {
                "id": edge_id,
                "length": traci.edge.getLength(edge_id),
                "max_speed": traci.edge.getMaxSpeed(edge_id),
                "occupancy": traci.edge.getLastStepOccupancy(edge_id),
                "waiting_time": traci.edge.getWaitingTime(edge_id),
                "mean_speed": traci.edge.getLastStepMeanSpeed(edge_id),
            }
        except Exception as exc:
            self.logger.error("Error getting edge info for %s: %s", edge_id, exc)
            return {}

    def get_all_edge_ids(self) -> List[str]:
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running")
        try:
            return traci.edge.getIDList()
        except Exception as exc:
            self.logger.error("Error getting edge IDs: %s", exc)
            return []

    def get_traffic_light_state(self, tl_id: str) -> str:
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running")
        try:
            return traci.trafficlight.getRedYellowGreenState(tl_id)
        except Exception as exc:
            self.logger.error("Error getting traffic light state for %s: %s", tl_id, exc)
            return ""

    def set_traffic_light_state(self, tl_id: str, state: str):
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running")
        try:
            traci.trafficlight.setRedYellowGreenState(tl_id, state)
        except Exception as exc:
            self.logger.error("Error setting traffic light state for %s: %s", tl_id, exc)

    def step(self, step_amount: float = 1.0):
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running")
        try:
            traci.simulationStep(step_amount)
        except Exception as exc:
            self.logger.error("Error during simulation step: %s", exc)

    def get_simulation_step(self) -> int:
        if not self.is_running:
            return 0
        try:
            return traci.simulation.getCurrentStep()
        except Exception as exc:
            self.logger.error("Error getting simulation step: %s", exc)
            return 0

    def get_vehicle_count(self) -> int:
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running")
        try:
            return traci.simulation.getMinExpectedNumber()
        except Exception as exc:
            self.logger.error("Error getting vehicle count: %s", exc)
            return 0

    def get_vehicle_info(self, veh_id: str) -> Dict:
        if not self.is_running:
            raise RuntimeError("SUMO simulation is not running")
        try:
            return {
                "id": veh_id,
                "position": traci.vehicle.getPosition(veh_id),
                "speed": traci.vehicle.getSpeed(veh_id),
                "route": traci.vehicle.getRoute(veh_id),
                "lane": traci.vehicle.getLaneID(veh_id),
                "acceleration": traci.vehicle.getAcceleration(veh_id),
            }
        except Exception as exc:
            self.logger.error("Error getting vehicle info for %s: %s", veh_id, exc)
            return {}


def load_sumo_network(net_path: str) -> Tuple:
    try:
        net = sumolib.net.readNet(net_path)
        edge_coords = {}
        for edge in net.getEdges():
            shape = edge.getShape()
            if shape:
                edge_coords[edge.getID()] = [(x, y) for x, y in shape]
        return net, edge_coords
    except Exception as exc:
        LOGGER.error("Error loading SUMO network %s: %s", net_path, exc)
        raise


def parse_net_file_for_edges(net_file_path: str) -> List[Dict]:
    edges_info = []
    try:
        tree = ET.parse(net_file_path)
        root = tree.getroot()
        for edge_elem in root.findall(".//edge"):
            edges_info.append(
                {
                    "id": edge_elem.get("id"),
                    "from": edge_elem.get("from"),
                    "to": edge_elem.get("to"),
                    "priority": edge_elem.get("priority"),
                    "speed": edge_elem.get("speed"),
                    "length": edge_elem.get("length"),
                }
            )
        return edges_info
    except Exception as exc:
        LOGGER.error("Error parsing net file %s: %s", net_file_path, exc)
        return []


def get_total_edges_from_net(net_file_path: str) -> int:
    try:
        return len(parse_net_file_for_edges(net_file_path))
    except Exception as exc:
        LOGGER.error("Error counting edges in %s: %s", net_file_path, exc)
        return 0
