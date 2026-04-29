#!/usr/bin/env python3
"""
build_urban_net.py
Generate a realistic Zhengzhou Jinshui-style urban road network for SUMO.

Layout (6 columns × 5 rows, spacing 600–900 m):
  Columns (N-S roads): C0=经一路  C1=经三路  C2=经六路  C3=经八路  C4=花园路  C5=未来路
  Rows    (E-W roads): R0=农业路  R1=红专路  R2=政七街  R3=黄河路  R4=纬五路

Road hierarchy:
  Major arterial   (农业路/黄河路/经六路/花园路) : 4 lanes, 60 km/h, priority 9
  Secondary artery (红专路/经三路/经八路)        : 3 lanes, 50 km/h, priority 7
  Minor road       (政七街/经一路/未来路/纬五路) : 2 lanes, 40 km/h, priority 5
"""

import os, subprocess, sys

# ─── Grid parameters ──────────────────────────────────────────
COL_NAMES = ["C0_Jing1",  "C1_Jing3",  "C2_Jing6",
             "C3_Jing8",  "C4_Huayuan","C5_Weilai"]
ROW_NAMES = ["R0_Nongye", "R1_Hongzhuan", "R2_Zheng7",
             "R3_Huanghe","R4_Wei5"]

# X positions (metres) — wider spacing on major arterials
COL_X = [0, 620, 1320, 1940, 2680, 3300]
# Y positions (metres, 0 = south, increases northward)
ROW_Y = [0, 680, 1260, 1940, 2540]

# Road type: 'A'=major arterial, 'S'=secondary, 'M'=minor
COL_TYPE = ['M', 'S', 'A', 'S', 'A', 'M']
ROW_TYPE = ['A', 'S', 'M', 'A', 'M']

TYPE_SPEED  = {'A': 16.67, 'S': 13.89, 'M': 11.11}   # m/s (60/50/40 km/h)
TYPE_LANES  = {'A': 4,     'S': 3,     'M': 2}
TYPE_PRIO   = {'A': 9,     'S': 7,     'M': 5}

def node_id(r, c):
    return f"N{r}{c}"

def edge_speed(rt):   return TYPE_SPEED[rt]
def edge_lanes(rt):   return TYPE_LANES[rt]
def edge_prio(rt):    return TYPE_PRIO[rt]

# ─── Write nodes.xml ──────────────────────────────────────────
def write_nodes(path):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<nodes>']
    for r in range(len(ROW_NAMES)):
        for c in range(len(COL_NAMES)):
            nid  = node_id(r, c)
            x, y = COL_X[c], ROW_Y[r]
            # Use traffic_light for major intersections, right_before_left otherwise
            rt = ROW_TYPE[r]; ct = COL_TYPE[c]
            if rt == 'A' or ct == 'A':
                ntype = 'traffic_light'
            elif rt == 'S' or ct == 'S':
                ntype = 'traffic_light'
            else:
                ntype = 'priority'
            lines.append(f'  <node id="{nid}" x="{x}" y="{y}" type="{ntype}"/>')
    lines.append('</nodes>')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✅ Wrote {path}")

# ─── Write edges.xml ──────────────────────────────────────────
def edge_type(r_type, c_type):
    # Intersection edge type = max priority of the two roads
    pA = TYPE_PRIO[r_type]; pB = TYPE_PRIO[c_type]
    for t in ['A','S','M']:
        if TYPE_PRIO[t] == max(pA, pB):
            return t
    return 'M'

def write_edges(path):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<edges>']
    eid = 0

    # Horizontal edges (E-W along each row)
    for r in range(len(ROW_NAMES)):
        rt = ROW_TYPE[r]
        for c in range(len(COL_NAMES)-1):
            et = edge_type(rt, COL_TYPE[c])
            fr = node_id(r, c)
            to = node_id(r, c+1)
            sp = edge_speed(et); ln = edge_lanes(et); pr = edge_prio(et)
            # Both directions
            lines.append(f'  <edge id="R{r}C{c}_E" from="{fr}" to="{to}" '
                         f'priority="{pr}" numLanes="{ln}" speed="{sp:.2f}"/>')
            lines.append(f'  <edge id="R{r}C{c}_W" from="{to}" to="{fr}" '
                         f'priority="{pr}" numLanes="{ln}" speed="{sp:.2f}"/>')
            eid += 2

    # Vertical edges (N-S along each column)
    for c in range(len(COL_NAMES)):
        ct = COL_TYPE[c]
        for r in range(len(ROW_NAMES)-1):
            et = edge_type(ROW_TYPE[r], ct)
            fr = node_id(r, c)
            to = node_id(r+1, c)
            sp = edge_speed(et); ln = edge_lanes(et); pr = edge_prio(et)
            lines.append(f'  <edge id="C{c}R{r}_N" from="{fr}" to="{to}" '
                         f'priority="{pr}" numLanes="{ln}" speed="{sp:.2f}"/>')
            lines.append(f'  <edge id="C{c}R{r}_S" from="{to}" to="{fr}" '
                         f'priority="{pr}" numLanes="{ln}" speed="{sp:.2f}"/>')
            eid += 2

    lines.append('</edges>')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✅ Wrote {path}  ({eid} edges)")
    return eid

# ─── Write routes.rou.xml ─────────────────────────────────────
def write_routes(path):
    # Named after real Zhengzhou roads for readability
    routes = [
        # E-W major arterials (农业路/黄河路)
        ("Nongye_EW",   ["R0C0_E","R0C1_E","R0C2_E","R0C3_E","R0C4_E"]),
        ("Nongye_WE",   ["R0C4_W","R0C3_W","R0C2_W","R0C1_W","R0C0_W"]),
        ("Huanghe_EW",  ["R3C0_E","R3C1_E","R3C2_E","R3C3_E","R3C4_E"]),
        ("Huanghe_WE",  ["R3C4_W","R3C3_W","R3C2_W","R3C1_W","R3C0_W"]),
        # E-W secondary (红专路)
        ("Hongzhuan_EW",["R1C0_E","R1C1_E","R1C2_E","R1C3_E","R1C4_E"]),
        ("Hongzhuan_WE",["R1C4_W","R1C3_W","R1C2_W","R1C1_W","R1C0_W"]),
        # N-S major arterials (花园路/经六路)
        ("Huayuan_NS",  ["C4R0_N","C4R1_N","C4R2_N","C4R3_N"]),
        ("Huayuan_SN",  ["C4R3_S","C4R2_S","C4R1_S","C4R0_S"]),
        ("Jing6_NS",    ["C2R0_N","C2R1_N","C2R2_N","C2R3_N"]),
        ("Jing6_SN",    ["C2R3_S","C2R2_S","C2R1_S","C2R0_S"]),
        # Diagonal / cross-town
        ("CrossNW_SE",  ["R0C0_E","R0C1_E","C1R0_N","C1R1_N","R2C1_E","R2C2_E","C2R2_N","C2R3_N"]),
        ("CrossSW_NE",  ["R4C0_E","R4C1_E","C1R3_N","C1R4_N","R4C1_E","R4C2_E","C2R3_N"]),
        # Short hops
        ("Short_R2",    ["R2C2_E","R2C3_E"]),
        ("Short_C3",    ["C3R1_N","C3R2_N"]),
    ]

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<routes>',
             '',
             '  <!-- Vehicle types -->',
             '  <vType id="car"   accel="2.6" decel="4.5" sigma="0.5" length="4.5" '
             'minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>',
             '  <vType id="bus"   accel="1.2" decel="3.5" sigma="0.3" length="12.0" '
             'minGap="3.0" maxSpeed="13.89" guiShape="bus"/>',
             '  <vType id="truck" accel="1.0" decel="3.0" sigma="0.3" length="9.0" '
             'minGap="3.5" maxSpeed="11.11" guiShape="truck"/>',
             '']

    # Route definitions
    for rname, edges in routes:
        elist = ' '.join(edges)
        lines.append(f'  <route id="{rname}" edges="{elist}"/>')
    lines.append('')

    # Flows — major arterials high frequency
    flows = [
        # Major arterial car flows
        ("f_Nongye_EW",  "Nongye_EW",   "car",   0,   3600, 15),
        ("f_Nongye_WE",  "Nongye_WE",   "car",   5,   3600, 15),
        ("f_Huanghe_EW", "Huanghe_EW",  "car",   0,   3600, 18),
        ("f_Huanghe_WE", "Huanghe_WE",  "car",   8,   3600, 18),
        ("f_Huayuan_NS", "Huayuan_NS",  "car",   0,   3600, 20),
        ("f_Huayuan_SN", "Huayuan_SN",  "car",   10,  3600, 20),
        ("f_Jing6_NS",   "Jing6_NS",    "car",   0,   3600, 22),
        ("f_Jing6_SN",   "Jing6_SN",    "car",   12,  3600, 22),
        # Secondary
        ("f_Hongz_EW",   "Hongzhuan_EW","car",   0,   3600, 30),
        ("f_Hongz_WE",   "Hongzhuan_WE","car",   15,  3600, 30),
        # Cross-town
        ("f_cross1",     "CrossNW_SE",  "car",   0,   3600, 45),
        # Bus on major
        ("f_bus_N",      "Nongye_EW",   "bus",   60,  3600, 180),
        ("f_bus_H",      "Huanghe_WE",  "bus",   90,  3600, 180),
        # Truck
        ("f_truck",      "Huanghe_EW",  "truck", 0,   3600, 300),
    ]

    for fid, route, vtype, beg, end, period in flows:
        lines.append(f'  <flow id="{fid}" route="{route}" type="{vtype}" '
                     f'begin="{beg}" end="{end}" period="{period}" '
                     f'departLane="best" departSpeed="max"/>')

    lines.append('</routes>')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✅ Wrote {path}")

# ─── Write SUMO config ────────────────────────────────────────
def write_sumocfg(path, net_rel, rou_rel):
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file    value="{net_rel}"/>
    <route-files value="{rou_rel}"/>
  </input>
  <time>
    <begin       value="0"/>
    <end         value="3600"/>
    <step-length value="0.1"/>
  </time>
  <processing>
    <ignore-route-errors value="true"/>
  </processing>
  <report>
    <no-warnings          value="true"/>
    <duration-log.disable value="true"/>
  </report>
</configuration>
"""
    with open(path, 'w') as f:
        f.write(xml)
    print(f"  ✅ Wrote {path}")

# ─── Print edge list for dataset generator ────────────────────
def print_edges():
    edges = []
    for r in range(len(ROW_NAMES)):
        for c in range(len(COL_NAMES)-1):
            edges.append(f"R{r}C{c}_E")
            edges.append(f"R{r}C{c}_W")
    for c in range(len(COL_NAMES)):
        for r in range(len(ROW_NAMES)-1):
            edges.append(f"C{c}R{r}_N")
            edges.append(f"C{c}R{r}_S")
    print(f"\n📋 Total edges: {len(edges)}")
    print("EDGES = " + repr(edges))
    return edges

# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT = "/root/autodl-tmp/SUMO"
    NET_DIR = f"{OUT}/net"
    os.makedirs(NET_DIR, exist_ok=True)

    node_path = f"{NET_DIR}/urban_nodes.xml"
    edge_path  = f"{NET_DIR}/urban_edges.xml"
    net_path   = f"{NET_DIR}/my_net.net.xml"
    rou_path   = f"{OUT}/routes.rou.xml"
    cfg_path   = f"{OUT}/config/my_config.sumocfg"

    print("\n🏗️  Building Zhengzhou-style urban SUMO network...\n")

    write_nodes(node_path)
    n_edges = write_edges(edge_path)
    write_routes(rou_path)
    write_sumocfg(cfg_path, "../net/my_net.net.xml", "../routes.rou.xml")

    # Build .net.xml with netconvert
    cmd = [
        "netconvert",
        "--node-files", node_path,
        "--edge-files", edge_path,
        "--output-file", net_path,
        "--tls.guess", "true",
        "--tls.default-type", "static",
        "--junctions.join", "true",
        "--geometry.remove", "true",
        "--roundabouts.guess", "true",
        "--no-warnings",
    ]
    print("\n🔧 Running netconvert...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ Network built: {net_path}")
    else:
        print("  ❌ netconvert failed:")
        print(result.stderr[-800:])
        sys.exit(1)

    # Print edge list
    edges = print_edges()

    print(f"\n✅ Done!  {n_edges} directed edges, {len(ROW_NAMES)*len(COL_NAMES)} nodes")
    print(f"   Net:    {net_path}")
    print(f"   Routes: {rou_path}")
    print(f"   Config: {cfg_path}")
    print("\n🧪 Validate with:")
    print(f"   sumo -c {cfg_path} --no-step-log 2>&1 | head -5")
