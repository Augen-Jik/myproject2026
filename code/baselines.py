"""
baselines.py — 对比基线算法集合（毕设实验 baseline 部分）

包含算法：
  1. UniformDijkstra    — 等权 Dijkstra（完全忽略拥堵，最差基线）
  2. RuleDijkstra       — 规则解析权重 + Dijkstra（无 LLM 的传统方法）
  3. LLMDijkstra        — LLM 原始权重 + Dijkstra（当前系统，未引入 GAT）
  4. GATDijkstra        — LLM-GAT 级联 + Dijkstra（本文提出方法）
  5. BellmanFord        — Bellman-Ford（经典单源最短路，可处理负权）
  6. ACO                — 蚁群算法（元启发式，生物启发算法代表）

所有算法统一接口：
    plan(start_node, end_node, weight_dict) → (path, cost, time_ms)
"""

import heapq
import time
import random
import re
import math
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    import sumolib
except Exception:  # pragma: no cover
    sumolib = None

# ════════════════════════════════════════════════════════════
#  路网图结构（与 gat_smoother.py 保持一致）
# ════════════════════════════════════════════════════════════

N_ROWS   = 5
N_COLS   = 6
ROW_TYPE = ['A', 'S', 'M', 'A', 'M']
COL_TYPE = ['M', 'S', 'A', 'S', 'A', 'M']
SUMO_NET_PATH = "/root/autodl-tmp/SUMO/net/my_net.net.xml"

LIVE_EDGE_SET = set()
if sumolib is not None and os.path.exists(SUMO_NET_PATH):
    try:
        _live_net = sumolib.net.readNet(SUMO_NET_PATH)
        LIVE_EDGE_SET = {edge.getID() for edge in _live_net.getEdges() if edge is not None}
    except Exception:
        LIVE_EDGE_SET = set()

# 交叉口节点：(row, col)，共 5×6=30 个
def all_nodes():
    return [(r, c) for r in range(N_ROWS) for c in range(N_COLS)]

# 有向边列表：(start_node, end_node, edge_id)
def all_directed_edges():
    edges = []
    # 横向边
    for r in range(N_ROWS):
        for c in range(N_COLS - 1):
            edges.append(((r, c),   (r, c+1), f"R{r}C{c}_E"))
            edges.append(((r, c+1), (r, c),   f"R{r}C{c}_W"))
    # 纵向边
    for c in range(N_COLS):
        for r in range(N_ROWS - 1):
            edges.append(((r,   c), (r+1, c), f"C{c}R{r}_N"))
            edges.append(((r+1, c), (r,   c), f"C{c}R{r}_S"))
    if LIVE_EDGE_SET:
        edges = [item for item in edges if item[2] in LIVE_EDGE_SET]
    return edges


def build_graph(weight_dict: Dict[str, float],
                default_w: float = 2.0) -> Dict:
    """
    将权重字典转换为邻接表
    graph[node] = [(cost, neighbor_node, edge_id), ...]
    """
    graph = defaultdict(list)
    for src, dst, eid in all_directed_edges():
        w = weight_dict.get(eid, default_w)
        graph[src].append((w, dst, eid))
    return graph


def reconstruct_path(prev: dict, start, end) -> List:
    """从前驱字典回溯完整路径（节点序列）"""
    path = []
    cur  = end
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path if path[0] == start else []


def heuristic(node, goal) -> float:
    """A* 曼哈顿距离启发函数（网格图）"""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


# ════════════════════════════════════════════════════════════
#  1. Uniform Dijkstra（等权，忽略拥堵）
# ════════════════════════════════════════════════════════════

class UniformDijkstra:
    """
    等权 Dijkstra：所有路段权重视为 1.0
    代表「完全不考虑交通状况」的朴素最短路算法
    等价于 BFS（无权图最短路）
    """
    name = "Uniform-Dijkstra"

    def plan(self, start, end, weight_dict=None) -> Tuple[List, float, float]:
        t0    = time.perf_counter()
        graph = build_graph({}, default_w=1.0)   # 忽略权重，全为 1
        dist  = {n: math.inf for n in all_nodes()}
        prev  = {}
        dist[start] = 0.0
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == end:
                break
            for w, v, eid in graph[u]:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v]  = u
                    heapq.heappush(pq, (nd, v))

        path    = reconstruct_path(prev, start, end)
        elapsed = (time.perf_counter() - t0) * 1000
        return path, dist.get(end, math.inf), elapsed


# ════════════════════════════════════════════════════════════
#  2. Rule-based Dijkstra（关键词规则解析权重）
# ════════════════════════════════════════════════════════════

ROW_NAMES = ["农业路", "红专路", "政七街", "黄河路", "纬五路"]
COL_NAMES = ["经一路", "经三路", "经六路", "经八路", "花园路", "未来路"]
ROW_MAP   = {n: i for i, n in enumerate(ROW_NAMES)}
COL_MAP   = {n: i for i, n in enumerate(COL_NAMES)}
DIR_MAP   = {"向东": "E", "向西": "W", "向北": "N", "向南": "S"}

RULE_KW = [
    (["封闭", "管制", "全封"],              9.5),
    (["极度拥堵", "连环追尾", "严重事故"],  8.5),
    (["严重拥堵", "严重"],                  7.5),
    (["中度拥堵", "中度"],                  5.0),
    (["轻微拥堵", "轻微"],                  3.0),
    (["畅通", "顺畅", "畅通无阻"],          0.8),
    (["正常", "正常通行"],                  2.0),
]

def rule_parse(constraint: str) -> Dict[str, float]:
    """
    纯规则方法：从自然语言描述中提取路段权重
    不依赖 LLM，代表「传统 NLP 规则」基线
    """
    weights = {eid: 2.0 for _, _, eid in all_directed_edges()}

    sentences = re.split(r'[；。\n]', constraint)
    for sent in sentences:
        # 识别道路名称
        row_match = next((n for n in ROW_NAMES if n in sent), None)
        col_match = next((n for n in COL_NAMES if n in sent), None)
        dir_match = next((d for d in DIR_MAP   if d in sent), None)

        # 识别拥堵等级
        w = 2.0
        for kws, wv in RULE_KW:
            if any(kw in sent for kw in kws):
                w = wv
                break

        if dir_match is None:
            continue

        if row_match and dir_match in ("向东", "向西"):
            r = ROW_MAP[row_match]
            d = DIR_MAP[dir_match]
            for c in range(N_COLS - 1):
                weights[f"R{r}C{c}_{d}"] = w
        elif col_match and dir_match in ("向北", "向南"):
            c = COL_MAP[col_match]
            d = DIR_MAP[dir_match]
            for r in range(N_ROWS - 1):
                weights[f"C{c}R{r}_{d}"] = w

    return weights


class RuleDijkstra:
    """
    规则解析 + Dijkstra：用 NLP 关键词规则生成权重，再做最短路
    代表「不引入 LLM 的传统方法」
    """
    name = "Rule-Dijkstra"

    def plan(self, start, end, weight_dict=None,
             constraint: str = "") -> Tuple[List, float, float]:
        t0      = time.perf_counter()
        wd      = rule_parse(constraint) if constraint else weight_dict or {}
        graph   = build_graph(wd)
        dist    = {n: math.inf for n in all_nodes()}
        prev    = {}
        dist[start] = 0.0
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == end:
                break
            for w, v, eid in graph[u]:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v]  = u
                    heapq.heappush(pq, (nd, v))

        path    = reconstruct_path(prev, start, end)
        elapsed = (time.perf_counter() - t0) * 1000
        return path, dist.get(end, math.inf), elapsed


# ════════════════════════════════════════════════════════════
#  3. LLM Dijkstra（LLM 原始权重 + A*）
# ════════════════════════════════════════════════════════════

class LLMDijkstra:
    """
    LLM 原始权重 + Dijkstra：当前系统（未引入 GAT 的 baseline）
    缺失边使用默认值 2.0
    """
    name = "LLM-Dijkstra"

    def plan(self, start, end, weight_dict: Dict[str, float],
             default_w: float = 2.0) -> Tuple[List, float, float]:
        t0    = time.perf_counter()
        graph = build_graph(weight_dict, default_w)
        dist  = {n: math.inf for n in all_nodes()}
        prev  = {}
        dist[start] = 0.0
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == end:
                break
            for w, v, eid in graph[u]:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v]  = u
                    heapq.heappush(pq, (nd, v))

        path    = reconstruct_path(prev, start, end)
        elapsed = (time.perf_counter() - t0) * 1000
        return path, dist.get(end, math.inf), elapsed


# ════════════════════════════════════════════════════════════
#  4. GAT Dijkstra（本文方法：LLM-GAT 级联 + Dijkstra）
# ════════════════════════════════════════════════════════════

class GATDijkstra:
    """
    ★ 本文提出方法：LLM 权重 → GAT 精炼补全 → Dijkstra
    对比 LLMDijkstra 可以看出 GAT 层的增益
    """
    name = "LLM-GAT-Dijkstra"

    def __init__(self, model_path: str = "/root/autodl-tmp/gat_model.pt",
                 alpha: float = 0.65, device: str = "cpu"):
        self.model_path = model_path
        self.alpha      = alpha
        self.device     = device
        self._inner     = LLMDijkstra()

    def plan(self, start, end,
             weight_dict: Dict[str, float]) -> Tuple[List, float, float]:
        t0 = time.perf_counter()

        # 引入 gat_smoother（延迟导入，避免强依赖 PyTorch 的纯 baseline 跑不起来）
        try:
            from gat_smoother import smooth_weights
            refined_wd, _ = smooth_weights(weight_dict, self.model_path,
                                           self.alpha, self.device)
        except ImportError:
            # 降级：直接用 LLM 权重
            refined_wd = weight_dict

        graph = build_graph(refined_wd)
        dist  = {n: math.inf for n in all_nodes()}
        prev  = {}
        dist[start] = 0.0
        pq = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == end:
                break
            for w, v, eid in graph[u]:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v]  = u
                    heapq.heappush(pq, (nd, v))

        path    = reconstruct_path(prev, start, end)
        elapsed = (time.perf_counter() - t0) * 1000
        return path, dist.get(end, math.inf), elapsed


# ════════════════════════════════════════════════════════════
#  5. Bellman-Ford
# ════════════════════════════════════════════════════════════

class BellmanFord:
    """
    Bellman-Ford 算法：O(V·E) 时间复杂度
    优势：可处理负权边（本场景无负权，但作为经典算法对比）
    作为「经典最短路，但更慢」的参照基线
    """
    name = "Bellman-Ford"

    def plan(self, start, end,
             weight_dict: Dict[str, float]) -> Tuple[List, float, float]:
        t0    = time.perf_counter()
        nodes = all_nodes()
        edges = all_directed_edges()

        dist  = {n: math.inf for n in nodes}
        prev  = {}
        dist[start] = 0.0

        # V-1 次松弛
        for _ in range(len(nodes) - 1):
            updated = False
            for src, dst, eid in edges:
                w  = weight_dict.get(eid, 2.0)
                nd = dist[src] + w
                if dist[src] < math.inf and nd < dist[dst]:
                    dist[dst] = nd
                    prev[dst]  = src
                    updated    = True
            if not updated:
                break   # 提前收敛

        path    = reconstruct_path(prev, start, end)
        elapsed = (time.perf_counter() - t0) * 1000
        return path, dist.get(end, math.inf), elapsed


# ════════════════════════════════════════════════════════════
#  6. 蚁群算法 ACO（Ant Colony Optimization）
# ════════════════════════════════════════════════════════════

class ACO:
    """
    蚁群算法：经典元启发式算法，对比神经网络方法
    参数：
        n_ants    每轮蚂蚁数量
        n_iter    迭代轮次
        alpha     信息素重要性系数
        beta      启发信息（1/权重）重要性系数
        rho       信息素挥发率
        Q         信息素更新强度
    """
    name = "ACO"

    def __init__(self, n_ants: int = 20, n_iter: int = 50,
                 alpha: float = 1.0, beta: float = 2.0,
                 rho: float = 0.3, Q: float = 10.0):
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha  = alpha
        self.beta   = beta
        self.rho    = rho
        self.Q      = Q

    def plan(self, start, end,
             weight_dict: Dict[str, float]) -> Tuple[List, float, float]:
        t0    = time.perf_counter()
        graph = build_graph(weight_dict)

        # 初始化信息素（均匀分布）
        pheromone: Dict[Tuple, float] = defaultdict(lambda: 1.0)
        best_path, best_cost = [], math.inf

        for it in range(self.n_iter):
            all_paths  = []
            all_costs  = []

            for _ in range(self.n_ants):
                path, cost = self._walk(start, end, graph, pheromone)
                all_paths.append(path)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            # 信息素挥发
            for key in pheromone:
                pheromone[key] *= (1 - self.rho)

            # 信息素沉积（只有找到目标的蚂蚁才沉积）
            for path, cost in zip(all_paths, all_costs):
                if cost < math.inf and len(path) >= 2:
                    deposit = self.Q / cost
                    for i in range(len(path) - 1):
                        pheromone[(path[i], path[i+1])] += deposit

        elapsed = (time.perf_counter() - t0) * 1000
        return best_path, best_cost, elapsed

    def _walk(self, start, end, graph, pheromone) -> Tuple[List, float]:
        """单只蚂蚁从 start 走到 end 或超过步数限制"""
        MAX_STEPS = 60    # 防止死循环（30节点最长路≤29步）
        path      = [start]
        cost      = 0.0
        visited   = {start}
        cur       = start

        for _ in range(MAX_STEPS):
            if cur == end:
                return path, cost

            neighbors = graph.get(cur, [])
            # 过滤已访问节点（避免回路）
            candidates = [(w, v, eid) for w, v, eid in neighbors if v not in visited]
            if not candidates:
                # 死路：尝试回溯（允许回头但代价 × 2）
                candidates = [(w * 2, v, eid) for w, v, eid in neighbors]
                if not candidates:
                    return path, math.inf

            # 计算转移概率 τ^α × η^β（η = 1/w）
            scores = []
            for w, v, eid in candidates:
                tau  = pheromone[(cur, v)] ** self.alpha
                eta  = (1.0 / max(w, 0.1)) ** self.beta
                scores.append(tau * eta)

            total = sum(scores)
            if total == 0:
                scores = [1.0] * len(scores)
                total  = len(scores)

            probs = [s / total for s in scores]

            # 轮盘选择
            r   = random.random()
            acc = 0.0
            chosen_idx = len(candidates) - 1
            for i, p in enumerate(probs):
                acc += p
                if r <= acc:
                    chosen_idx = i
                    break

            w, nxt, _ = candidates[chosen_idx]
            path.append(nxt)
            cost += w
            visited.add(nxt)
            cur = nxt

        return path, math.inf   # 超步数未到达


# ════════════════════════════════════════════════════════════
#  统一接口工厂
# ════════════════════════════════════════════════════════════

def get_all_baselines(gat_model_path: str = "/root/autodl-tmp/gat_model.pt"):
    """返回所有算法实例列表，按性能预期从弱到强排列"""
    return [
        UniformDijkstra(),
        RuleDijkstra(),
        BellmanFord(),
        ACO(n_ants=15, n_iter=30),   # 轻量版，评测时间可控
        LLMDijkstra(),
        GATDijkstra(model_path=gat_model_path),
    ]


if __name__ == "__main__":
    print("=" * 50)
    print("  Baseline 快速验证")
    print("=" * 50)

    # 简单测试场景
    wd = {f"R3C{c}_E": 8.0 for c in range(5)}   # 黄河路向东全程拥堵
    wd.update({f"R3C{c}_W": 2.0 for c in range(5)})

    start = (3, 0)  # R3C0（黄河路×经一路）
    end   = (3, 5)  # R3C5（黄河路×未来路）

    for algo in get_all_baselines():
        if isinstance(algo, RuleDijkstra):
            path, cost, ms = algo.plan(start, end,
                constraint="黄河路向东严重拥堵；黄河路向西畅通。请为全部98条路段生成权重。")
        else:
            path, cost, ms = algo.plan(start, end, wd)

        print(f"  {algo.name:20s}  cost={cost:6.1f}  t={ms:.2f}ms  "
              f"hops={len(path)-1 if path else 0}")

    print("=" * 50)
