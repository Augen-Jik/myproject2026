"""96-edge aligned GAT smoother for sparse-anchor topology completion."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from roadnet_meta import TYPE2ID, load_roadnet_meta

DEFAULT_ALWAYS_GAT_PATH = "/root/autodl-tmp/gat_model_v2_always.pt"
DEFAULT_GATED_GAT_PATH = "/root/autodl-tmp/gat_model_v2_gated.pt"

ROADNET = load_roadnet_meta()
EDGES = list(ROADNET.edge_ids)
EDGE_META = ROADNET.edge_meta
EDGE_IDX = ROADNET.edge_index
N_EDGES = len(EDGES)
FEATURE_DIM = 6  # weight + parsed_mask + confidence + road_type_onehot(3)


def build_line_graph_adj() -> torch.Tensor:
    adj = torch.zeros(N_EDGES, N_EDGES)
    for src, neighbors in ROADNET.line_graph_neighbors.items():
        src_idx = EDGE_IDX[src]
        for dst in neighbors:
            adj[src_idx, EDGE_IDX[dst]] = 1.0
    return adj


def build_edge_type_onehot() -> torch.Tensor:
    feats = torch.zeros(N_EDGES, 3)
    for idx, eid in enumerate(EDGES):
        feats[idx, TYPE2ID[EDGE_META[eid]["rtype"]]] = 1.0
    return feats


def build_feature_tensor(
    llm_weight_dict: Dict[str, float],
    *,
    edge_confidence: Optional[Dict[str, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_vec = torch.zeros(N_EDGES)
    parsed_mask = torch.zeros(N_EDGES)
    conf_vec = torch.zeros(N_EDGES)
    for eid, weight in llm_weight_dict.items():
        if eid not in EDGE_IDX:
            continue
        idx = EDGE_IDX[eid]
        raw_vec[idx] = float(weight)
        parsed_mask[idx] = 1.0
        conf_vec[idx] = float(edge_confidence.get(eid, 0.55) if edge_confidence else 0.55)
    type_feats = build_edge_type_onehot()
    features = torch.cat(
        [
            (raw_vec / 10.0).unsqueeze(-1),
            parsed_mask.unsqueeze(-1),
            conf_vec.unsqueeze(-1),
            type_feats,
        ],
        dim=-1,
    )
    return features, raw_vec, parsed_mask.bool()


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        heads = self.heads
        dim = self.out_dim
        proj = self.proj(x).view(n_nodes, heads, dim)
        left = proj.unsqueeze(1).expand(n_nodes, n_nodes, heads, dim)
        right = proj.unsqueeze(0).expand(n_nodes, n_nodes, heads, dim)
        pair = torch.cat([left, right], dim=-1)
        score = self.leaky_relu((pair * self.attn).sum(dim=-1))
        mask = (adj == 0).unsqueeze(-1).expand_as(score)
        score = score.masked_fill(mask, float("-inf"))
        alpha = F.softmax(score, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)
        out = torch.einsum("ijk,jkd->ikd", alpha, proj)
        if self.concat:
            return F.elu(out.reshape(n_nodes, heads * dim))
        return out.mean(dim=1)


class RoadGAT(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 24, heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.heads = heads
        self.dropout = dropout
        self.gat1 = GATLayer(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATLayer(hidden * heads, hidden, heads=heads, dropout=dropout, concat=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, 24),
            nn.ELU(),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )
        self.register_buffer("adj", build_line_graph_adj())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        outputs = []
        for sample in features:
            hidden = self.gat1(sample, self.adj)
            hidden = self.gat2(hidden, self.adj)
            outputs.append(self.head(hidden).squeeze(-1) * 10.0)
        stacked = torch.stack(outputs)
        return stacked.squeeze(0) if squeeze else stacked


def gate_alpha(
    *,
    parse_confidence: float,
    protected_edges: int = 0,
    direction_conflict: bool = False,
    scene_type: str | None = None,
) -> float:
    if parse_confidence >= 0.82:
        alpha = 0.86
    elif parse_confidence >= 0.68:
        alpha = 0.74
    elif parse_confidence >= 0.50:
        alpha = 0.62
    else:
        alpha = 0.48

    if direction_conflict:
        alpha += 0.08
    if protected_edges >= 4:
        alpha += 0.06
    if scene_type in {"compound_disaster", "temporal_switch", "propagation_range"}:
        alpha -= 0.05

    return max(0.35, min(0.92, alpha))


def save_gat_checkpoint(
    model: RoadGAT,
    save_path: str,
    *,
    mode: str,
    training_summary: Optional[dict] = None,
    gate_config: Optional[dict] = None,
) -> None:
    payload = {
        "meta": {
            "version": "gat_v2",
            "mode": mode,
            "edge_ids": EDGES,
            "edge_count": N_EDGES,
            "feature_dim": FEATURE_DIM,
            "adj_shape": list(build_line_graph_adj().shape),
            "hidden": model.hidden,
            "heads": model.heads,
            "dropout": model.dropout,
            "gate_config": gate_config or {},
            "training_summary": training_summary or {},
        },
        "model_state": model.state_dict(),
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(payload, save_path)


def describe_gat_checkpoint(model_path: str = DEFAULT_ALWAYS_GAT_PATH) -> dict:
    status = {
        "path": model_path,
        "exists": os.path.exists(model_path),
        "aligned": False,
        "edge_count": N_EDGES,
        "feature_dim": FEATURE_DIM,
        "message": "checkpoint missing",
        "mode": None,
    }
    if not status["exists"]:
        return status
    try:
        payload = torch.load(model_path, map_location="cpu")
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        ckpt_edges = meta.get("edge_ids", [])
        ckpt_feature_dim = meta.get("feature_dim")
        aligned = ckpt_edges == EDGES and ckpt_feature_dim == FEATURE_DIM
        status.update(
            {
                "aligned": aligned,
                "message": "96-edge aligned checkpoint" if aligned else "checkpoint meta mismatch",
                "mode": meta.get("mode"),
                "checkpoint_edge_count": len(ckpt_edges),
                "checkpoint_feature_dim": ckpt_feature_dim,
                "training_summary": meta.get("training_summary", {}),
            }
        )
    except Exception as exc:
        status["message"] = f"checkpoint error: {exc}"
    return status


_CACHE: dict[tuple[str, str], tuple[RoadGAT, dict]] = {}


def load_gat_model(
    model_path: str = DEFAULT_ALWAYS_GAT_PATH,
    *,
    device: str = "cpu",
) -> tuple[RoadGAT, dict]:
    cache_key = (model_path, device)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    model = RoadGAT().to(device)
    meta = {
        "mode": "always",
        "gate_config": {},
        "training_summary": {},
    }
    if os.path.exists(model_path):
        payload = torch.load(model_path, map_location=device)
        ckpt_meta = payload.get("meta", {})
        if ckpt_meta.get("edge_ids") != EDGES or ckpt_meta.get("feature_dim") != FEATURE_DIM:
            raise RuntimeError(
                f"GAT checkpoint 与当前96边图结构不一致: {model_path}"
            )
        model.load_state_dict(payload["model_state"], strict=True)
        meta = ckpt_meta
    model.eval()
    _CACHE[cache_key] = (model, meta)
    return model, meta


def smooth_weights(
    llm_weight_dict: Dict[str, float],
    model_path: str = DEFAULT_ALWAYS_GAT_PATH,
    *,
    alpha: float | None = None,
    device: str = "cpu",
    edge_confidence: Optional[Dict[str, float]] = None,
    protected_edges: Optional[list[str] | set[str]] = None,
    parse_confidence: float | None = None,
    scene_type: str | None = None,
    direction_conflict: bool = False,
    mode: str | None = None,
    return_metadata: bool = False,
):
    model, checkpoint_meta = load_gat_model(model_path=model_path, device=device)
    features, raw_vec, parsed_mask = build_feature_tensor(llm_weight_dict, edge_confidence=edge_confidence)
    protected_set = set(protected_edges or [])
    with torch.no_grad():
        pred_vec = model(features.to(device)).cpu()

    conf_vec = torch.zeros(N_EDGES)
    if edge_confidence:
        for eid, conf in edge_confidence.items():
            if eid in EDGE_IDX:
                conf_vec[EDGE_IDX[eid]] = float(conf)

    protected_mask = torch.tensor([eid in protected_set or raw_vec[EDGE_IDX[eid]] >= 9.0 for eid in EDGES], dtype=torch.bool)
    high_conf_mask = parsed_mask & ((conf_vec >= 0.82) | protected_mask)
    low_conf_mask = parsed_mask & ~high_conf_mask
    missing_mask = ~parsed_mask

    run_mode = mode or checkpoint_meta.get("mode", "always")
    used_alpha = float(alpha) if alpha is not None else 0.62
    if run_mode == "gated":
        used_alpha = gate_alpha(
            parse_confidence=float(parse_confidence or 0.0),
            protected_edges=int(protected_mask.sum().item()),
            direction_conflict=direction_conflict,
            scene_type=scene_type,
        )

    refined = pred_vec.clone()
    if run_mode == "always":
        refined[parsed_mask] = used_alpha * raw_vec[parsed_mask] + (1.0 - used_alpha) * pred_vec[parsed_mask]
    else:
        refined[high_conf_mask] = raw_vec[high_conf_mask]
        refined[low_conf_mask] = used_alpha * raw_vec[low_conf_mask] + (1.0 - used_alpha) * pred_vec[low_conf_mask]
    refined = refined.clamp(0.3, 10.0)

    result = {eid: round(float(refined[idx].item()), 2) for idx, eid in enumerate(EDGES)}
    for eid in protected_set:
        if eid in llm_weight_dict:
            result[eid] = max(result.get(eid, llm_weight_dict[eid]), float(llm_weight_dict[eid]))

    metadata = {
        "gat_mode": "always_topology_completion" if run_mode == "always" else "gated_topology_completion",
        "high_conf_edges": [eid for eid in EDGES if high_conf_mask[EDGE_IDX[eid]]],
        "low_conf_edges": [eid for eid in EDGES if low_conf_mask[EDGE_IDX[eid]]],
        "missing_edges": [eid for eid in EDGES if missing_mask[EDGE_IDX[eid]]],
        "protected_edges": sorted(set(protected_set)),
        "alpha_used": round(used_alpha, 3),
        "scene_type": scene_type,
        "parse_confidence": float(parse_confidence or 0.0),
    }
    if return_metadata:
        return result, used_alpha, metadata
    return result, used_alpha


def get_attention_matrix(
    llm_weight_dict: Dict[str, float],
    *,
    model_path: str = DEFAULT_ALWAYS_GAT_PATH,
    head: int = 0,
) -> np.ndarray:
    model, _ = load_gat_model(model_path=model_path)
    features, _, _ = build_feature_tensor(llm_weight_dict)
    with torch.no_grad():
        sample = features
        n_nodes = sample.size(0)
        heads = model.gat1.heads
        dim = model.gat1.out_dim
        proj = model.gat1.proj(sample).view(n_nodes, heads, dim)
        left = proj.unsqueeze(1).expand(n_nodes, n_nodes, heads, dim)
        right = proj.unsqueeze(0).expand(n_nodes, n_nodes, heads, dim)
        pair = torch.cat([left, right], dim=-1)
        score = model.gat1.leaky_relu((pair * model.gat1.attn).sum(dim=-1))
        mask = (model.adj == 0).unsqueeze(-1).expand_as(score)
        score = score.masked_fill(mask, float("-inf"))
        alpha = F.softmax(score, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
    return alpha[:, :, head].cpu().numpy()


if __name__ == "__main__":
    print(json.dumps(describe_gat_checkpoint(DEFAULT_ALWAYS_GAT_PATH), ensure_ascii=False, indent=2))
