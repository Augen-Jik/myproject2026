#!/usr/bin/env python3
"""Train 96-edge aligned Always-GAT-v2 and export Gated-GAT-v2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CODE_DIR = "/root/autodl-tmp/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from gat_smoother import (  # noqa: E402
    DEFAULT_ALWAYS_GAT_PATH,
    DEFAULT_GATED_GAT_PATH,
    EDGES,
    EDGE_IDX,
    N_EDGES,
    RoadGAT,
    build_feature_tensor,
    gate_alpha,
    save_gat_checkpoint,
)
from gat_v2_defaults import DEFAULT_SPARSE_UPSTREAM  # noqa: E402
from roadnet_meta import NORMAL_DEFAULT_WEIGHT  # noqa: E402
from sparse_utils import parse_sparse_output_bundle  # noqa: E402


def load_rows(parquet_dir: str, limit: int | None = None) -> list[dict]:
    table = pq.read_table(os.path.join(parquet_dir, "data.parquet"))
    rows = table.to_pylist()
    return rows[:limit] if limit else rows


def build_sparse_cache(
    *,
    rows: list[dict],
    tokenizer,
    model,
    device: str,
    cache_path: str | None = None,
) -> dict[str, torch.Tensor | list]:
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    features_list = []
    targets_list = []
    raw_list = []
    parsed_mask_list = []
    conf_list = []
    parse_conf_list = []
    protected_list = []
    scene_type_list = []

    for idx, row in enumerate(rows, 1):
        messages = json.loads(row["messages"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": messages[0]["content"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False).to(device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
        bundle = parse_sparse_output_bundle(decoded, scene_type=row.get("scene_type"))
        edge_confidence = {
            eid: conf * max(bundle.get("parse_confidence", 0.0), 0.35)
            for eid, conf in bundle.get("edge_confidence", {}).items()
        }
        features, raw_vec, parsed_mask = build_feature_tensor(bundle.get("mapped_weights", {}), edge_confidence=edge_confidence)
        target = torch.tensor([json.loads(row["ground_truth_json"]).get(eid, NORMAL_DEFAULT_WEIGHT) for eid in EDGES], dtype=torch.float32)

        features_list.append(features)
        targets_list.append(target)
        raw_list.append(raw_vec)
        parsed_mask_list.append(parsed_mask.float())
        conf_list.append(torch.tensor([edge_confidence.get(eid, 0.0) for eid in EDGES], dtype=torch.float32))
        parse_conf_list.append(float(bundle.get("parse_confidence", 0.0)))
        protected_list.append(torch.tensor([1.0 if eid in bundle.get("protected_edges", []) else 0.0 for eid in EDGES], dtype=torch.float32))
        scene_type_list.append(row["scene_type"])

        if idx % 50 == 0:
            print(f"  sparse-cache {idx}/{len(rows)}")

    cache = {
        "features": torch.stack(features_list),
        "targets": torch.stack(targets_list),
        "raw_vecs": torch.stack(raw_list),
        "parsed_masks": torch.stack(parsed_mask_list),
        "conf_vecs": torch.stack(conf_list),
        "parse_confidence": torch.tensor(parse_conf_list, dtype=torch.float32),
        "protected_masks": torch.stack(protected_list),
        "scene_types": scene_type_list,
    }
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(cache, cache_path)
    return cache


def evaluate_model(model: RoadGAT, cache: dict, *, mode: str, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(cache["features"].to(device)).cpu()

    raw = cache["raw_vecs"]
    parsed_masks = cache["parsed_masks"].bool()
    conf_vecs = cache["conf_vecs"]
    parse_conf = cache["parse_confidence"]
    protected = cache["protected_masks"].bool()
    targets = cache["targets"]

    always_out = pred.clone()
    gated_out = pred.clone()

    for idx in range(pred.size(0)):
        alpha = gate_alpha(
            parse_confidence=float(parse_conf[idx].item()),
            protected_edges=int(protected[idx].sum().item()),
            direction_conflict=False,
            scene_type=cache["scene_types"][idx],
        )
        always_out[idx][parsed_masks[idx]] = 0.62 * raw[idx][parsed_masks[idx]] + 0.38 * pred[idx][parsed_masks[idx]]

        high_conf = parsed_masks[idx] & ((conf_vecs[idx] >= 0.82) | protected[idx])
        low_conf = parsed_masks[idx] & ~high_conf
        gated_out[idx][high_conf] = raw[idx][high_conf]
        gated_out[idx][low_conf] = alpha * raw[idx][low_conf] + (1 - alpha) * pred[idx][low_conf]

    out = always_out if mode == "always" else gated_out
    mae = torch.mean(torch.abs(out - targets)).item()
    return {"weight_mae": round(float(mae), 4)}


def train_model(train_cache: dict, eval_cache: dict, *, epochs: int, batch_size: int, lr: float, device: str) -> tuple[RoadGAT, dict]:
    model = RoadGAT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_mae = float("inf")
    best_epoch = 0

    n_samples = train_cache["features"].size(0)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        steps = 0
        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start:start + batch_size]
            features = train_cache["features"][batch_idx].to(device)
            targets = train_cache["targets"][batch_idx].to(device)
            parsed_masks = train_cache["parsed_masks"][batch_idx].to(device)

            pred = model(features)
            w_missing = torch.where(parsed_masks < 0.5, torch.full_like(targets, 1.9), torch.ones_like(targets))
            w_extreme = torch.where((targets >= 7.5) | (targets <= 1.3), torch.full_like(targets, 2.8), torch.ones_like(targets))
            weight = torch.maximum(w_missing, w_extreme)
            loss = (((pred - targets) ** 2) * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            nn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        eval_stats = evaluate_model(model, eval_cache, mode="always", device=device)
        print(
            f"epoch {epoch}/{epochs} loss={total_loss/max(steps,1):.4f} "
            f"eval_mae={eval_stats['weight_mae']:.4f}"
        )
        if eval_stats["weight_mae"] < best_mae:
            best_mae = eval_stats["weight_mae"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    summary = {
        "best_epoch": best_epoch,
        "best_eval_mae": round(float(best_mae), 4),
        "train_samples": int(train_cache["features"].size(0)),
        "eval_samples": int(eval_cache["features"].size(0)),
    }
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_SPARSE_UPSTREAM)
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--eval-limit", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--always-output", default=DEFAULT_ALWAYS_GAT_PATH)
    parser.add_argument("--gated-output", default=DEFAULT_GATED_GAT_PATH)
    parser.add_argument("--cache-root", default="/root/autodl-tmp/cache/gat_v2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32),
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    train_rows = load_rows(os.path.join(args.dataset_root, "train"), limit=args.train_limit)
    eval_rows = load_rows(os.path.join(args.dataset_root, "eval"), limit=args.eval_limit)
    t0 = time.time()
    train_cache = build_sparse_cache(
        rows=train_rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        cache_path=os.path.join(args.cache_root, f"train_{len(train_rows)}.pt"),
    )
    eval_cache = build_sparse_cache(
        rows=eval_rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        cache_path=os.path.join(args.cache_root, f"eval_{len(eval_rows)}.pt"),
    )
    print(f"sparse cache ready in {time.time()-t0:.1f}s")

    gat_model, summary = train_model(
        train_cache,
        eval_cache,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    always_eval = evaluate_model(gat_model, eval_cache, mode="always", device=device)
    gated_eval = evaluate_model(gat_model, eval_cache, mode="gated", device=device)
    summary.update(
        {
            "base_model_path": args.model_path,
            "dataset_root": args.dataset_root,
            "graph_edge_count": N_EDGES,
            "graph_feature_dim": 6,
            "always_eval_mae": always_eval["weight_mae"],
            "gated_eval_mae": gated_eval["weight_mae"],
        }
    )

    save_gat_checkpoint(
        gat_model,
        args.always_output,
        mode="always",
        training_summary=summary,
    )
    save_gat_checkpoint(
        gat_model,
        args.gated_output,
        mode="gated",
        training_summary=summary,
        gate_config={
            "high_conf_threshold": 0.82,
            "alpha_schedule": "rule_based_parse_confidence",
            "protected_edge_boost": True,
            "scene_type_adjustment": True,
        },
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
