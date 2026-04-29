#!/usr/bin/env python3
"""Validate Sparse-LoRA-v2 on normal/special/anti-truncation splits."""

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

from roadnet_meta import NORMAL_DEFAULT_WEIGHT, live_edge_ids  # noqa: E402
from sparse_utils import parse_sparse_output_bundle  # noqa: E402

_DIR_RULE = "DIR_RULE=ANCHOR中DIR字段须与EVENT中DIR完全一致；不得颠倒方向（向东≠向西，向北≠向南）；无明确方向时填写双向。"


def _inject_dir_rule(prompt: str) -> str:
    """Insert DIR_RULE after OUTPUT_RULE line in stored prompt."""
    marker = "OUTPUT_RULE="
    lines = prompt.splitlines()
    out = []
    for line in lines:
        out.append(line)
        if line.startswith(marker) and _DIR_RULE not in prompt:
            out.append(_DIR_RULE)
    return "\n".join(out)


def load_rows(parquet_dir: str, limit: int | None = None) -> list[dict]:
    table = pq.read_table(os.path.join(parquet_dir, "data.parquet"))
    rows = table.to_pylist()
    return rows[:limit] if limit else rows


def mae(pred: dict[str, float], truth: dict[str, float], edge_ids: list[str]) -> float:
    return sum(abs(float(pred.get(eid, NORMAL_DEFAULT_WEIGHT)) - float(truth.get(eid, NORMAL_DEFAULT_WEIGHT))) for eid in edge_ids) / len(edge_ids)


def evaluate_split(model, tokenizer, device: str, split_dir: str, edge_ids: list[str], limit: int | None = None) -> dict:
    rows = load_rows(split_dir, limit=limit)
    total_fail = 0
    total_ratio = 0.0
    total_mae = 0.0
    total_conf = 0.0
    total_time = 0.0
    preview_rows = []

    for row in rows:
        messages = json.loads(row["messages"])
        user_content = _inject_dir_rule(messages[0]["content"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False).to(device)
        in_len = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        infer_time = time.time() - t0
        decoded = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
        bundle = parse_sparse_output_bundle(decoded, scene_type=row.get("scene_type"))
        pred_weights = {eid: NORMAL_DEFAULT_WEIGHT for eid in edge_ids}
        pred_weights.update(bundle.get("mapped_weights", {}))
        truth = json.loads(row["ground_truth_json"])

        total_fail += int(bundle.get("parse_fail", False))
        total_ratio += len(bundle.get("mapped_weights", {})) / len(edge_ids)
        total_mae += mae(pred_weights, truth, edge_ids)
        total_conf += bundle.get("parse_confidence", 0.0)
        total_time += infer_time
        if len(preview_rows) < 3:
            preview_rows.append(
                {
                    "scene_type": row["scene_type"],
                    "length_bucket": row["length_bucket"],
                    "anchor_count": bundle.get("anchor_count", 0),
                    "parse_confidence": round(bundle.get("parse_confidence", 0.0), 3),
                    "parse_fail": bundle.get("parse_fail", False),
                }
            )

    count = max(len(rows), 1)
    return {
        "count": len(rows),
        "parse_fail_rate": round(total_fail / count, 4),
        "parsed_edge_ratio": round(total_ratio / count, 4),
        "weight_mae": round(total_mae / count, 4),
        "parse_confidence": round(total_conf / count, 4),
        "inference_time": round(total_time / count, 4),
        "preview": preview_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/model_merged_sparse_v2_stage4_fix")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="/root/autodl-tmp/results/sparse_v2_validation.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32),
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    edge_ids = live_edge_ids()
    results = {}
    for split in ("val_normal", "val_special", "val_anti_truncation"):
        results[split] = evaluate_split(model, tokenizer, device, os.path.join(args.dataset_root, split), edge_ids, limit=args.limit)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
