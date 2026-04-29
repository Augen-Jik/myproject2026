#!/usr/bin/env python3
"""Consistency audit for Sparse-LoRA-v2 adapter/merged/validation chains."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CODE_DIR = "/root/autodl-tmp/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from roadnet_meta import NORMAL_DEFAULT_WEIGHT, live_edge_ids  # noqa: E402
from sparse_utils import parse_sparse_output_bundle  # noqa: E402


def load_rows(parquet_dir: str, limit: int | None = None) -> list[dict]:
    table = pq.read_table(os.path.join(parquet_dir, "data.parquet"))
    rows = table.to_pylist()
    return rows[:limit] if limit else rows


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_merged_model(model_path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32),
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def load_adapter_model(base_path: str, adapter_path: str, device: str):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32),
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=base_model.dtype)
    model.eval()
    return model


def unload_model(model) -> None:
    try:
        del model
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def mae(pred: dict[str, float], truth: dict[str, float], edge_ids: list[str]) -> float:
    return sum(abs(float(pred.get(eid, NORMAL_DEFAULT_WEIGHT)) - float(truth.get(eid, NORMAL_DEFAULT_WEIGHT))) for eid in edge_ids) / len(edge_ids)


def abnormal_edges(weights: dict[str, float], *, limit: int = 12) -> list[dict]:
    items = []
    for edge_id, value in weights.items():
        if abs(float(value) - NORMAL_DEFAULT_WEIGHT) > 1e-9:
            items.append(
                {
                    "edge_id": edge_id,
                    "weight": round(float(value), 2),
                    "delta_from_default": round(float(value) - NORMAL_DEFAULT_WEIGHT, 2),
                }
            )
    items.sort(key=lambda item: (-abs(item["delta_from_default"]), item["edge_id"]))
    return items[:limit]


def bundle_view(bundle: dict) -> dict:
    return {
        "scene_type": bundle.get("scene_type"),
        "anchor_count": int(bundle.get("anchor_count", 0)),
        "anchors": bundle.get("anchors", []),
        "mapped_edge_count": len(bundle.get("mapped_weights", {})),
        "mapped_edges_preview": [
            {"edge_id": edge_id, "weight": bundle["mapped_weights"][edge_id]}
            for edge_id in sorted(bundle.get("mapped_weights", {}))[:12]
        ],
        "parse_confidence": round(float(bundle.get("parse_confidence", 0.0)), 3),
        "parse_fail": bool(bundle.get("parse_fail", False)),
        "no_anchor": int(bundle.get("anchor_count", 0)) == 0 and not bool(bundle.get("parse_fail", False)),
        "protected_edges": bundle.get("protected_edges", []),
        "conflicts": bundle.get("conflicts", []),
    }


def generate_validate_chain(model, tokenizer, prompt_text: str, scene_type: str | None, device: str) -> dict:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
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
    raw_generation = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
    bundle = parse_sparse_output_bundle(raw_generation, scene_type=scene_type)
    return {
        "prompt": prompt,
        "raw_generation": raw_generation,
        "bundle": bundle,
        "infer_time_s": round(infer_time, 4),
    }


def generate_stage3_smoke_chain(model, tokenizer, prompt_text: str, scene_type: str | None, device: str) -> dict:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1800,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    infer_time = time.time() - t0
    full_decoded = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    parse_target = full_decoded.split("</think>")[-1].strip() if "</think>" in full_decoded else full_decoded
    bundle = parse_sparse_output_bundle(full_decoded, scene_type=scene_type)
    return {
        "prompt": prompt,
        "raw_generation": full_decoded,
        "parse_target": parse_target,
        "bundle": bundle,
        "infer_time_s": round(infer_time, 4),
    }


def build_dense_prediction(bundle: dict, edge_ids: list[str]) -> dict[str, float]:
    pred = {edge_id: NORMAL_DEFAULT_WEIGHT for edge_id in edge_ids}
    pred.update(bundle.get("mapped_weights", {}))
    return pred


def pick_audit_specs() -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    specs.extend(("val_normal", idx) for idx in range(4))
    specs.extend(("val_special", idx) for idx in range(3))
    specs.extend(("val_anti_truncation", idx) for idx in range(3))
    return specs


def load_audit_samples(dataset_root: str) -> list[dict]:
    rows_by_split = {
        split: load_rows(os.path.join(dataset_root, split), limit=60)
        for split in ("val_normal", "val_special", "val_anti_truncation")
    }
    samples = []
    for split, idx in pick_audit_specs():
        row = rows_by_split[split][idx]
        messages = json.loads(row["messages"])
        truth = json.loads(row["ground_truth_json"])
        samples.append(
            {
                "sample_id": f"{split}:{idx}",
                "split": split,
                "split_index": idx,
                "scene_type": row.get("scene_type"),
                "length_bucket": row.get("length_bucket"),
                "prompt_text": messages[0]["content"],
                "ground_truth_json": row["ground_truth_json"],
                "ground_truth_abnormal_edges": abnormal_edges(truth),
                "ground_truth_abnormal_edge_count": len(abnormal_edges(truth, limit=10_000)),
            }
        )
    return samples


def run_model_audit(samples: list[dict], *, tokenizer_path: str, model_loader, loader_arg: str, device: str, chain_name: str) -> dict[str, dict]:
    tokenizer = load_tokenizer(tokenizer_path)
    model = model_loader(loader_arg, device) if chain_name != "adapter_validate" and chain_name != "adapter_stage3_smoke" else None
    raise RuntimeError("unreachable")


def verify_file_hashes(paths: list[str]) -> dict[str, str]:
    import hashlib

    hashes = {}
    for path in paths:
        data = Path(path).read_bytes()
        hashes[path] = hashlib.sha256(data).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="/root/autodl-tmp/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="/root/autodl-tmp/model_lora_sparse_v2")
    parser.add_argument("--merged-path", default="/root/autodl-tmp/model_merged_sparse_v2_stage4_fix")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--output", default="/root/autodl-tmp/results/sparse_v2_consistency_audit.json")
    parser.add_argument("--validation-output", default="/root/autodl-tmp/results/sparse_v2_validation_60_audit_rerun.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_ids = live_edge_ids()
    samples = load_audit_samples(args.dataset_root)

    adapter_tokenizer = load_tokenizer(args.adapter_path)
    adapter_model = load_adapter_model(args.base_path, args.adapter_path, device)
    adapter_results = {}
    stage3_results = {}
    for sample in samples:
        validate_run = generate_validate_chain(adapter_model, adapter_tokenizer, sample["prompt_text"], sample["scene_type"], device)
        validate_bundle = validate_run["bundle"]
        validate_pred = build_dense_prediction(validate_bundle, edge_ids)
        truth = json.loads(sample["ground_truth_json"])
        adapter_results[sample["sample_id"]] = {
            "raw_generation": validate_run["raw_generation"],
            "prompt_preview": sample["prompt_text"][:220],
            "bundle": bundle_view(validate_bundle),
            "weight_mae": round(mae(validate_pred, truth, edge_ids), 4),
            "pred_abnormal_edges": abnormal_edges(validate_pred),
        }

        smoke_run = generate_stage3_smoke_chain(adapter_model, adapter_tokenizer, sample["prompt_text"], sample["scene_type"], device)
        smoke_bundle = smoke_run["bundle"]
        smoke_pred = build_dense_prediction(smoke_bundle, edge_ids)
        stage3_results[sample["sample_id"]] = {
            "raw_generation": smoke_run["raw_generation"],
            "parse_target": smoke_run["parse_target"],
            "bundle": bundle_view(smoke_bundle),
            "weight_mae": round(mae(smoke_pred, truth, edge_ids), 4),
            "pred_abnormal_edges": abnormal_edges(smoke_pred),
        }
    unload_model(adapter_model)

    merged_tokenizer = load_tokenizer(args.merged_path)
    merged_model = load_merged_model(args.merged_path, device)
    merged_results = {}
    for sample in samples:
        validate_run = generate_validate_chain(merged_model, merged_tokenizer, sample["prompt_text"], sample["scene_type"], device)
        validate_bundle = validate_run["bundle"]
        validate_pred = build_dense_prediction(validate_bundle, edge_ids)
        truth = json.loads(sample["ground_truth_json"])
        merged_results[sample["sample_id"]] = {
            "raw_generation": validate_run["raw_generation"],
            "bundle": bundle_view(validate_bundle),
            "weight_mae": round(mae(validate_pred, truth, edge_ids), 4),
            "pred_abnormal_edges": abnormal_edges(validate_pred),
        }

    rerun_validation = {}
    validation_audit_rows = []
    for split in ("val_normal", "val_special", "val_anti_truncation"):
        rows = load_rows(os.path.join(args.dataset_root, split), limit=60)
        total_fail = 0
        total_ratio = 0.0
        total_mae = 0.0
        total_conf = 0.0
        total_time = 0.0
        no_anchor_count = 0
        no_anchor_mae_sum = 0.0
        gt_changed_edge_sum = 0
        for idx, row in enumerate(rows):
            messages = json.loads(row["messages"])
            truth = json.loads(row["ground_truth_json"])
            run = generate_validate_chain(merged_model, merged_tokenizer, messages[0]["content"], row.get("scene_type"), device)
            bundle = run["bundle"]
            pred = build_dense_prediction(bundle, edge_ids)
            row_mae = mae(pred, truth, edge_ids)
            total_fail += int(bundle.get("parse_fail", False))
            total_ratio += len(bundle.get("mapped_weights", {})) / len(edge_ids)
            total_mae += row_mae
            total_conf += bundle.get("parse_confidence", 0.0)
            total_time += run["infer_time_s"]
            changed_edge_count = len(abnormal_edges(truth, limit=10_000))
            gt_changed_edge_sum += changed_edge_count
            is_no_anchor = int(bundle.get("anchor_count", 0)) == 0 and not bool(bundle.get("parse_fail", False))
            if is_no_anchor:
                no_anchor_count += 1
                no_anchor_mae_sum += row_mae
            if any(sample["sample_id"] == f"{split}:{idx}" for sample in samples):
                validation_audit_rows.append(
                    {
                        "sample_id": f"{split}:{idx}",
                        "split": split,
                        "split_index": idx,
                        "scene_type": row.get("scene_type"),
                        "length_bucket": row.get("length_bucket"),
                        "prompt_text": messages[0]["content"],
                        "ground_truth_json": row["ground_truth_json"],
                        "ground_truth_abnormal_edges": abnormal_edges(truth),
                        "prediction_raw": run["raw_generation"],
                        "parser_output": bundle_view(bundle),
                        "pred_used_abnormal_edges": abnormal_edges(pred),
                        "gt_changed_edge_count": changed_edge_count,
                        "weight_mae": round(row_mae, 4),
                    }
                )
        count = max(len(rows), 1)
        rerun_validation[split] = {
            "count": len(rows),
            "parse_fail_rate": round(total_fail / count, 4),
            "parsed_edge_ratio": round(total_ratio / count, 4),
            "weight_mae": round(total_mae / count, 4),
            "parse_confidence": round(total_conf / count, 4),
            "inference_time": round(total_time / count, 4),
            "no_anchor_rate": round(no_anchor_count / count, 4),
            "mean_mae_on_no_anchor_rows": round(no_anchor_mae_sum / max(no_anchor_count, 1), 4),
            "avg_gt_changed_edge_count": round(gt_changed_edge_sum / count, 4),
        }
    unload_model(merged_model)

    path_consistency = {
        "stage3_final_adapter_path": args.adapter_path,
        "merged_model_path": args.merged_path,
        "validation_script_model_path": "/root/autodl-tmp/model_merged_sparse_v2_stage4_fix",
        "current_reproducible_smoke_chain_path": args.adapter_path,
        "stage3_config_path": "/root/autodl-tmp/sft_config_lora_sparse_v2_stage3.yaml",
        "train_pipeline_script": "/root/autodl-tmp/train_sparse_lora_v2.sh",
        "merge_script": "/root/autodl-tmp/merge_lora.py",
    }

    tokenizer_consistency = {
        "adapter_eos_token": adapter_tokenizer.eos_token,
        "adapter_eos_token_id": adapter_tokenizer.eos_token_id,
        "adapter_pad_token": adapter_tokenizer.pad_token,
        "adapter_pad_token_id": adapter_tokenizer.pad_token_id,
        "merged_eos_token": merged_tokenizer.eos_token,
        "merged_eos_token_id": merged_tokenizer.eos_token_id,
        "merged_pad_token": merged_tokenizer.pad_token,
        "merged_pad_token_id": merged_tokenizer.pad_token_id,
        "file_hashes": verify_file_hashes(
            [
                "/root/autodl-tmp/model_lora_sparse_v2/chat_template.jinja",
                "/root/autodl-tmp/model_merged_sparse_v2_stage4_fix/chat_template.jinja",
                "/root/autodl-tmp/model_lora_sparse_v2/tokenizer.json",
                "/root/autodl-tmp/model_merged_sparse_v2_stage4_fix/tokenizer.json",
                "/root/autodl-tmp/model_lora_sparse_v2/tokenizer_config.json",
                "/root/autodl-tmp/model_merged_sparse_v2_stage4_fix/tokenizer_config.json",
            ]
        ),
    }

    detailed_samples = []
    for sample in samples:
        sample_id = sample["sample_id"]
        detailed_samples.append(
            {
                "sample_id": sample_id,
                "split": sample["split"],
                "split_index": sample["split_index"],
                "scene_type": sample["scene_type"],
                "length_bucket": sample["length_bucket"],
                "input": sample["prompt_text"],
                "ground_truth_abnormal_edges": sample["ground_truth_abnormal_edges"],
                "ground_truth_abnormal_edge_count": sample["ground_truth_abnormal_edge_count"],
                "adapter_validate": adapter_results[sample_id],
                "merged_validate": merged_results[sample_id],
                "stage3_adapter_smoke_style": stage3_results[sample_id],
            }
        )

    output = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "edge_count": len(edge_ids),
        "paths": path_consistency,
        "tokenizer_template_consistency": tokenizer_consistency,
        "metric_formulas": {
            "weight_mae": "mean over samples of mean over all live edges of abs(pred_weight - gt_weight); pred defaults every missing edge to NORMAL_DEFAULT_WEIGHT=2.0",
            "parsed_edge_ratio": "mean over samples of len(bundle.mapped_weights) / len(live_edge_ids)",
            "parse_fail_rate": "mean over samples of int(bundle.parse_fail)",
            "parse_confidence": "mean over samples of bundle.parse_confidence",
            "recall_in_current_60_validation": "not computed by validate_sparse_v2.py",
        },
        "same_sample_comparison": detailed_samples,
        "validation_rerun_60": rerun_validation,
        "validation_audit_rows": validation_audit_rows,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(args.validation_output, "w", encoding="utf-8") as f:
        json.dump(rerun_validation, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
