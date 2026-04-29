#!/usr/bin/env python3
"""Strict validation for Sparse-LoRA-v2."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

CODE_DIR = "/root/autodl-tmp/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from sparse_eval_strict import (  # noqa: E402
    build_report,
    evaluate_rows,
    load_model,
    load_rows,
    load_tokenizer,
    unload_model,
    write_detail_csv,
    write_summary_csv,
)
from roadnet_meta import live_edge_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/autodl-tmp/model_merged_sparse_v2_stage4_fix")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/dataset_sparse_v2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default="/root/autodl-tmp/results/sparse_v2_validation_strict.json")
    parser.add_argument("--output-summary-csv", default="/root/autodl-tmp/results/sparse_v2_validation_strict_summary.csv")
    parser.add_argument("--output-detail-csv", default="/root/autodl-tmp/results/sparse_v2_validation_strict_details.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, device)
    edge_ids = live_edge_ids()

    split_details = {}
    all_details = []
    try:
        for split in ("val_normal", "val_special", "val_anti_truncation"):
            rows = load_rows(os.path.join(args.dataset_root, split), limit=args.limit)
            details = evaluate_rows(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=rows,
                split=split,
                edge_ids=edge_ids,
            )
            split_details[split] = details
            all_details.extend(details)
    finally:
        unload_model(model)

    report = build_report(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        split_details=split_details,
    )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_summary_csv(report, args.output_summary_csv)
    write_detail_csv(all_details, args.output_detail_csv)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
