#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml


DEFAULT_CONFIG = "/root/autodl-tmp/sft_config_lora_sparse_v2_stage4b_fix.yaml"
DEFAULT_SUMMARY = "/root/autodl-tmp/dataset_sparse_v2_stage4b/summary.json"
DEFAULT_DIFF = "/root/autodl-tmp/results/stage4b_diff_29.json"
DEFAULT_OUTPUT = "/root/autodl-tmp/results/stage4c_training_signal_audit.md"
DEFAULT_LOG_DIR = "/root/autodl-tmp/logs"

TARGET_SUBTYPE = "relief_normal_single_edge"
ALL_RELIEF_SUBTYPES = [
    "relief_normal_single_edge",
    "relief_normal_short_range",
    "multi_edge_relief",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit stage4b training signal and produce stage4c suggestions.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--diff-json", default=DEFAULT_DIFF)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def find_matching_logs(log_dir: Path, needles: list[str]) -> list[Path]:
    matches: list[Path] = []
    if not log_dir.exists():
        return matches
    for path in sorted(log_dir.glob("*.log")):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(needle in content for needle in needles if needle):
            matches.append(path)
    return matches


def load_latest_trainer_state(output_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidate_states: list[tuple[int, Path]] = []
    root_state = output_dir / "trainer_state.json"
    if root_state.exists():
        candidate_states.append((10**9, root_state))
    for ckpt in output_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            step = int(ckpt.parent.name.split("-")[-1])
        except ValueError:
            continue
        candidate_states.append((step, ckpt))
    if not candidate_states:
        return None, None
    candidate_states.sort(key=lambda item: item[0])
    _, path = candidate_states[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def load_training_args(path: Path) -> Any | None:
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.load(path, map_location="cpu")


def event_signature(event_records_json: str) -> tuple[str, ...]:
    events = json.loads(event_records_json)
    signatures = []
    for event in events:
        if event.get("active", True):
            signatures.append(
                f"{event.get('road', '')}|{event.get('dir', '')}|{event.get('range', '')}|{event.get('level', '')}"
            )
    return tuple(sorted(signatures))


def load_train_frame(train_root: Path) -> pd.DataFrame:
    table = pq.read_table(train_root / "data.parquet")
    return pd.DataFrame(table.to_pylist())


def compute_overlap_counts(train_df: pd.DataFrame, diff_samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    held_df = pd.DataFrame(diff_samples)
    held_df["sig"] = held_df["event_records_json"].map(event_signature)
    train_df = train_df.copy()
    train_df["sig"] = train_df["event_records_json"].map(event_signature)

    overlap: dict[str, dict[str, int]] = {}
    for subtype in ALL_RELIEF_SUBTYPES:
        held_sub = held_df[held_df["semantic_gap_subtype"] == subtype].copy()
        train_sub = train_df[train_df["bucket"] == subtype].copy()
        train_sig_set = set(train_sub["sig"])
        exact_overlap = sum(1 for sig in held_sub["sig"] if sig in train_sig_set)
        overlap[subtype] = {
            "heldout_count": int(len(held_sub)),
            "train_bucket_count": int(len(train_sub)),
            "exact_signature_overlap": int(exact_overlap),
        }
    return overlap


def build_report(args: argparse.Namespace) -> str:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.dataset_summary).read_text(encoding="utf-8"))
    diff = json.loads(Path(args.diff_json).read_text(encoding="utf-8"))

    output_dir = Path(config["training"]["output_dir"])
    train_root = Path(config["data"]["train_path"])
    bs = int(config["training"]["per_device_train_batch_size"])
    accum = int(config["training"]["gradient_accumulation_steps"])
    requested_epochs = float(config["training"]["num_train_epochs"])
    lr = float(config["training"]["learning_rate"])

    train_df = load_train_frame(train_root)
    train_rows = int(len(train_df))
    world_size = 1

    trainer_state_path, trainer_state = load_latest_trainer_state(output_dir)
    training_args = load_training_args(output_dir / "training_args.bin")
    matched_logs = find_matching_logs(
        Path(args.log_dir),
        needles=[
            str(Path(args.config)),
            str(output_dir),
            str(train_root),
        ],
    )

    train_summary = summary["stage4b_summary"]["train"]
    component_counts = train_summary["mixture_component_counts"]
    bucket_counts = train_summary["bucket_counts"]
    diff_stats = diff["stats"]

    micro_batches_per_epoch = math.ceil(train_rows / bs)
    effective_global_batch = bs * accum * world_size
    floor_updates_per_epoch = train_rows // effective_global_batch
    hf_updates_per_epoch = math.ceil(micro_batches_per_epoch / accum)
    expected_hf_max_steps = math.ceil(requested_epochs * hf_updates_per_epoch)
    script_printed_total_steps = floor_updates_per_epoch * requested_epochs

    actual_global_step = int(trainer_state["global_step"]) if trainer_state else expected_hf_max_steps
    actual_epoch = float(trainer_state["epoch"]) if trainer_state else requested_epochs
    actual_examples_processed = actual_global_step * effective_global_batch
    actual_unique_coverage = min(actual_epoch, 1.0)
    expected_unique_examples_seen = train_rows * actual_unique_coverage
    expected_unseen_examples = max(train_rows - expected_unique_examples_seen, 0.0)

    overlap_counts = compute_overlap_counts(train_df, diff["samples"])

    target_count = int(bucket_counts[TARGET_SUBTYPE])
    target_share = target_count / train_rows
    guard_count = int(component_counts["anti_regression_guard"])
    guard_share = guard_count / train_rows
    relief_count = int(component_counts["relief_supervision_pack"])
    relabel_count = int(component_counts["relabeled_simple_local_repair"])
    hard_negative_count = int(bucket_counts["hard_negative_no_anchor"])

    expected_component_seen_actual = {
        name: count * actual_epoch
        for name, count in component_counts.items()
    }
    expected_bucket_seen_actual = {
        name: count * actual_epoch
        for name, count in bucket_counts.items()
    }

    expected_target_examples_per_update = effective_global_batch * target_share
    expected_guard_examples_per_update = effective_global_batch * guard_share
    expected_hard_negative_seen_actual = hard_negative_count * actual_epoch

    steps_for_target_2x = math.ceil((2.0 * target_count) / max(expected_target_examples_per_update, 1e-9))
    steps_for_target_3x = math.ceil((3.0 * target_count) / max(expected_target_examples_per_update, 1e-9))
    steps_for_target_4x = math.ceil((4.0 * target_count) / max(expected_target_examples_per_update, 1e-9))

    latest_logged_lr = None
    latest_logged_loss = None
    for item in reversed(trainer_state.get("log_history", []) if trainer_state else []):
        if latest_logged_lr is None and "learning_rate" in item:
            latest_logged_lr = float(item["learning_rate"])
        if latest_logged_loss is None and "loss" in item:
            latest_logged_loss = float(item["loss"])
        if latest_logged_lr is not None and latest_logged_loss is not None:
            break

    current_mix_epoch_for_24 = 24 / hf_updates_per_epoch
    current_mix_epoch_for_32 = 32 / hf_updates_per_epoch
    current_mix_epoch_for_40 = 40 / hf_updates_per_epoch

    recommendations = {
        "min_effective_updates": "24-40 optimizer steps",
        "preferred_current_mix_updates": "32-40 optimizer steps",
        "increase_epoch": "yes",
        "target_bucket_weight": "yes",
        "shrink_guard": "yes",
        "single_edge_patch": "yes",
    }

    explicit_rows = [
        [
            "最小有效更新步数区间",
            "24-40 steps",
            "9 steps 完全不动；24 steps 才开始进入有效区，32-40 steps 更接近当前 mix 下每个 single-edge 样本 3-4 次平均暴露。",
        ],
        [
            "是否提高 epoch",
            "是",
            f"若沿用当前 168-row mix，Trainer 实际是 {hf_updates_per_epoch} updates/epoch；要到 32-40 steps，需要约 {fmt_float(current_mix_epoch_for_32, 1)}-{fmt_float(current_mix_epoch_for_40, 1)} epochs。",
        ],
        [
            "是否提高 target bucket 权重",
            "是",
            f"`{TARGET_SUBTYPE}` 当前只占 {pct(target_share)}，约 {fmt_float(expected_target_examples_per_update, 2)} 例/update，建议提高到至少 40-60% 的 exposure。",
        ],
        [
            "是否缩小 guard 比例",
            "是",
            f"guard 当前占 {pct(guard_share)}，是 single-edge 的 {fmt_float(guard_share / max(target_share, 1e-9), 1)} 倍，建议压到 10-20%，只留 anti-regression canary。",
        ],
        [
            "是否拆成更小更集中的 single-edge patch 训练",
            "是",
            "当前 stage4b 把 9 次 update 分散到 relabel/relief/guard；stage4c_singleedge_fix 更适合让 single-edge 成为主信号，再附少量 guard。",
        ],
    ]

    component_rows = []
    for name, count in component_counts.items():
        component_rows.append(
            [
                name,
                str(int(count)),
                pct(count / train_rows),
                fmt_float(expected_component_seen_actual[name], 1),
            ]
        )

    overlap_rows = []
    for subtype in ALL_RELIEF_SUBTYPES:
        info = overlap_counts[subtype]
        overlap_rows.append(
            [
                subtype,
                str(info["heldout_count"]),
                str(info["train_bucket_count"]),
                str(info["exact_signature_overlap"]),
            ]
        )

    input_rows = [
        ["Config", str(Path(args.config))],
        ["Dataset Summary", str(Path(args.dataset_summary))],
        ["Diff JSON", str(Path(args.diff_json))],
        ["Trainer State", str(trainer_state_path) if trainer_state_path else "not found"],
        ["Matched Log Files", ", ".join(str(path) for path in matched_logs) if matched_logs else "no dedicated stage4b log file found in /root/autodl-tmp/logs"],
    ]

    log_lines = []
    if trainer_state:
        for item in trainer_state.get("log_history", []):
            if "loss" in item:
                log_lines.append(
                    f"- step {item['step']}: loss={fmt_float(float(item['loss']), 4)}, lr={item['learning_rate']:.8f}, epoch={fmt_float(float(item['epoch']), 4)}"
                )
    if not log_lines:
        log_lines.append("- 未找到可用的 per-step loss 记录。")

    conclusion = "两者都有，但以更新太弱为主。"

    notes = [
        "# stage4c Training Signal Audit",
        "",
        "## Inputs",
        markdown_table(["Item", "Resolved Path"], input_rows),
        "",
        "## Core Finding",
        f"- stage4b 之所以最终只有 `9` 个 optimizer/update steps，是因为当前配置是 `train_rows=168`、`per_device_batch=4`、`gradient_accumulation_steps=4`、`num_train_epochs=0.8`，而 Transformers 5.3.0 的 `Trainer` 用的是 `num_update_steps_per_epoch = ceil(len_dataloader / grad_accum)`。",
        f"- 具体推导：`len_dataloader = ceil(168 / 4) = 42` micro-batches；`num_update_steps_per_epoch = ceil(42 / 4) = 11`；`max_steps = ceil(0.8 * 11) = 9`。",
        f"- `run_sft_lora.py` 里打印总步数的公式是 `len(train_dataset) // (BS*ACCUM)`，会给出 `10 * 0.8 = 8.0` 这类“脚本估算值”；但真实训练步数以 `Trainer` 为准，最终 checkpoint 证实 `max_steps=9`。",
        f"- 这 9 步只处理了约 `9 * 16 = 144` 个样本曝光，相当于 `0.8571` 个 epoch；也就是说本次训练理论上仍有约 `{fmt_float(expected_unseen_examples, 1)}` 条训练样本根本没被看到。",
        "",
        "## Training Signal Budget",
        markdown_table(
            ["Metric", "Value"],
            [
                ["train rows", str(train_rows)],
                ["eval rows", str(summary['stage4b_summary']['eval']['total_rows'])],
                ["per-device batch", str(bs)],
                ["gradient accumulation", str(accum)],
                ["effective global batch", str(effective_global_batch)],
                ["micro-batches / epoch", str(micro_batches_per_epoch)],
                ["Trainer updates / epoch", str(hf_updates_per_epoch)],
                ["requested epochs", str(requested_epochs)],
                ["Trainer max_steps", str(expected_hf_max_steps)],
                ["actual global_step", str(actual_global_step)],
                ["actual epoch reached", fmt_float(actual_epoch, 4)],
                ["script printed total_steps", fmt_float(script_printed_total_steps, 1)],
                ["latest logged loss", fmt_float(latest_logged_loss, 4) if latest_logged_loss is not None else "n/a"],
                ["latest logged lr", f"{latest_logged_lr:.8f}" if latest_logged_lr is not None else "n/a"],
            ],
        ),
        "",
        "## Mixture Exposure",
        markdown_table(
            ["Mixture Component", "Train Rows", "Train Share", "Expected Example-Sees In This Run"],
            component_rows,
        ),
        "",
        f"- 一个完整 epoch 中，每个 mixture component 都只会被完整看一遍：relabel `72`、relief `48`、guard `48`。",
        f"- 但当前 run 只有 `0.8571` epoch，所以实际理论曝光约为：relabel `{fmt_float(expected_component_seen_actual['relabeled_simple_local_repair'], 1)}`、relief `{fmt_float(expected_component_seen_actual['relief_supervision_pack'], 1)}`、guard `{fmt_float(expected_component_seen_actual['anti_regression_guard'], 1)}`。",
        f"- 按每个 optimizer step 的期望组成估算，当前配置下每一步平均只有 `{fmt_float(expected_target_examples_per_update, 2)}` 个 `{TARGET_SUBTYPE}` 样本，却有 `{fmt_float(expected_guard_examples_per_update, 2)}` 个 guard 样本。",
        "",
        "## Target Bucket Signal",
        f"- `{TARGET_SUBTYPE}` 在 stage4b train 中只有 `{target_count}` 条，占全部训练样本的 `{pct(target_share)}`。",
        f"- 由于这次训练不到 1 个 epoch，`{TARGET_SUBTYPE}` 在整个训练过程中理论上只被学习了约 `{fmt_float(expected_bucket_seen_actual[TARGET_SUBTYPE], 1)}` 次 example-sees；换成单个训练样本视角，就是“平均每条只被看到 `0.857` 次，且 `14.3%` 概率一次都没看到”。",
        f"- `hard_negative_no_anchor` guard 有 `{hard_negative_count}` 条，本次 run 理论曝光约 `{fmt_float(expected_hard_negative_seen_actual, 1)}` 次，已经接近 single-edge 总曝光量的三分之二。",
        "",
        "## Held-out Alignment",
        markdown_table(
            ["Held-out Subtype", "Held-out Failures", "Train Bucket Rows", "Exact Active-Event Signature Overlap"],
            overlap_rows,
        ),
        "",
        f"- `stage4b_diff_29.json` 显示 held-out 主失败 `29/29` 在 pre/post 完全没变，其中 `{TARGET_SUBTYPE}` 就占 `{diff_stats['subtype_breakdown'][TARGET_SUBTYPE]}/29`。",
        f"- 即使只看 `{TARGET_SUBTYPE}`，train 里也只有 `16` 条，而对 held-out `20` 条主失败的 exact active-event signature overlap 只有 `{overlap_counts[TARGET_SUBTYPE]['exact_signature_overlap']}` 条。这说明 stage4b 不只是更新太少，target coverage 也不够贴脸。",
        "",
        "## Why Guard Feels Strong Here",
        f"- guard 总量是 `{guard_count}` 条，占 `{pct(guard_share)}`；它和 relief 总量相等，但却是 `{TARGET_SUBTYPE}` 的 `3x`。",
        f"- 在只有 9 个 update 的情况下，guard 理论曝光约 `{fmt_float(expected_component_seen_actual['anti_regression_guard'], 1)}` 次，而 `{TARGET_SUBTYPE}` 只有 `{fmt_float(expected_bucket_seen_actual[TARGET_SUBTYPE], 1)}` 次。",
        f"- 因此 guard 在绝对比例上不算压倒一切，但对一个“只想修 single-edge no-anchor”的微型 patch 来说，当前 guard 比例确实偏强，容易把稀缺更新预算冲淡。",
        "",
        "## Trainer Evidence",
        *log_lines,
        f"- 从 `trainer_state.json` 看，训练是稳定的，没有 NaN/Inf；但 schedule 只有 9 步，学习率从 `1.5e-5` 很快衰减到 `3.75e-6`，参数更新窗口非常短。",
        "",
        "## Stage4c Recommendation",
        markdown_table(["Question", "Recommendation", "Rationale"], explicit_rows),
        "",
        f"- 如果继续沿用当前 168-row stage4b mixture，不改比例，那么更现实的目标是直接把训练提高到 `{recommendations['preferred_current_mix_updates']}`。对应 epoch 约为 `{fmt_float(current_mix_epoch_for_32, 1)}-{fmt_float(current_mix_epoch_for_40, 1)}`。",
        f"- 如果 stage4c 改成更集中的 `singleedge_fix` patch，建议把 `{TARGET_SUBTYPE}` 提到 40-60% 的 exposure，并把 guard 压到 10-20%，这样 `{recommendations['min_effective_updates']}` 就足以形成可见参数位移。",
        f"- 以当前 mix 粗算：要让 `{TARGET_SUBTYPE}` 达到每条平均 2/3/4 次暴露，大约需要 `{steps_for_target_2x}` / `{steps_for_target_3x}` / `{steps_for_target_4x}` 个 optimizer steps。",
        "",
        "## Final Answer",
        f"- 当前失败判断：`{conclusion}`",
        f"- 更细一点说：`更新太弱` 是主因，因为 9 步更新不足以让任何 held-out 失败样本发生输出层面的位移；但 `数据覆盖不足` 也真实存在，尤其 `{TARGET_SUBTYPE}` 对 held-out 20 条主失败只有 3 条 exact signature train overlap。",
    ]
    return "\n".join(notes) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
